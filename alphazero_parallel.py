import multiprocessing.connection
import multiprocessing.context

import torch
import multiprocessing, time, sys
import dill
import traceback, gc

from pathlib import Path
from tqdm import tqdm
from players import *

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

from alphazero import AlphaZero, parse_args, build_alphazero
from pit_puct_mcts import pit
from util import *

def execute_episode_worker(
    conn: multiprocessing.connection.Connection,
    rank: str,
    args,
):
    args = dill.loads(args)
    args.seed += rank * 100

    log_file = Path(args.log_file)
    log_file = log_file.with_stem(f"{log_file.stem}_worker{rank}")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    logger.info(f"[Worker {rank}] Initializing worker {rank}")
    st0 = time.time()

    if torch.cuda.is_available():
        num_gpu = torch.cuda.device_count()
        gpu_id = rank % num_gpu
        logger.debug(f"[Worker {rank}] num_gpu={num_gpu} gpu={gpu_id}")
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    alphazero = build_alphazero(args, device)

    logger.info(f"[Worker {rank}] Ready (init_time={time.time()-st0:.3f})")

    def run_episode():
        episode, data = alphazero.execute_episode()
        logger.debug(f"[Worker {rank}] Collected an episode (length={len(episode)})")
        conn.send((episode, data))

    def load_net(target: str, filename: str):
        if target == 'best':
            alphazero.last_net.load_checkpoint(alphazero.checkpoint_dir / filename)
        else:
            alphazero.net.load_checkpoint(alphazero.checkpoint_dir / filename)
        logger.debug(f"[Worker {rank}] Loaded {target} net from {filename}")

    def pit_update(role: str):
        player = alphazero.build_puct_player(alphazero.net)
        opponent = alphazero.build_puct_player(alphazero.last_net)

        if role == "black":
            players = player, opponent
        else:
            players = opponent, player

        reward, data = pit(alphazero.env, *players, log_output=False)
        logger.debug(f"[Worker {rank}] Pit update finished: role={role}, reward={reward}")
        conn.send((reward, data))

    while True:
        try:
            command, *args = conn.recv()
            start_time = time.time()
            logger.debug(f"[Worker {rank}] Received command {command} (args={args})")

            if command == 'close':
                conn.close()
                return

            elif command == 'run_episode':
                run_episode(*args)

            elif command == 'load_net':
                load_net(*args)

            elif command == 'pit_update':
                pit_update(*args)

            elif command == 'resign_stats':
                stats = alphazero.resign_test_stats
                conn.send(stats)
                stats.clear()

            else:
                logger.warning(f"[Worker {rank}] Unknown command: {command}")

            logger.debug(f"[Worker {rank}] Finished {command} in {time.time()-start_time:.3f}s")

        except Exception as e:
            logger.error(e)
            traceback.print_exc()
            logger.warn(f"[Worker {rank}] shutting down worker {rank}")
            if locals().get('conn') and conn.closed == False:
                conn.close()
            exit(1)


class AlphaZeroWorker:
    def __init__(self, ctx: multiprocessing.context.SpawnContext, rank: int, args):
        self.rank = rank
        self.pipe, child_pipe = ctx.Pipe()
        self.worker = ctx.Process(
            target=execute_episode_worker,
            args=(child_pipe, rank, dill.dumps(args, recurse=True)),
        )

    def start(self):
        self.worker.start()

    def send(self, command, *args):
        logger.debug(f"[AlphaZeroWorker:{self.rank}] Sending command {command} (args={args})")
        self.pipe.send((command, *args))

    def recv(self):
        return self.pipe.recv()

    def close(self):
        self.pipe.send(("close"))
        self.worker.join()
        self.worker.close()
        self.pipe.close()


class AlphaZeroParallel(AlphaZero):
    def init_workers(self, n_worker: int, args):
        self.n_worker = n_worker
        assert n_worker > 0

        logger.info(f"[AlphaZeroParallel] Creating {n_worker} workers")

        ctx = torch.multiprocessing.get_context("spawn")
        self.workers = [AlphaZeroWorker(ctx, i, args) for i in range(n_worker)]

        for worker in self.workers:
            worker.start()

        logger.info(f"[AlphaZeroParallel] Started {n_worker} workers")

    def close(self):
        for worker in self.workers:
            worker.close()

    def self_play(self, sgf_file: SGFFile):
        # Send self-play tasks to workers
        tasks = []
        for i in range(self.config.n_match_train):
            worker_id = i % self.n_worker
            self.workers[worker_id].send('run_episode')
            tasks.append(worker_id)

        # Receive self-play results from workers
        stats = PlayerStats()
        for i, worker_id in enumerate(tqdm(tasks, desc="Self Play")):
            episode, data = self.workers[worker_id].recv()
            self.train_examples_queue += episode
            stats.update(episode[0][-1])
            sgf_file.save(f"{i}", data)
        logger.info(f"[NEW TRAIN DATA COLLECTED]: {stats}")

        # Collect resignation test stats from workers
        for worker in self.workers:
            worker.send('resign_stats')
            self.resign_test_stats += worker.recv()
        logger.info(f"[RESIGNATION TEST RESULT]: {self.resign_test_stats}")

    def eval_update(self, sgf_file: SGFFile) -> PlayerStats:
        # Send evaluation tasks to workers
        tasks = []
        for i in range(self.config.n_match_update):
            worker_id = i % self.n_worker
            role, index = ("black", "white")[i % 2], i // 2
            self.workers[worker_id].send('pit_update', role)
            tasks.append((worker_id, role, index))

        # Receive evaluation results from workers
        stats = {role: PlayerStats() for role in ("black", "white")}
        for i, (worker_id, role, index) in enumerate(tqdm(tasks, desc="Update Evaluation")):
            reward, data = self.workers[worker_id].recv()
            stats[role].update(reward if role == "black" else -reward)
            sgf_file.save(f"{role}_{index}", data)

        combined = stats['black'] + stats['white']
        logger.info(f"[EVALUATION RESULT]: {combined}")
        logger.info(f"[EVALUATION RESULT]:(black) {stats['black']}")
        logger.info(f"[EVALUATION RESULT]:(white) {stats['white']}")
        return combined

    def sync_model(self, target: str, filename: str):
        for worker in self.workers:
            worker.send('load_net', target, filename)


if __name__ == "__main__":
    args = parse_args()

    file_handler = logging.FileHandler(args.log_file)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    alphazero: AlphaZeroParallel = build_alphazero(args, device, cls=AlphaZeroParallel)

    n_worker = torch.cuda.device_count() if torch.cuda.is_available() else 1
    alphazero.init_workers(n_worker, args)

    if args.mode == "eval":
        alphazero.eval(args.checkpoint_name)
    else:
        alphazero.learn(last_iter=args.last_iter)

    alphazero.close()
