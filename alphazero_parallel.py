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
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)

from alphazero import AlphaZero, parse_args, build_alphazero
from pit_puct_mcts import pit
from util import *

def execute_episode_worker(
    conn: multiprocessing.connection.Connection,
    rank: str,
    args,
):
    logger.debug(f"[Worker {rank}] Initializing worker {rank}")
    st0 = time.time()

    if torch.cuda.is_available():
        num_gpu = torch.cuda.device_count()
        gpu_id = (rank + 1) % num_gpu
        logger.debug(f"[Worker {rank}] num_gpu={num_gpu} gpu={gpu_id}")
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    args = dill.loads(args)
    args.seed += rank * 100
    args.log_file = str(Path(args.log_file).with_stem(f"{Path(args.log_file).stem}_worker{rank}"))
    alphazero = build_alphazero(args, device, logger=logger)

    logger.debug(f"[Worker {rank}] Ready (init_time={time.time()-st0:.3f})")

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

    def pit_update(role: int):
        player1 = alphazero.build_puct_player(alphazero.net)
        player2 = alphazero.build_puct_player(alphazero.last_net)

        if role == -1:
            player1, player2 = player2, player1

        reward, data = pit(alphazero.env, player1, player2, log_output=False)
        logger.debug(f"[Worker {rank}] Pit update finished: role={role}, reward={reward}")
        conn.send((reward, data))

    while True:
        try:
            command, *args = conn.recv()
            start_time = time.time()
            logger.debug(f"[Worker {rank}] Received command {command}")

            if command == 'close':
                conn.close()
                return
            
            elif command == 'run_episode':
                run_episode(*args)

            elif command == 'load_net':
                load_net(*args)
                
            elif command == 'pit_update':
                pit_update(*args)

            else:
                logger.warning(f"[Worker {rank}] Unknown command: {command}")

            logger.debug(f"[Worker {rank}] Finished in {time.time()-start_time:.3f}s")

        except Exception as e:
            print(e)
            traceback.print_exc()
            print(f"[Worker {rank}] shutting down worker-{rank}")
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
        logger.debug(f"[AlphaZeroWorker:{self.rank}] Sending command {command}")
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
        worker_ids = [i % self.n_worker for i in range(self.config.n_match_train)]

        # Send self-play tasks to workers
        for i in worker_ids:
            self.workers[i].send('run_episode')

        # Receive self-play results from workers
        stats = PlayerStats()
        for i in tqdm(worker_ids, desc="Self Play"):
            episode, data = self.workers[i].recv()
            self.train_eamples_queue += episode
            stats.update(episode[0][-1])
            sgf_file.save(f"{i}", data)
        logger.info(f"[NEW TRAIN DATA COLLECTED]: {stats}")

    def eval_update(self, sgf_file: SGFFile) -> PlayerStats:
        n_match = self.config.n_match_update
        worker_ids = [i % self.n_worker for i in range(n_match)]
        roles = [1] * (n_match//2) + [-1] * (n_match//2)
        tasks = list(zip(worker_ids, roles))

        # Send evaluation tasks to workers
        for i, role in tasks:
            self.workers[i].send('pit_update', role)
        
        # Receive evaluation results from workers
        first_play, second_play = PlayerStats(), PlayerStats()
        for i, role in tqdm(tasks, desc="Evaluation Update"):
            reward, data = self.workers[i].recv()
            if role == 1:
                first_play.update(reward)
                sgf_file.save(f"black_{i}", data)
            else:
                second_play.update(-reward)
                sgf_file.save(f"white_{i}", data)

        combined = first_play + second_play
        logger.info(f"[EVALUATION RESULT]: {combined}")
        logger.info(f"[EVALUATION RESULT]:(first)  {first_play}")
        logger.info(f"[EVALUATION RESULT]:(second) {second_play}")
        return combined
    
    def sync_model(self, target: str, filename: str):
        super().sync_model(target, filename)
        for worker in self.workers:
            worker.send('load_net', target, filename)


if __name__ == "__main__":
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    alphazero: AlphaZeroParallel = build_alphazero(args, device, logger=logger, cls=AlphaZeroParallel)

    n_worker = torch.cuda.device_count() if torch.cuda.is_available() else 1
    alphazero.init_workers(n_worker, args)

    if args.mode == "eval":
        alphazero.eval(args.checkpoint_name)
    else:
        alphazero.learn(last_epoch=args.last_epoch)

    alphazero.close()
