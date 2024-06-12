import multiprocessing.connection
from env.base_env import BaseGame, get_symmetries, ResultCounter
from torch.nn import Module
from model.wrapper import ModelWrapper, ModelTrainingConfig
from model.example_net import MLPNet, ConvNet, MyNet, BaseNetConfig
from mcts import puct_mcts

import torch
import multiprocessing, os, time, sys
import dill
import traceback, gc

import numpy as np
import random
import copy
from tqdm import tqdm
from random import shuffle
from players import *
from players.uct_player import UCTMCTSConfig
from alphazero import AlphaZeroConfig

from pit_puct_mcts import multi_match

import logging
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)

class AlphaZeroParallelConfig(AlphaZeroConfig):
    def __init__(
        self, 
        n_train_iter:int=300,
        n_match_train:int=20,
        n_match_update:int=20,
        n_match_eval:int=20,
        max_queue_length:int=8000,
        update_threshold:float=0.501,
        n_search:int=200, 
        temperature:float=1.0, 
        C:float=1.0,
        checkpoint_path:str="checkpoint"
    ):
        self.n_train_iter = n_train_iter
        self.n_match_train = n_match_train
        self.max_queue_length = max_queue_length
        self.n_match_update = n_match_update
        self.n_match_eval = n_match_eval
        self.update_threshold = update_threshold
        self.n_search = n_search
        self.temperature = temperature
        self.C = C
        
        self.checkpoint_path = checkpoint_path

def execute_episode_worker(
    conn:multiprocessing.connection.Connection, 
    env_builder,
    net_builder, 
    mcts_config:puct_mcts.MCTSConfig, 
    id:int, 
    seed:int,
    checkpoint_path:str="",
    ):
    logger.debug(f"[Worker {id}] Initializing worker {id}")
    st0 = time.time()
    env_builder = dill.loads(env_builder)
    net_builder = dill.loads(net_builder)
    root_env:BaseGame = env_builder()
    if torch.cuda.is_available():
        num_gpu = torch.cuda.device_count()
        gpu_id = (id + 1) % num_gpu
        logger.debug(f"[Worker {id}] num_gpu={num_gpu} id={id} gpu={gpu_id}")
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(gpu_id)
    else:
        device = "cpu"
    # input()
    net = net_builder(device)
    opp_net = net.copy()
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    logger.debug(f"[Worker {id}] Worker {id} is Ready (init_time={time.time()-st0:.3f})")
    while True:
        try:
            command, args = conn.recv()
            
            if command == 'close':
                break
            
            if command == 'run':
                logger.debug(f"[Worker {id}] Start collect {int(args)} episodes")
                st0 = time.time()
                all_examples = []
                all_episode_len = []
                result_counter = ResultCounter()
                
                for e in range(int(args)):
                    env = root_env.fork()
                    state = env.reset()
                    config = copy.copy(mcts_config)
                    mcts = puct_mcts.PUCTMCTS(env, net, config)
                    episode_step = 0
                    train_examples = []
                    
                    while True:
                        player = env.current_player
                        episode_step += 1
                        policy = mcts.search()
                        
                        l = get_symmetries(state, policy) # rotated/flipped [(state, policy), ...]
                        # l = [(state, policy)]
                        train_examples += [(x[0]*player, x[1], player) for x in l] # [(state, pi, player), ...]
                        
                        action = np.random.choice(len(policy), p=policy)
                        state, reward, done = env.step(action)
                        if done:
                            examples = [(x[0], x[1], reward*((-1)**(x[-1]!=player))) for x in train_examples] # [(state, pi, reward), ...]
                            all_examples += examples
                            all_episode_len.append(episode_step)
                            result_counter.add(reward, player)
                            break
                        mcts = mcts.get_subtree(action)
                        if mcts is None:
                            mcts = puct_mcts.PUCTMCTS(env, net, mcts_config)
                logger.debug(f"[Worker {id}] Finished {int(args)} episodes (length={all_episode_len}) in {time.time()-st0:.3f}s, {result_counter}")
                conn.send((all_examples, result_counter))
                
            if command == 'load_net':
                file_name = str(args)
                if not file_name or file_name == 'None':
                    file_name = 'best.pth.tar'
                net.load_checkpoint(folder=checkpoint_path, filename=file_name)
                logger.debug(f"[Worker {id}] Loaded net from {checkpoint_path}")
            
            if command == 'load_opp_net':
                file_name = str(args)
                if not file_name:
                    file_name = 'temp.pth.tar'
                opp_net.load_checkpoint(folder=checkpoint_path, filename=file_name)
                logger.debug(f"[Worker {id}] Loaded net from {checkpoint_path}")
            
            if command == 'pit_opp':
                n_run = int(args)
                logger.debug(f"[Worker {id}] Start evaluating for {int(args)} round")
                ############################################################
                #                  TODO: Your Code Here                    #
                # do multi_match with current net and opp_net, return the result
                ret = None
                ############################################################
                logger.debug(f"[Worker {id}] Finished evaluating for {int(args)} round")
                conn.send(ret)
                logger.debug(f"[Worker {id}] Finished evaluating for {int(args)} round")
            
            if command == 'pit_eval':
                n_run = int(args)
                logger.debug(f"[Worker {id}] Start evaluating for {int(args)} round")
                opponent = RandomPlayer()
                # opponent = AlphaBetaPlayer()
                # opponent = NuralNetPlayer()
                # opponent = UCTPlayer(UCTMCTSConfig(n_rollout=9, n_search=33))
                current_mcts_player = PUCTPlayer(mcts_config, net, deterministic=True)
                ret = multi_match(root_env.fork(), current_mcts_player, opponent, n_run, disable_tqdm=True)
                conn.send(ret)
                logger.debug(f"[Worker {id}] Finished evaluating for {int(args)} round")
                
                
        except Exception as e:
            print(e)
            traceback.print_exc()
            print(f"[Worker {id}] shutting down worker-{id}")
            if locals().get('conn') and conn.closed == False:
                conn.close()
            exit(0)
            
    conn.close()
    return

class AlphaZeroParallel:
    def __init__(self, env:BaseGame, env_builder, net_builder, config:AlphaZeroConfig, n_worker:int, seed:int=None):
        self.env = env
        self.net = net_builder()
        self.last_net = self.net.copy()
        self.config = config
        self.mcts_config = puct_mcts.MCTSConfig(
            C=config.C, 
            n_search=config.n_search, 
            temperature=config.temperature
        )
        assert n_worker > 0
        self.n_worker = n_worker
        self.mcts_config.with_noise = False
        self.mcts = None
        self.train_eamples_queue = [] 
        
        if seed is None:
            seed = 0
        logger.info(f"[AlphaZeroParallel] Creating {self.n_worker} workers")
        ctx = torch.multiprocessing.get_context("spawn")
        self.pipes = [ctx.Pipe() for _ in range(self.n_worker)]
        self.workers = [ctx.Process(
            target=execute_episode_worker,
            args=(
                child_conn,
                dill.dumps(env_builder, recurse=True),
                dill.dumps(net_builder, recurse=True),
                self.mcts_config,
                i,
                seed + i,
                self.config.checkpoint_path
            ),
        ) for i, (_, child_conn) in enumerate(self.pipes)]
        self.pipes = [(parent_conn, None) for parent_conn, _ in self.pipes]
        for worker in self.workers:
            worker.start()
        logger.debug(f"[AlphaZeroParallel] Started {self.n_worker} workers")
    
    def close(self):
        for (parent_pipe, _), worker in zip(self.pipes, self.workers):
            parent_pipe.send(('close', ""))
            worker.join()
            worker.close()
            parent_pipe.close()
    
    def execute_episode_parallel(self):
        start_search_t = time.time()
        train_examples = []
        n = self.config.n_match_train
        repeat_num = [n // self.n_worker + int(i < (n % self.n_worker)) for i in range(self.n_worker)]
        for work_cnt, (parent_pipe, _) in zip(repeat_num, self.pipes):
            if work_cnt <= 0:
                continue
            parent_pipe.send(('load_net', ""))
            parent_pipe.send(('run', work_cnt))
        result_counter = ResultCounter()
        for work_cnt, (parent_pipe, _) in zip(repeat_num, self.pipes):
            if work_cnt <= 0:
                continue
            result, cnt = parent_pipe.recv()
            train_examples += result
            result_counter.merge_with(cnt)
        logger.info(f"[AlphaZeroParallel] Finished {n} episodes ({len(train_examples)} examples) in {time.time()-start_search_t:.3f}s, {result_counter}")
        return train_examples
    
    def pit_with_last(self, n_run:int, opp_checkpt_filename:str, current_checkpt_filename:str):
        n_run = self.config.n_match_update
        assert n_run % 2 == 0
        n_run = n_run // 2
        repeat_num = [n_run // self.n_worker + int(i < (n_run % self.n_worker)) for i in range(self.n_worker)]
        logger.info(f"[AlphaZeroParallel] Start evaluating with last best model for {n_run*2} round")
        st0 = time.time()
        for work_cnt, (parent_pipe, _) in zip(repeat_num, self.pipes):
            if work_cnt <= 0:
                continue
            parent_pipe.send(('load_net', current_checkpt_filename))
            parent_pipe.send(('load_opp_net', opp_checkpt_filename))
            parent_pipe.send(('pit_opp', work_cnt*2))
        n_p1_win, n_p2_win, n_draw = 0, 0, 0
        for work_cnt, (parent_pipe, _) in zip(repeat_num, self.pipes):
            if work_cnt <= 0:
                continue
            r = parent_pipe.recv()[0]
            n_p1_win += r[0]
            n_p2_win += r[1]
            n_draw += r[2]
        logger.info(f"[AlphaZeroParallel] Finished evaluating with last best model for {n_run*2} round in {time.time()-st0:.3f}s")
        return (n_p1_win, n_p2_win, n_draw)
    
    def evaluate(self):
        n_run = self.config.n_match_eval
        assert n_run % 2 == 0
        n_run = n_run // 2
        repeat_num = [n_run // self.n_worker + int(i < (n_run % self.n_worker)) for i in range(self.n_worker)]
        logger.info(f"[AlphaZeroParallel] Start evaluating with baseline for {n_run*2} round")
        st0 = time.time()
        for work_cnt, (parent_pipe, _) in zip(repeat_num, self.pipes):
            if work_cnt <= 0:
                continue
            parent_pipe.send(('load_net', ""))
            parent_pipe.send(('pit_eval', work_cnt*2))
        class ans_pack:
            def __init__(self, r=(0, 0, 0)):
                self.n_p1_win = r[0]
                self.n_p2_win = r[1]
                self.n_draw = r[2]
            def add(self, r):
                self.n_p1_win += r[0]
                self.n_p2_win += r[1]
                self.n_draw += r[2]
            def tolist(self):
                return self.n_p1_win, self.n_p2_win, self.n_draw
        total_result = ans_pack()
        first_result = ans_pack()
        latter_result = ans_pack()
        for work_cnt, (parent_pipe, _) in zip(repeat_num, self.pipes):
            if work_cnt <= 0:
                continue
            r, first, latter = parent_pipe.recv()
            total_result.add(r)
            first_result.add(first)
            latter_result.add(latter)
        result = [total_result.tolist(), first_result.tolist(), latter_result.tolist()]
        logger.info(f"[AlphaZeroParallel] Finished evaluating with baseline for {n_run*2} round in {time.time()-st0:.3f}s")
        logger.info(f"[EVALUATION RESULT]: win{result[0][0]}, lose{result[0][1]}, draw{result[0][2]}")
        logger.info(f"[EVALUATION RESULT]:(first)  win{result[1][0]}, lose{result[1][1]}, draw{result[1][2]}")
        logger.info(f"[EVALUATION RESULT]:(second) win{result[2][0]}, lose{result[2][1]}, draw{result[2][2]}")
    
    def learn(self):
        self.net.save_checkpoint(folder=self.config.checkpoint_path, filename='best.pth.tar')
        for iter in range(1, self.config.n_train_iter + 1):
            st = time.time()
            logger.info(f"------ Start Self-Play Iteration {iter} ------")
            
            # collect new examples
            self.train_eamples_queue += self.execute_episode_parallel()
            
            # pop old examples
            if len(self.train_eamples_queue) > self.config.max_queue_length:
                self.train_eamples_queue = self.train_eamples_queue[-self.config.max_queue_length:]
            
            # shuffle examples for training
            train_data = copy.copy(self.train_eamples_queue)
            shuffle(train_data)
            logger.info(f"[TRAIN DATA SIZE]: {len(train_data)}")
            ############################################################
            #                  TODO: Your Code Here                    #
            
            # train the network with train_data
            # you can use self.net.train(train_data) to train the network
            
            # update the parameters of network if winning rate of new model is larger than update_threshold (against a older version of model)
            # you can use self.evaluate() to evaluate the performance of the current model (against a baseline player)
            
            # you can use self.pit_with_last(n_run, opp_checkpt_filename, current_checkpt_filename) to evaluate the performance of the current model
            
            ############################################################
            logger.info(f"------ Finished Self-Play Iteration {iter} in {time.time()-st:.3f}s ------\n")
    
    def eval(self):
        # NOTE: This automatically load the best model
        self.evaluate()
        

if __name__ == "__main__":
    from env import *
    import torch
    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler("log.txt")
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    config = AlphaZeroConfig(
        n_train_iter=50,
        n_match_train=20,
        n_match_update=20,
        n_match_eval=20,
        max_queue_length=40000,
        update_threshold=0.501,
        n_search=120, 
        temperature=1.0, 
        C=1.0,
        checkpoint_path="checkpoint"
    )
    model_training_config = ModelTrainingConfig(
        epochs=10,
        batch_size=128,
        lr=0.0007,
        dropout=0.3,
        num_channels=512,
        weight_decay=0,
    )
    network_config = BaseNetConfig()
    assert config.n_match_update % 2 == 0
    assert config.n_match_eval % 2 == 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    env = GoGame(9)
    def net_builder(device=device):
        net = MyNet(env.observation_size, env.action_space_size, network_config, device=device)
        return ModelWrapper(env.observation_size, env.action_space_size, net, model_training_config)
        
    N_WORKER = 20
    env_builder = lambda: env.__class__(*env.init_param_list())
    alphazero = AlphaZeroParallel(env, env_builder,  net_builder, config, N_WORKER)
    
    if sys.argv[1] == "eval":
        alphazero.eval()
    else:
        alphazero.learn()

    alphazero.close()