from env.base_env import BaseGame, get_symmetries, ResultCounter
from torch.nn import Module
from model.wrapper import ModelWrapper, ModelTrainingConfig
from model.example_net import MLPNet, ConvNet, MyNet, BaseNetConfig
from mcts import puct_mcts

import numpy as np
import random, sys
import copy
from tqdm import tqdm
from random import shuffle
from players import *
from mcts.uct_mcts import UCTMCTSConfig

from pit_puct_mcts import multi_match

import logging
logger = logging.getLogger(__name__)


class AlphaZeroConfig():
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

class AlphaZero:
    def __init__(self, env:BaseGame, net:ModelWrapper, config:AlphaZeroConfig):
        self.env = env
        self.net = net
        self.last_net = net.copy()
        self.config = config
        self.mcts_config = puct_mcts.MCTSConfig(
            C=config.C, 
            n_search=config.n_search, 
            temperature=config.temperature
        )
        self.mcts_config.with_noise = False
        self.mcts = None
        self.train_eamples_queue = [] 
    
    def execute_episode(self):
        # collect examples from one game episode
        train_examples = []
        env = self.env.fork()
        state = env.reset()
        config = copy.copy(self.mcts_config)
        config.with_noise = True
        mcts = puct_mcts.PUCTMCTS(env, self.net, config)
        episodeStep = 0
        fg = False
        while True:
            player = env.current_player
            episodeStep += 1
            policy = mcts.search()
            
            
            l = get_symmetries(state, policy) # rotated/flipped [(state, policy), ...]
            train_examples += [(x[0], x[1], player) for x in l] # [(state, pi, player), ...]
            
            action = np.random.choice(len(policy), p=policy)
            state, reward, done = env.step(action)
            if done:
                examples = [(x[0]*player, x[1], reward*((-1)**(x[-1]!=player))) for x in train_examples] # [(state, pi, reward), ...]
                return examples
            mcts = mcts.get_subtree(action)
            if mcts is None:
                mcts = puct_mcts.PUCTMCTS(env, self.net, self.mcts_config)
    
    def evaluate(self):
        player = PUCTPlayer(self.mcts_config, self.net, deterministic=True)
        # baseline_player = AlphaBetaPlayer()
        baseline_player = RandomPlayer()
        # baseline_player = UCTPlayer(UCTMCTSConfig(n_rollout=9, n_search=33))
        result = multi_match(self.env, player, baseline_player, self.config.n_match_eval)
        logger.info(f"[EVALUATION RESULT]: win{result[0][0]}, lose{result[0][1]}, draw{result[0][2]}")
        logger.info(f"[EVALUATION RESULT]:(first)  win{result[1][0]}, lose{result[1][1]}, draw{result[1][2]}")
        logger.info(f"[EVALUATION RESULT]:(second) win{result[2][0]}, lose{result[2][1]}, draw{result[2][2]}")
    
    def learn(self):
        for iter in range(1, self.config.n_train_iter + 1):
            logger.info(f"------ Start Self-Play Iteration {iter} ------")
            
            # collect new examples
            T = tqdm(range(self.config.n_match_train), desc="Self Play")
            cnt = ResultCounter()
            for _ in T:
                episode = self.execute_episode()
                self.train_eamples_queue += episode
                cnt.add(episode[0][-1], 1)
            logger.info(f"[NEW TRAIN DATA COLLECTED]: {str(cnt)}")
            
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
            
            # you can use `self.net.load_checkpoint(folder=self.config.checkpoint_path, filename='temp.pth.tar')` to load the last model
            # you can use `self.net.save_checkpoint(folder=self.config.checkpoint_path, filename='temp.pth.tar')` to save the current model
                
            ############################################################

    def eval(self):
        self.net.load_checkpoint(folder=self.config.checkpoint_path, filename='best.pth.tar')
        self.evaluate()

if __name__ == "__main__":
    from env import *
    import torch
    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    env = GoGame(9)
    # net = MLPNet(env.observation_size, env.action_space_size, BaseNetConfig(), device=device)
    net = MyNet(env.observation_size, env.action_space_size, BaseNetConfig(), device=device)
    net = ModelWrapper(env.observation_size, env.action_space_size, net, model_training_config)
    
    alphazero = AlphaZero(env, net, config)
    if sys.argv[1] == "eval":
        alphazero.eval()
    else:
        alphazero.learn()