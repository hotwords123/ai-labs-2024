from env.base_env import BaseGame, get_symmetries, ResultCounter
from torch.nn import Module
from model.wrapper import ModelWrapper, ModelTrainingConfig
from model.model import AlphaZeroNet, AlphaZeroNetConfig
from mcts import puct_mcts
from util import *

import numpy as np
import random, sys
import copy
from tqdm import tqdm
from random import shuffle
from players import *
from mcts.uct_mcts import UCTMCTSConfig

from pathlib import Path
from datetime import datetime

from env import *
import torch

from pit_puct_mcts import multi_match

import logging
logger = logging.getLogger(__name__)


class AlphaZeroConfig():
    def __init__(
        self, 
        job_id: str,
        n_train_iter:int=300,
        n_match_train:int=20,
        n_match_update:int=20,
        n_match_eval:int=20,
        max_queue_length:int=8000,
        update_threshold:float=0.501,
        opening_moves:int=10,
        checkpoint_path:str="checkpoint",
        sgf_path:str="sgf",
    ):
        self.job_id = job_id

        self.n_train_iter = n_train_iter
        self.n_match_train = n_match_train
        self.max_queue_length = max_queue_length
        self.n_match_update = n_match_update
        self.n_match_eval = n_match_eval
        self.update_threshold = update_threshold
        self.opening_moves = opening_moves
        
        self.checkpoint_path = checkpoint_path
        self.sgf_path = sgf_path

class AlphaZero:
    def __init__(self, env:BaseGame, net:ModelWrapper, config:AlphaZeroConfig, mcts_config:puct_mcts.MCTSConfig):
        self.env = env
        self.net = net
        self.last_net = net.copy()
        self.config = config
        self.mcts_config = mcts_config
        self.mcts = None
        self.train_eamples_queue = [] 
        self.checkpoint_dir = Path(config.checkpoint_path) / config.job_id
    
    def execute_episode(self, sgf_name: str | None = None):
        # collect examples from one game episode
        train_examples = []
        data = GameData(
            self.env.n,
            PB="AlphaZero",
            PW="AlphaZero",
            KM=0.0,
        )
        env = self.env.fork()
        state = env.reset()
        mcts_config = copy.copy(self.mcts_config)
        mcts = puct_mcts.PUCTMCTS(env, self.net, mcts_config)
        episodeStep = 0
        fg = False
        while True:
            player = env.current_player
            episodeStep += 1
            policy = mcts.search()
            
            
            l = get_symmetries(state, policy) # rotated/flipped [(state, policy), ...]
            train_examples += [(l_state * player, l_pi, player) for l_state, l_pi in l] # [(state, pi, player), ...]
            
            if episodeStep <= self.config.opening_moves:
                # For the first few moves, we use a higher temperature to encourage exploration
                action = np.random.choice(len(policy), p=policy)
            else:
                # For the rest of the moves, we always select the action with the highest probability
                action = np.argmax(policy)

            state, reward, done = env.step(action)
            comments = [
                f'Priors: {self.format_top_moves(mcts.root.child_priors)}',
                f'Top moves: {self.format_top_moves(policy)}',
                f'PV: {self.format_node(mcts.root, 0)}',
            ]
            comments.extend(
                f'- {self.format_action(node.action)}: {self.format_node(node, i + 1)}'
                for i, node in enumerate(mcts.get_path())
            )
            data.add_move(player, action, '\n'.join(comments))
            if done:
                reward *= player
                data.set_result(reward)
                examples = [(l_state, l_pi, reward * l_player) for l_state, l_pi, l_player in train_examples] # [(state, pi, reward), ...]
                if sgf_name:
                    self.save_sgf(sgf_name, data)
                return examples
            mcts = mcts.get_subtree(action)
            if mcts is None:
                mcts = puct_mcts.PUCTMCTS(env, self.net, mcts_config)

    def format_top_moves(self, policy: np.ndarray, top_n: int = 10, p_threshold: float = 0.01) -> str:
        top_indices = np.argsort(policy)[::-1][:top_n]
        return ' '.join(
            f'{self.format_action(i)}({policy[i]:.3f})'
            for i in top_indices if policy[i] > p_threshold
        )

    def format_action(self, action: int) -> str:
        n = self.env.n
        if action == n * n:
            return 'PASS'
        else:
            x, y = action % n, action // n
            return 'ABCDEFGHJKLMNOPQRST'[x] + str(n - y)

    def format_node(self, node: puct_mcts.MCTSNode, depth: int) -> str:
        V = node.value
        N = np.sum(node.child_N_visit)
        Q = np.sum(node.child_V_total) / N if N > 0 else V
        player = (-1)**depth
        return f'V/{V * player:.3f} Q/{Q * player:.3f} N/{N}'

    def save_sgf(self, sgf_name: str, data: GameData):
        if not self.config.sgf_path:
            return
        sgf_file = Path(self.config.sgf_path) / self.config.job_id / sgf_name
        sgf_file.parent.mkdir(parents=True, exist_ok=True)
        sgf_file.write_text(data.to_sgf())
    
    def evaluate(self, sgf_prefix: str | None = None):
        player = PUCTPlayer(self.mcts_config, self.net, deterministic=True)
        # baseline_player = AlphaBetaPlayer()
        baseline_player = RandomPlayer()
        # baseline_player = UCTPlayer(UCTMCTSConfig(n_rollout=9, n_search=33))
        if self.config.sgf_path and sgf_prefix:
            def save_sgf(role: int, i: int, data: GameData):
                self.save_sgf(f'{sgf_prefix}eval-{"AB"[role]}{i}_{format_datetime()}.sgf', data)
        else:
            save_sgf = None
        first_play, second_play = multi_match(self.env, player, baseline_player, self.config.n_match_eval, save_sgf=save_sgf)
        logger.info(f"[EVALUATION RESULT]: {first_play + second_play}")
        logger.info(f"[EVALUATION RESULT]:(first)  {first_play}")
        logger.info(f"[EVALUATION RESULT]:(second) {second_play}")
    
    def learn(self, last_epoch: int = 0):
        if last_epoch > 0:
            self.net.load_checkpoint(self.checkpoint_dir / f'train-{last_epoch}.pth.tar')

        for iter in range(last_epoch + 1, self.config.n_train_iter + 1):
            logger.info(f"------ Start Self-Play Iteration {iter} ------")
            
            # collect new examples
            T = tqdm(range(self.config.n_match_train), desc="Self Play")
            cnt = ResultCounter()
            for i in T:
                episode = self.execute_episode(sgf_name=f'iter{iter}/train-{i}_{format_datetime()}.sgf')
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

            self.net.train(train_data)

            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.net.save_checkpoint(self.checkpoint_dir / f'train-{iter}.pth.tar')

    def eval(self, checkpoint_name: str = 'best'):
        self.net.load_checkpoint(self.checkpoint_dir / f'{checkpoint_name}.pth.tar')
        self.evaluate(sgf_prefix=f'{checkpoint_name.replace("train-", "iter")}/')


GAME_CLASS = {
    'tictactoe': TicTacToeGame,
    'gobang': GobangGame,
    'go': GoGame,
}


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    # General settings
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--job_id', type=str, help='job id')
    parser.add_argument('--log_file', type=str, default='log.txt', help='path to save the log file')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint', help='path to save the model')
    parser.add_argument('--sgf_path', type=str, default='sgf', help='path to save game records')
    # Environment settings
    parser.add_argument('--game', choices=GAME_CLASS.keys(), default='go', help='Game to play')
    parser.add_argument('--args', type=int, nargs='*', default=[9], help='Arguments for the game')
    # Reinforcement learning settings
    parser.add_argument('--n_train_iter', type=int, default=50, help='number of training iterations')
    parser.add_argument('--n_match_train', type=int, default=20, help='number of self-play matches for each training iteration')
    parser.add_argument('--n_match_update', type=int, default=20, 
    help='number of self-play matches for updating the model')
    parser.add_argument('--n_match_eval', type=int, default=20, help='number of matches for evaluating the model')
    parser.add_argument('--max_queue_length', type=int, default=40000, help='max length of training examples queue')
    parser.add_argument('--update_threshold', type=float, default=0.501, help='winning rate threshold for updating the model')
    # MCTS settings
    parser.add_argument('--n_search', type=int, default=120, help='number of MCTS simulations')
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature for MCTS')
    parser.add_argument('--C', type=float, default=1.0, help='exploration constant for MCTS')
    parser.add_argument('--with_noise', action='store_true', help='use dirichlet noise for MCTS')
    parser.add_argument('--dir_epsilon', type=float, default=0.25, help='dirichlet noise epsilon')
    parser.add_argument('--dir_alpha', type=float, default=0.15, help='dirichlet noise alpha')
    parser.add_argument("--opening_moves", type=int, default=10, help="number of opening moves")

    subparsers = parser.add_subparsers(dest='mode', required=True)

    subparser = subparsers.add_parser('train')
    # Training settings
    subparser.add_argument('--last_epoch', type=int, default=0, help='last epoch')
    # Model training settings
    subparser.add_argument('--epochs', type=int, default=10, help='number of epochs for training the model')
    subparser.add_argument('--batch_size', type=int, default=128, help='batch size for training the model')
    subparser.add_argument('--lr', type=float, default=0.0007, help='learning rate for training the model')
    subparser.add_argument('--dropout', type=float, default=0.3, help='dropout rate for training the model')
    subparser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay for training the model')
    # Model settings (TODO)

    subparser = subparsers.add_parser('eval')
    # Evaluation settings
    subparser.add_argument('--n_match_eval', type=int, default=20, help='number of matches for evaluating the model')
    subparser.add_argument('--checkpoint-name', type=str, default='best', help='model checkpoint to evaluate')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler(args.log_file)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    config = AlphaZeroConfig(
        # General settings
        job_id = args.job_id or '_'.join((args.mode, format_datetime(), str(args.seed))),
        checkpoint_path=args.checkpoint_path,
        sgf_path=args.sgf_path,
        # Reinforcement learning settings
        n_train_iter=args.n_train_iter,
        n_match_train=args.n_match_train,
        n_match_update=args.n_match_update,
        n_match_eval=args.n_match_eval,
        max_queue_length=args.max_queue_length,
        update_threshold=args.update_threshold,
        opening_moves=args.opening_moves,
    )

    mcts_config = puct_mcts.MCTSConfig(
        n_search=args.n_search,
        temperature=args.temperature,
        C=args.C,
        with_noise=args.with_noise,
        dir_epsilon=args.dir_epsilon,
        dir_alpha=args.dir_alpha,
    )

    model_training_config = None
    if args.mode == "train":
        model_training_config = ModelTrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    env = GAME_CLASS[args.game](*args.args)
    model_config = AlphaZeroNetConfig()
    net = AlphaZeroNet(env.observation_size, env.action_space_size, config=model_config, device=device)
    net = ModelWrapper(env.observation_size, env.action_space_size, net, model_training_config)
    
    alphazero = AlphaZero(env, net, config, mcts_config)
    if args.mode == "eval":
        alphazero.eval(args.checkpoint_name)
    else:
        alphazero.learn(last_epoch=args.last_epoch)