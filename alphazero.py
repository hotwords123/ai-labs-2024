from env.base_env import BaseGame, get_symmetries, ResultCounter
from torch.nn import Module
from model.wrapper import ModelWrapper, ModelTrainingConfig
from model.model import AlphaZeroNet, AlphaZeroNetConfig
from mcts import puct_mcts
from util import *

import numpy as np
import random
import copy
from tqdm import tqdm
from players import *

from pathlib import Path
from dataclasses import dataclass
import pickle

from env import *
import torch

from pit_puct_mcts import multi_match

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)


@dataclass
class AlphaZeroConfig():
    # General settings
    job_id: str
    checkpoint_path: str = "checkpoint"
    sgf_path: str = "sgf"
    result_path: str = "results"

    # MCTS settings
    eval_temperature: float = 0.1
    
    # Reinforcement learning settings
    n_train_iter: int = 50
    n_match_train: int = 20
    n_match_update: int = 20
    max_queue_length: int = 300000
    update_threshold: float = 0.551
    use_latest: bool = False
    eval_every: int = 1
    enable_resign: bool = False
    resign_threshold: float = -0.90
    n_resign_min_turn: int = 20
    n_resign_low_turn: int = 3
    resign_test_ratio: float = 0.1
    opening_moves: int = 10

    # Evaluation settings
    n_match_eval: int = 20


class AlphaZero:
    def __init__(self, env:BaseGame, net:ModelWrapper, config:AlphaZeroConfig, mcts_config:puct_mcts.MCTSConfig):
        self.env = env
        self.net = net
        self.last_net = net.copy()
        self.config = config
        self.mcts_config = mcts_config
        self.mcts = None
        self.train_examples_queue = [] 

        self.checkpoint_dir = Path(config.checkpoint_path) / config.job_id
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.result_dir = Path(config.result_path) / config.job_id
        self.result_dir.mkdir(parents=True, exist_ok=True)

        # Resignation statistics for testing false positives
        self.resign_test_stats = PlayerStats()
    
    def execute_episode(self, net: ModelWrapper = None):
        """
        Execute one episode of self-play.

        Args:
            net: the neural network used for MCTS, default to the best model

        Returns:
            examples: a list of training examples (state, pi, reward)
            data: game data for saving to SGF file
        """
        if net is None:
            net = self.net if self.config.use_latest else self.last_net

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
        mcts = puct_mcts.PUCTMCTS(env, net, mcts_config)
        episodeStep = 0

        resign_enabled = self.config.enable_resign and np.random.rand() >= self.config.resign_test_ratio
        resign_counter = {1: 0, -1: 0}
        resigned_player = None

        while True:
            player = env.current_player
            episodeStep += 1
            policy = mcts.search()
            
            # Augment training data with symmetries
            l = get_symmetries(state, policy) # rotated/flipped [(state, policy), ...]
            train_examples += [(l_state * player, l_pi, player) for l_state, l_pi in l] # [(state, pi, player), ...]
            
            if episodeStep <= self.config.opening_moves:
                # For the first few moves, we use a higher temperature to encourage exploration
                action = np.random.choice(len(policy), p=policy)
            else:
                # For the rest of the moves, we always select the action with the highest probability
                action = np.argmax(policy)

            state, reward, done = env.step(action)

            # Generate comments for the move
            pv = mcts.get_path()
            color = {1: "B", -1: "W"}[player]
            comments = [
                f'Priors: {self.format_top_moves(mcts.root.child_priors)}',
                f'Top moves: {self.format_top_moves(policy)}',
                f'PV: {color}{" ".join(self.format_action(node.action) for node in pv)}',
                f'Root: {self.format_node(mcts.root, 0)}'
            ]
            comments.extend(
                f'{i + 1}. {self.format_action(node.action)} {self.format_node(node, i + 1)}'
                for i, node in enumerate(pv)
            )
            data.add_move(player, action, '\n'.join(comments))

            # Check for end of game
            if done:
                reward *= player
                data.set_result(reward)
                if resigned_player is not None:
                    self.resign_test_stats.update(reward * resigned_player)
                break

            # Check for resignation
            if resigned_player is None:
                utility = np.sum(mcts.root.child_V_total) / np.sum(mcts.root.child_N_visit)
                if utility < self.config.resign_threshold:
                    resign_counter[player] += 1
                else:
                    resign_counter[player] = 0

                if episodeStep > 2 * self.config.n_resign_min_turn and resign_counter[player] >= self.config.n_resign_low_turn:
                    resigned_player = player
                    if resign_enabled:
                        reward = -player
                        data.set_result(reward, score='R')
                        break

            # Update MCTS with the new state
            mcts = mcts.get_subtree(action)
            if mcts is None:
                mcts = puct_mcts.PUCTMCTS(env, net, mcts_config)

        examples = [(state, pi, reward * player) for state, pi, player in train_examples] # [(state, pi, reward), ...]
        return examples, data

    def format_top_moves(self, policy: np.ndarray, top_n: int = 10, p_threshold: float = 0.01) -> str:
        """
        Format the top moves for SGF comments, e.g. "A1(0.123) B2(0.456) C3(0.789)".
        """
        top_indices = np.argsort(policy)[::-1][:top_n]
        return ' '.join(
            f'{self.format_action(i)}({policy[i]:.3f})'
            for i in top_indices if policy[i] > p_threshold
        )

    def format_action(self, action: int) -> str:
        """
        Format the action for SGF comments, e.g. "A1", "PASS".
        """
        n = self.env.n
        if action == n * n:
            return 'PASS'
        else:
            x, y = action % n, action // n
            return 'ABCDEFGHJKLMNOPQRST'[x] + str(n - y)

    def format_node(self, node: puct_mcts.MCTSNode, depth: int) -> str:
        """
        Format the node for SGF comments, e.g. "V / Q / N".
        The estimated value and utility function are from the perspective of the current player.
        """
        V = node.value
        N = np.sum(node.child_N_visit)
        Q = np.sum(node.child_V_total) / N if N > 0 else V
        player = (-1)**depth
        return f'{V * player:.3f} / {Q * player:.3f} / {N}'
    
    def build_puct_player(self, net: ModelWrapper):
        """
        Build a PUCT player using the specified neural network.
        """
        config = copy.copy(self.mcts_config)
        # Instead of making the player deterministic, we use a low temperature for evaluation to avoid repeating the same moves
        config.temperature = self.config.eval_temperature
        config.with_noise = False
        return PUCTPlayer(config, net)
    
    def evaluate(
        self,
        net: ModelWrapper = None,
        baseline_player: BasePlayer = None,
        n_match: int = None,
        sgf_file: SGFFile | None = None,
    ) -> tuple[PlayerStats, PlayerStats]:
        """
        Evaluate the model against a baseline player.

        Args:
            net: the neural network used for MCTS, default to the latest model
            baseline_player: the baseline player, default to a random player
            n_match: the number of matches to play, default to the evaluation setting
            sgf_file: the SGF file to save the game records

        Returns:
            first_play: the statistics when the model plays first
            second_play: the statistics when the model plays second
        """
        if net is None:
            net = self.net
        if baseline_player is None:
            # baseline_player = AlphaBetaPlayer()
            baseline_player = RandomPlayer()
            # baseline_player = UCTPlayer(UCTMCTSConfig(n_rollout=9, n_search=33))
        if n_match is None:
            n_match = self.config.n_match_eval
            
        player = self.build_puct_player(net)
        first_play, second_play = multi_match(self.env, player, baseline_player, n_match, sgf_file=sgf_file)
        logger.info(f"[EVALUATION RESULT]: {first_play + second_play}")
        logger.info(f"[EVALUATION RESULT]:(first)  {first_play}")
        logger.info(f"[EVALUATION RESULT]:(second) {second_play}")
        return first_play, second_play

    def open_sgf(self, epoch: str, name: str, prefix: str = '') -> SGFFile:
        """
        Open a SGF file for saving game records.
        The file path is in the format of "{job_id}/{epoch}/{name}_{datetime}.sgf".

        Args:
            epoch: the epoch number
            name: the name of the file
            prefix: the prefix of game names in the SGF file
        """
        sgf_dir = Path(self.config.sgf_path) / self.config.job_id / epoch
        sgf_dir.mkdir(parents=True, exist_ok=True)
        comment = "\n".join([
            f"Config: {self.config.__dict__}",
            f"MCTS Config: {self.mcts_config.__dict__}",
            f"Model Training Config: {self.net.config.__dict__}",
            f"Model Config: {self.net.net.config.__dict__}",
        ])
        return SGFFile(sgf_dir / f'{name}_{format_datetime()}.sgf', prefix=prefix, props={"C": comment})
    
    def learn(self, last_iter: int = 0):
        """
        Train the model using self-play.

        Args:
            last_iter: the last iteration number to start from
        """
        if last_iter > 0:
            self.net.load_checkpoint(self.checkpoint_dir / f'train-{last_iter}.pth.tar')
            self.sync_model('latest', f'train-{last_iter}.pth.tar')

            try:
                self.last_net.load_checkpoint(self.checkpoint_dir / f'best.pth.tar')
                self.sync_model('best', 'best.pth.tar')
            except FileNotFoundError:
                logger.warning("Best model not found, using the latest model instead")
                self.last_net.net.load_state_dict(self.net.net.state_dict())
                self.sync_model('best', f'train-{last_iter}.pth.tar')

            try:
                with open(self.result_dir / f'train-{last_iter}_data.pkl', 'rb') as f:
                    self.train_examples_queue = pickle.load(f)
            except FileNotFoundError:
                logger.warning("Training data not found, starting from scratch")

        for iter in range(last_iter + 1, self.config.n_train_iter + 1):
            logger.info(f"------ Start Self-Play Iteration {iter} ------")

            # collect new examples
            with self.open_sgf(f'iter{iter}', 'train', prefix='train_') as f:
                self.self_play(f)
            
            # pop old examples
            if len(self.train_examples_queue) > self.config.max_queue_length:
                self.train_examples_queue = self.train_examples_queue[-self.config.max_queue_length:]

            # save training data so we can resume training later
            with open(self.result_dir / f'train-{iter}_data.pkl', 'wb') as f:
                pickle.dump(self.train_examples_queue, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            train_data = copy.copy(self.train_examples_queue)
            # shuffling is done by dataloader, so no need to shuffle here
            # shuffle(train_data)
            logger.info(f"[TRAIN DATA SIZE]: {len(train_data)}")

            loss_history = self.net.train(train_data)
            logger.info(f"[TRAINING LOSS]: {loss_history.mean(axis=1)}")
            np.save(self.result_dir / f'train-{iter}_loss.npy', loss_history)

            self.net.save_checkpoint(self.checkpoint_dir / f'train-{iter}.pth.tar')
            self.sync_model('latest', f'train-{iter}.pth.tar')

            if not self.config.use_latest and iter % self.config.eval_every == 0:
                # evaluate the model and decide whether to update every few iterations
                with self.open_sgf(f'iter{iter}', 'eval', prefix='eval_') as f:
                    stats = self.eval_update(f)
                    score = (stats.n_win + 0.5 * stats.n_draw) / stats.n_match
                    if score > self.config.update_threshold:
                        self.last_net.net.load_state_dict(self.net.net.state_dict())
                        self.last_net.save_checkpoint(self.checkpoint_dir / 'best.pth.tar')
                        self.sync_model('best', 'best.pth.tar')
                        logger.info(f"[MODEL UPDATED]: iteration {iter}")

    def self_play(self, sgf_file: SGFFile):
        """
        Collect new examples using self-play.
        """
        cnt = ResultCounter()
        for i in tqdm(range(self.config.n_match_train), desc="Self Play"):
            episode, data = self.execute_episode()
            self.train_examples_queue += episode
            cnt.add(episode[0][-1], 1)
            sgf_file.save(f"{i}", data)
        logger.info(f"[NEW TRAIN DATA COLLECTED]: {str(cnt)}")
        logger.info(f"[RESIGNATION TEST]: {self.resign_test_stats}")

    def eval_update(self, sgf_file: SGFFile) -> PlayerStats:
        """
        Evaluate the model against the best model to decide whether to update.
        """
        opponent = self.build_puct_player(self.last_net)
        first_play, second_play = self.evaluate(baseline_player=opponent, n_match=self.config.n_match_update, sgf_file=sgf_file)
        return first_play + second_play
    
    def sync_model(self, target: str, filename: str):
        """
        Synchronize the model to workers.

        Args:
            target: the target model to synchronize (latest or best)
            filename: the filename of the model
        """
        pass

    def eval(self, checkpoint_name: str = 'best'):
        """
        Evaluate the model using the specified checkpoint.

        Args:
            checkpoint_name: the name of the checkpoint
        """
        self.net.load_checkpoint(self.checkpoint_dir / f'{checkpoint_name}.pth.tar')
        with self.open_sgf(checkpoint_name.replace("train-", "iter"), 'eval', prefix='eval_') as f:
            self.evaluate(sgf_file=f)


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
    parser.add_argument('--result_path', type=str, default='results', help='path to save auxiliary results')

    # Environment settings
    parser.add_argument('--game', choices=GAME_CLASS.keys(), default='go', help='Game to play')
    parser.add_argument('--args', type=int, nargs='*', default=[9], help='Arguments for the game')

    # MCTS settings
    parser.add_argument('--n_search', type=int, default=200, help='number of MCTS simulations')
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature for MCTS')
    parser.add_argument('--eval_temperature', type=float, default=0.1, help='override temperature for evaluation')
    parser.add_argument('--C', type=float, default=1.0, help='exploration constant for MCTS')

    subparsers = parser.add_subparsers(dest='mode', required=True)

    # Training settings
    subparser = subparsers.add_parser('train')
    subparser.add_argument('--last_iter', type=int, default=0, help='last iteration to start from')

    # Reinforcement learning settings
    subparser.add_argument('--n_train_iter', type=int, default=50, help='number of training iterations')
    subparser.add_argument('--n_match_train', type=int, default=20, help='number of self-play matches for each training iteration')
    subparser.add_argument('--n_match_update', type=int, default=20, help='number of self-play matches for updating the model')
    subparser.add_argument('--max_queue_length', type=int, default=300000, help='max length of training examples queue')
    subparser.add_argument('--update_threshold', type=float, default=0.551, help='winning rate threshold for updating the model')
    subparser.add_argument('--use_latest', action='store_true', help='always use the latest instead of the best model for self-play')
    subparser.add_argument('--eval_every', type=int, default=1, help='evaluate the model every n iterations')
    subparser.add_argument('--enable_resign', action='store_true', help='enable resignation for self-play')
    subparser.add_argument('--resign_threshold', type=float, default=-0.90, help='resignation threshold for self-play')
    subparser.add_argument('--n_resign_min_turn', type=int, default=20, help='minimum number of turns before resigning')
    subparser.add_argument('--n_resign_low_turn', type=int, default=3, help='resign if the score is below the threshold for n turns')
    subparser.add_argument('--resign_test_ratio', type=float, default=0.1, help='ratio of games not resigned to test false positives')
    subparser.add_argument("--opening_moves", type=int, default=10, help="number of opening moves")

    # MCTS settings
    subparser.add_argument('--with_noise', action='store_true', help='use dirichlet noise for MCTS')
    subparser.add_argument('--dir_epsilon', type=float, default=0.25, help='dirichlet noise epsilon')
    subparser.add_argument('--dir_alpha', type=float, default=0.15, help='dirichlet noise alpha')

    # Model training settings
    subparser.add_argument('--epochs', type=int, default=10, help='number of epochs for training the model')
    subparser.add_argument('--batch_size', type=int, default=256, help='batch size for training the model')
    subparser.add_argument('--lr', type=float, default=0.01, help='learning rate for training the model')
    # subparser.add_argument('--dropout', type=float, default=0.3, help='dropout rate for training the model')
    subparser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay for training the model')

    # Model settings (TODO)

    # Evaluation settings
    subparser = subparsers.add_parser('eval')
    subparser.add_argument('--n_match_eval', type=int, default=20, help='number of matches for evaluating the model')
    subparser.add_argument('--checkpoint-name', type=str, default='best', help='model checkpoint to evaluate')

    args = parser.parse_args()
    if not args.job_id:
        args.job_id = '_'.join((args.mode, format_datetime(), str(args.seed)))

    return args


def build_alphazero(args, device: torch.device, cls = AlphaZero):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    config = AlphaZeroConfig(
        # General settings
        job_id = args.job_id,
        checkpoint_path=args.checkpoint_path,
        sgf_path=args.sgf_path,
        
        # MCTS settings
        eval_temperature=args.eval_temperature,
    )

    mcts_config = puct_mcts.MCTSConfig(
        n_search=args.n_search,
        temperature=args.temperature,
        C=args.C,
    )

    # Model settings (TODO)
    model_config = AlphaZeroNetConfig()

    model_training_config = None

    if args.mode == "train":
        # Reinforcement learning settings
        config.n_train_iter = args.n_train_iter
        config.n_match_train = args.n_match_train
        config.n_match_update = args.n_match_update
        config.max_queue_length = args.max_queue_length
        config.update_threshold = args.update_threshold
        config.use_latest = args.use_latest
        config.eval_every = args.eval_every
        config.enable_resign = args.enable_resign
        config.resign_threshold = args.resign_threshold
        config.n_resign_min_turn = args.n_resign_min_turn
        config.n_resign_low_turn = args.n_resign_low_turn
        config.resign_test_ratio = args.resign_test_ratio
        config.opening_moves = args.opening_moves

        # MCTS settings
        mcts_config.with_noise = args.with_noise
        mcts_config.dir_epsilon = args.dir_epsilon
        mcts_config.dir_alpha = args.dir_alpha

        # Model training settings
        model_training_config = ModelTrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            # dropout=args.dropout,
            weight_decay=args.weight_decay,
        )

    elif args.mode == "eval":
        # Evaluation settings
        config.n_match_eval = args.n_match_eval
    
    env: BaseGame = GAME_CLASS[args.game](*args.args)
    net = AlphaZeroNet(env.observation_size, env.action_space_size, config=model_config, device=device)
    net = ModelWrapper(env.observation_size, env.action_space_size, net, model_training_config)
    
    return cls(env, net, config, mcts_config)


if __name__ == "__main__":
    args = parse_args()

    file_handler = logging.FileHandler(args.log_file)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    alphazero = build_alphazero(args, device)
    
    if args.mode == "eval":
        alphazero.eval(args.checkpoint_name)
    else:
        alphazero.learn(last_iter=args.last_iter)