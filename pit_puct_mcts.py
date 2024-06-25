from env import *
from players import *
from mcts.uct_mcts import UCTMCTSConfig
from tqdm import trange, tqdm
from multiprocessing import Process
from torch.distributed.elastic.multiprocessing import Std, start_processes
from util import *
from model.model import AlphaZeroNet
from model.wrapper import ModelWrapper

import logging
logger = logging.getLogger(__name__)

import numpy as np
import torch
from pathlib import Path

def log_devide_line(n=50):
    logger.info("--"*n)

def pit(game:BaseGame, player1:PUCTPlayer, player2:PUCTPlayer, log_output:bool=False):
    game.reset()
    if log_output:
        logger.info(f"start playing {type(game)}")
        log_devide_line()
    reward = 0
    data = GameData(
        game.n,
        PB=str(player1),
        PW=str(player2),
        KM=0.0,
    )
    
    for player in [player1, player2]:
        if hasattr(player, "clear"):
            player.clear()
            
    while True:
        a1 = player1.play(game)
        _, reward, done = game.step(a1)
        data.add_move(1, a1)
        if hasattr(player2, "opp_play"):
            player2.opp_play(a1)
        if log_output:
            logger.info(f"Player 1 ({player1}) move: {a1}")
            logger.info(game.to_string())
            log_devide_line()
        if done:
            break
        a2 = player2.play(game)
        _, reward, done = game.step(a2)
        data.add_move(-1, a2)
        if hasattr(player1, "opp_play"):
            player1.opp_play(a2)
        if log_output:
            logger.info(f"Player 2 ({player2}) move: {a2}")
            logger.info(game.to_string())
            log_devide_line()
        if done:
            reward *= -1
            break
    data.set_result(reward)
    if log_output:
        if reward == 1:
            logger.info(f"Player 1 ({player1}) win")
        elif reward == -1:
            logger.info(f"Player 2 ({player2}) win")
        else:
            logger.info("Draw")
    # print(game.observation, reward)
    return reward, data

def multi_match(game:BaseGame, player1:PUCTPlayer, player2:PUCTPlayer, n_match=100, disable_tqdm=False, sgf_file:SGFFile|None=None):
    assert n_match % 2 == 0 and n_match > 1, "n_match should be an even number greater than 1"

    first_play = PlayerStats()
    for i in trange(n_match//2, disable=disable_tqdm, desc="Playing first"):
        reward, data = pit(game, player1, player2, log_output=False)
        first_play.update(reward)
        if sgf_file is not None:
            sgf_file.save(f'black_{i}', data)

    second_play = PlayerStats()
    for i in trange(n_match//2, disable=disable_tqdm, desc="Playing second"):
        reward, data = pit(game, player2, player1, log_output=False)
        second_play.update(-reward)
        if sgf_file is not None:
            sgf_file.save(f'white_{i}', data)

    return first_play, second_play


def main():
    import argparse

    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    GAME_CLASS = {
        'tictactoe': TicTacToeGame,
        'gobang': GobangGame,
        'go': GoGame,
    }

    PLAYER_CLASS = {
        'human': HumanPlayer,
        'random': RandomPlayer,
        'alphabeta': AlphaBetaPlayer,
        'uct': lambda: UCTPlayer(config, deterministic=args.deterministic, log_policy=args.log_policy),
        'uct2': lambda: UCTPlayer(config2, deterministic=args.deterministic, log_policy=args.log_policy),
        'puct': lambda: PUCTPlayer(config, load_model(args.model_path), deterministic=args.deterministic),
        'puct2': lambda: PUCTPlayer(config2, load_model(args.model_path2), deterministic=args.deterministic),
    }

    parser = argparse.ArgumentParser(description='Pit two players against each other')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--quiet', action='store_true', help='Do not print the game state')
    parser.add_argument('--game', choices=GAME_CLASS.keys(), default='tictactoe', help='Game to play')
    parser.add_argument('--args', type=int, nargs='*', help='Arguments for the game')
    parser.add_argument('--players', choices=PLAYER_CLASS.keys(), nargs=2, default=['random', 'alphabeta'], help='Players to play')
    parser.add_argument('--n_match', type=int, help='Number of matches to play')
    parser.add_argument('--C', type=float, default=1.0, help='C value for UCTPlayer')
    parser.add_argument('--n_rollout', type=int, default=7, help='Number of rollouts for UCTPlayer')
    parser.add_argument('--n_search', type=int, default=64, help='Number of searches for UCTPlayer')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for UCTPlayer')
    parser.add_argument('--deterministic', action='store_true', help='Deterministic UCTPlayer or PUCTPlayer')
    parser.add_argument("--log_policy", action="store_true", help="Log policy of UCTPlayer")
    parser.add_argument('--C2', type=float, default=1.0, help='C value for UCTPlayer 2')
    parser.add_argument('--n_rollout2', type=int, default=7, help='Number of rollouts for UCTPlayer 2')
    parser.add_argument('--n_search2', type=int, default=64, help='Number of searches for UCTPlayer 2')
    parser.add_argument('--temperature2', type=float, default=1.0, help='Temperature for UCTPlayer 2')
    parser.add_argument('--model_path', type=str, help='Path to the model for PUCTPlayer')
    parser.add_argument('--model_path2', type=str, help='Path to the model for PUCTPlayer 2')
    parser.add_argument('--sgf_path', type=str, help='Save sgf file')
    parser.add_argument('--device', type=str, help='Device for PUCTPlayer')

    args = parser.parse_args()

    # set seed to reproduce the result
    if args.seed is not None:
        np.random.seed(args.seed)

    game_args = args.args if args.args is not None else []
    game: BaseGame = GAME_CLASS[args.game](*game_args)

    # config for MCTS
    config = UCTMCTSConfig(
        C=args.C,
        n_rollout=args.n_rollout,
        n_search=args.n_search,
        temperature=args.temperature,
    )

    config2 = UCTMCTSConfig(
        C=args.C2,
        n_rollout=args.n_rollout2,
        n_search=args.n_search2,
        temperature=args.temperature2,
    )

    # load model
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def load_model(model_path: str) -> ModelWrapper:
        model = AlphaZeroNet(game.observation_size, game.action_space_size, device=device)
        model = ModelWrapper(game.observation_size, game.action_space_size, model)
        model.load_checkpoint(model_path)
        return model

    # player initialization
    player1 = PLAYER_CLASS[args.players[0]]()
    player2 = PLAYER_CLASS[args.players[1]]()

    result_text = [
        f"Player 1 ({player1}) win",
        f"Player 2 ({player2}) win",
        "Draw"
    ]

    if args.sgf_path:
        sgf_file = SGFFile(args.sgf_path, prefix='pit_', props={"C": f"Args: {args.__dict__}"})
    else:
        sgf_file = None
    
    # single match
    if args.n_match is None:
        reward, data = pit(game, player1, player2, log_output=not args.quiet)
        if args.quiet:
            print(result_text[0 if reward > 0 else 1 if reward < 0 else 2])
        if sgf_file is not None:
            sgf_file.save("match", data)
    else:
        first_play, second_play = multi_match(game, player1, player2, n_match=args.n_match, sgf_file=sgf_file)
        print(f"[EVALUATION RESULT]: {first_play + second_play}")
        print(f"[EVALUATION RESULT]:(first)  {first_play}")
        print(f"[EVALUATION RESULT]:(second) {second_play}")

    if sgf_file is not None:
        sgf_file.close()


if __name__ == "__main__":
    main()
