from env import *
from players import *
from mcts.puct_mcts import MCTSConfig as PUCTMCTSConfig
from tqdm import trange, tqdm
from multiprocessing import Process
from torch.distributed.elastic.multiprocessing import Std, start_processes
from util import GameData

import logging
logger = logging.getLogger(__name__)

import numpy as np


class PlayerStats:
    def __init__(self, n_win: int = 0, n_lose: int = 0, n_draw: int = 0):
        self.n_win = n_win
        self.n_lose = n_lose
        self.n_draw = n_draw

    def __repr__(self):
        return f"Win: {self.n_win}, Lose: {self.n_lose}, Draw: {self.n_draw}"

    @property
    def n_match(self):
        return self.n_win + self.n_lose + self.n_draw

    @property
    def win_rate(self):
        return self.n_win / self.n_match

    @property
    def unbeaten_rate(self):
        return 1 - self.n_lose / self.n_match

    def update(self, reward):
        if reward == 1:
            self.n_win += 1
        elif reward == -1:
            self.n_lose += 1
        else:
            self.n_draw += 1

    def __add__(self, other: "PlayerStats"):
        return PlayerStats(
            self.n_win + other.n_win,
            self.n_lose + other.n_lose,
            self.n_draw + other.n_draw
        )


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

def multi_match(game:BaseGame, player1:PUCTPlayer, player2:PUCTPlayer, n_match=100, disable_tqdm=False, save_sgf=None):
    assert n_match % 2 == 0 and n_match > 1, "n_match should be an even number greater than 1"

    first_play = PlayerStats()
    for i in trange(n_match//2, disable=disable_tqdm):
        reward, data = pit(game, player1, player2, log_output=False)
        first_play.update(reward)
        if save_sgf is not None:
            save_sgf(0, i, data)

    second_play = PlayerStats()
    for i in trange(n_match//2, disable=disable_tqdm):
        reward, data = pit(game, player2, player1, log_output=False)
        second_play.update(-reward)
        if save_sgf is not None:
            save_sgf(1, i, data)

    return first_play, second_play
