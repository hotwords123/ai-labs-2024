from env.base_env import BaseGame
from .base_player import BasePlayer
import numpy as np
import re

coord2d_re = re.compile(r'^([A-Za-z])(\d+)$')

class HumanPlayer(BasePlayer):
    def __init__(self):
        pass
    
    def __str__(self):
        return "Human Player"

    def play(self, state:BaseGame):
        valid = state.action_mask
        while True:
            print("valid moves:", valid.nonzero()[0])
            s = input()
            try:
                if s == 'pass':
                    a = state.action_space_size - 1
                elif m := coord2d_re.match(s):
                    x = "ABCDEFGHJKLMNOPQRST".index(m.group(1).upper())
                    y = state.n - int(m.group(2))
                    a = y * state.n + x
                else:
                    a = int(s)
                if valid[a]:
                    break
            except:
                pass
            print('Invalid')
        return a