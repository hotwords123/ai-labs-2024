import numpy as np
from env.base_env import *

MAX_VALUE = 1 

class AlphaBetaSearch:

    def __init__(self):
        self.cnt = 0

    def max_value(self, game_state:BaseGame, alpha, beta):
        self.cnt += 1 # for debug
        v = -np.inf
        valid_moves_mask = game_state.action_mask
        valid_moves = np.where(valid_moves_mask == 1)[0]
        for a in valid_moves:
            # copy the game state and take the action
            next_state = game_state.fork()
            _, r, _ = next_state.step(a, return_obs=False)
            
            # check result
            if r != NOTEND: # if game is end
                v = max(v, r)
            else: 
                # if game is not end, recursive search, and use -max_value of next state
                # also swap alpha and beta as the player has changed
                v = max(v, - self.max_value(next_state, beta, alpha))
                
            # pruning; 
            # if -v <= beta, cut by beta (the maximum value of the opponent)
            # if v == MAX_VALUE, v can not be larger
            if -v <= beta or v == MAX_VALUE: 
                return v
            
            # update alpha, which is the maximum value of current player
            alpha = max(alpha, v)
        return v

    def get_best_move(self, game_state:BaseGame):
        best_move, best_value = -1, -np.inf
        alpha, beta = -np.inf, -np.inf
        valid_moves_mask = game_state.action_mask
        valid_moves = np.where(valid_moves_mask == 1)[0]
        for a in valid_moves:
            next_state = game_state.fork()
            _, r, _ = next_state.step(a, return_obs=False)
            v = r if r !=NOTEND else -self.max_value(next_state, beta, alpha)
            if v > best_value:
                best_value = v
                best_move = a
        # print(self.cnt) # use this to check the number of nodes expanded
        self.cnt = 0
        return best_move
