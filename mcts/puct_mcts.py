from .node import MCTSNode, INF
from .config import MCTSConfig
from env.base_env import BaseGame

from model.wrapper import ModelWrapper
import numpy as np


class PUCTMCTS:
    def __init__(self, init_env:BaseGame, model: ModelWrapper, config: MCTSConfig, root:MCTSNode=None):
        self.model = model
        self.config = config
        self.root = root
        if root is None:
            self.init_tree(init_env)
        self.root.cut_parent()
    
    def init_tree(self, init_env:BaseGame):
        env = init_env.fork()
        obs = env.observation
        self.root = MCTSNode(
            action=None, env=env, reward=0
        )
        child_prior, _ = self.model.predict(obs * env.current_player)
        # print("child_prior=", child_prior)
        self.root.set_prior(child_prior)
    
    def get_subtree(self, action:int):
        # return a subtree with root as the child of the current root
        # the subtree represents the state after taking action
        if self.root.has_child(action):
            new_root = self.root.get_child(action)
            return PUCTMCTS(new_root.env, self.model, self.config, new_root)
        else:
            return None
    
    def puct_action_select(self, node:MCTSNode) -> int:
        # select the best action based on UCB when expanding the tree
        
        ########################
        # TODO: your code here #
        ########################
        return 0
        ########################

    def backup(self, node:MCTSNode, value:float) -> None:
        # backup the value of the leaf node to the root
        # update N_visit and V_total of each node in the path
        
        ########################
        # TODO: your code here #
        ########################
        pass 
        ########################    
            
    
    def rollout(self, node:MCTSNode) -> float:
        # simulate the game until the end
        # return the reward of the game
        # NOTE: the reward should be convert to the perspective of the current player!
        
        ########################
        # TODO: your code here #
        ########################
        return 1
        ########################
    
    def pick_leaf(self) -> MCTSNode:
        # select the leaf node to expand
        # the leaf node is the node that has not been expanded
        # create and return a new node if game is not ended
        
        ########################
        # TODO: your code here #
        ########################
        return self.root
        ########################
    
    def get_policy(self, node:MCTSNode = None) -> np.ndarray:
        # return the policy of the tree(root) after the search
        # the policy conmes from the visit count of each action 
        
        ########################
        # TODO: your code here #
        ########################
        return np.ones(len(node.child_N_visit)) / len(node.child_N_visit)
        ########################

    def search(self):
        # search the tree for n_search times
        # eachtime, pick a leaf node, rollout the game (if game is not ended) 
        #   for n_rollout times, and backup the value.
        # return the policy of the tree after the search
        for _ in range(self.config.n_search):
            leaf = self.pick_leaf()
            value = 0
            if leaf.done:
                ########################
                # TODO: your code here #
                ########################
                pass 
                ########################
            else:
                ########################
                # TODO: your code here #
                ########################
                pass
                ########################
            self.backup(leaf, value)

        return self.get_policy(self.root)