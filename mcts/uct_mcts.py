from .node import MCTSNode, INF
from .config import MCTSConfig
from env.base_env import BaseGame

import numpy as np
import itertools

class UCTMCTSConfig(MCTSConfig):
    def __init__(
        self,
        n_rollout:int = 1,
        *args, **kwargs
    ):
        MCTSConfig.__init__(self, *args, **kwargs)
        self.n_rollout = n_rollout


class UCTMCTS:
    def __init__(self, init_env:BaseGame, config: UCTMCTSConfig, root:MCTSNode=None):
        self.config = config
        self.root = root
        if root is None:
            self.init_tree(init_env)
        self.root.cut_parent()
    
    def init_tree(self, init_env:BaseGame):
        # initialize the tree with the current state
        # fork the environment to avoid side effects
        env = init_env.fork()
        self.root = MCTSNode(
            action=None, env=env, reward=0,
        )
    
    def get_subtree(self, action:int):
        # return a subtree with root as the child of the current root
        # the subtree represents the state after taking action
        if self.root.has_child(action):
            new_root = self.root.get_child(action)
            return UCTMCTS(new_root.env, self.config, new_root)
        else:
            return None
    
    def uct_action_select(self, node:MCTSNode) -> int:
        """
        select the best action based on UCB when expanding the tree
        """
        # collect the available actions
        actions = np.nonzero(node.action_mask)[0]
        V_total = node.child_V_total[node.action_mask]
        N_visit = node.child_N_visit[node.action_mask]
        
        # if there are unvisited actions, return one of them
        if np.any(N_visit == 0):
            return np.random.choice(actions[N_visit == 0])
        
        # otherwise, return the action with the highest UCB
        ucb = V_total / N_visit + self.config.C * np.sqrt(np.log(np.sum(N_visit)) / N_visit)
        return actions[np.argmax(ucb)]

    def backup(self, node:MCTSNode, value:float) -> None:
        """
        backup the value of the leaf node to the root
        update N_visit and V_total of each node in the path
        """
        while (parent := node.parent) is not None:
            # update the statistics of the node
            action = node.action
            parent.child_V_total[action] += value
            parent.child_N_visit[action] += 1
            # move to the parent node
            node = parent
            value = -value
    
    def rollout(self, node:MCTSNode) -> float:
        """
        simulate the game until the end
        return the reward of the game
        NOTE: the reward is converted to the perspective of the current player
        """
        env = node.env.fork()

        for turn in itertools.cycle((0, 1)):
            # pick a random action
            actions = np.nonzero(env.action_mask)[0]
            action = np.random.choice(actions)
            # take the action
            _, reward, done = env.step(action, return_obs=False)
            if done:
                # return the reward from the perspective of the current player
                return reward if turn == 0 else -reward
    
    def pick_leaf(self) -> MCTSNode:
        """
        select the leaf node to expand
        the leaf node is the node that has not been expanded
        create and return a new node if game is not ended
        """
        node = self.root
        while not node.done:
            action = self.uct_action_select(node)
            if node.has_child(action):
                node = node.get_child(action)
            else:
                return node.add_child(action)
        return node
    
    def get_policy(self, node:MCTSNode = None) -> np.ndarray:
        """
        return the policy of the tree (root) after the search
        the policy comes from the visit count of each action 
        """
        node = self.root if node is None else node
        return node.child_N_visit / np.sum(node.child_N_visit)

    def search(self):
        """
        search the tree for n_search times
        each time, pick a leaf node, rollout the game (if game is not ended) 
        for n_rollout times, and backup the value.
        return the policy of the tree after the search
        """
        assert not self.root.done, "The game has ended"

        for _ in range(self.config.n_search):
            leaf = self.pick_leaf()
            value = 0
            if leaf.done:
                # if the game is ended, the value is the final reward
                value = leaf.reward
            else:
                # rollout the game for n_rollout times and average the value
                for _ in range(self.config.n_rollout):
                    value += self.rollout(leaf)
                value /= self.config.n_rollout
            # backup the value to the root
            self.backup(leaf, value)

        return self.get_policy(self.root)