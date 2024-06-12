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
        """
        select the best action based on UCB when expanding the tree
        """
        # collect the available actions
        actions = np.nonzero(node.action_mask)[0]
        V_total = node.child_V_total[node.action_mask]
        N_visit = node.child_N_visit[node.action_mask]
        priors = node.child_priors[node.action_mask]

        # calculate the UCB of each action
        ucb = np.divide(V_total, N_visit, out=np.zeros_like(V_total), where=N_visit > 0)
        ucb += self.config.C * priors * np.sqrt(np.sum(N_visit)) / (1 + N_visit)

        # return the action with the highest UCB
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
    
    def pick_leaf(self) -> MCTSNode:
        """
        select the leaf node to expand
        the leaf node is the node that has not been expanded
        create and return a new node if game is not ended
        """
        node = self.root
        while not node.done:
            action = self.puct_action_select(node)
            if node.has_child(action):
                node = node.get_child(action)
            else:
                return self.expand(node, action)
        return node
    
    def expand(self, node: MCTSNode, action: int) -> MCTSNode:
        """
        expand the leaf node by taking the given action
        """
        env = node.env.fork()
        obs, reward, _ = env.step(action)

        policy, value = self.model.predict(obs * env.current_player)

        child = MCTSNode(action=action, env=env, reward=reward, value=value, parent=node)
        child.set_prior(policy)

        node.children[action] = child
        return child
    
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
        eachtime, pick a leaf node, compute v&p with neural-network (if game is not ended) 
        , and backup the value.
        NOTE: the value returned by the neural-network is the value of the the leaf state
            maybe you should multiply it by -1 before backup
        return the policy of the tree after the search
        """
        assert not self.root.done, "The game has ended"

        for _ in range(self.config.n_search):
            leaf = self.pick_leaf()
            if leaf.done:
                # if the game has ended, the utility is the final reward
                value = leaf.reward
            else:
                # otherwise, evaluate the game using the model
                # negate the value because it is from the perspective of the opponent
                value = -leaf.value
            self.backup(leaf, value)

        return self.get_policy(self.root)