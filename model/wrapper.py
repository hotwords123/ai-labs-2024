import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.append('../../')
from torch import nn

import torch
import torch.optim as optim

from env.base_env import BaseGame
from .example_net import ConvNet

import logging
logger = logging.getLogger(__name__)

# TODO: Refactor this later (dcy11011)
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
    
class ModelTrainingConfig:
    def __init__(
        self, lr:float=0.0007, 
        dropout:float=0.3, 
        epochs:int=20, 
        batch_size:int=128, 
        num_channels:int=512,
        weight_decay=1e-6,
    ):
        self.lr = lr
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.weight_decay = weight_decay

class AverageMeter(object):

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ModelWrapper():
    def __init__(self, observation_size:tuple[int, int], action_size:int, net:ConvNet, config:ModelTrainingConfig=None):
        self.net = net
        self.observation_size = observation_size
        self.board_x, self.board_y = observation_size
        self.action_size = action_size
        if config is None:
            config = ModelTrainingConfig()
        self.config = config
    
    @property
    def device(self):
        return self.net.device
    
    @device.setter
    def device(self, value):
        self.net.device = value
        
    def to(self, device):
        self.net.to(device)
        self.net.device = device
        return self
    
    def copy(self):
        return ModelWrapper(self.observation_size, self.action_size, self.net.__class__(self.observation_size, self.action_size, self.net.config, self.net.device))
    
    def train(self, examples):
        optimizer = optim.Adam(self.net.parameters(), weight_decay=self.config.weight_decay)
        t = tqdm(range(self.config.epochs), desc='Training Net')
        for epoch in t:
            # logger.info('EPOCH ::: ' + str(epoch + 1))
            self.net.train()

            batch_count = int(len(examples) / self.config.batch_size)

            for bc in range(batch_count):
                sample_ids = np.random.randint(len(examples), size=self.config.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64)).to(self.net.device)
                target_pis = torch.FloatTensor(np.array(pis)).to(self.net.device)
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64)).to(self.net.device)

                # compute output
                out_pi, out_v = self.net(boards)
                ############################################################
                #                  TODO: Your Code Here                    #
                # compute loss here
                total_loss = 0
                ############################################################

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, board):
        """
        board: np array with board
        """
        # preparing input
        board = torch.FloatTensor(board.astype(np.float64)).to(self.net.device)
        board = board.view(1, self.board_x, self.board_y)
        self.net.eval()
        with torch.no_grad():
            pi, v = self.net(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            logger.warning("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            logger.debug("Checkpoint Directory exists. ")
        torch.save({
            'state_dict': self.net.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = self.net.device
        checkpoint = torch.load(filepath, map_location=map_location)
        self.net.load_state_dict(checkpoint['state_dict'])
