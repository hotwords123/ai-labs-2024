import os

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from env.base_env import BaseGame
from .model import AlphaZeroNet
from .dataset import GameDataset

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
        weight_decay=1e-6,
    ):
        self.lr = lr
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(self, pred, gt):
        loss = self.cross_entropy(pred['policy'], gt['policy'])
        loss += self.mse(pred['value'], gt['value'])
        return loss


def to_device(data: object, device: torch.device) -> object:
    if isinstance(data, tuple):
        return tuple(to_device(x, device) for x in data)
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    return data.to(device)


class ModelWrapper():
    def __init__(self, observation_size:tuple[int, int], action_size:int, net:AlphaZeroNet, config:ModelTrainingConfig=None):
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
    
    def transform_data(self, observation: np.ndarray):
        return np.stack((observation == 1, observation == -1), axis=0, dtype=np.float32)

    def train(self, examples):
        dataset = GameDataset(examples, transform=self.transform_data)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        criterion = CombinedLoss()

        optimizer = optim.Adam(self.net.parameters(), weight_decay=self.config.weight_decay)

        t = tqdm(range(self.config.epochs), desc='Training Net')
        for epoch in t:
            # logger.info('EPOCH ::: ' + str(epoch + 1))
            self.net.train()

            progress_bar = tqdm(dataloader, total=len(dataloader), leave=False)
            for x, y in progress_bar:
                x, y = to_device(x, self.device), to_device(y, self.device)

                optimizer.zero_grad()
                out = self.net(x)

                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

                progress_bar.set_postfix(loss=loss.item())

    def predict(self, board):
        """
        board: np array with board
        """
        # preparing input
        board = self.transform_data(board)
        board = torch.tensor(board, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.net.eval()
        with torch.no_grad():
            out = self.net(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return F.softmax(out['policy'], dim=1).cpu().numpy()[0], out['value'].item()

    def save_checkpoint(self, file):
        torch.save({'state_dict': self.net.state_dict()}, file)

    def load_checkpoint(self, file):
        checkpoint = torch.load(file, map_location=self.device)
        self.net.load_state_dict(checkpoint['state_dict'])
