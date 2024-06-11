import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from env.base_env import BaseGame

class BaseNetConfig:
    def __init__(
        self, 
        num_channels:int = 256,
        dropout:float = 0.3,
        linear_hidden:list[int] = [256, 128],
    ):
        self.num_channels = num_channels
        self.linear_hidden = linear_hidden
        self.dropout = dropout
        
class MLPNet(nn.Module):
    def __init__(self, observation_size:tuple[int, int], action_space_size:int, config:BaseNetConfig, device:torch.device='cpu'):
        super().__init__()
        self.config = config
        self.device = device
        input_dim = observation_size[0] * observation_size[1]
        self.layer1 = nn.Linear(input_dim, config.linear_hidden[0])
        self.layer2 = nn.Linear(config.linear_hidden[0], config.linear_hidden[1])
        
        self.policy_head = nn.Linear(config.linear_hidden[1], action_space_size)
        self.value_head = nn.Linear(config.linear_hidden[1], 1)
        self.relu = nn.ReLU()
        self.to(device)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        pi = self.policy_head(x)
        v = self.value_head(x)
        return F.log_softmax(pi, dim=1), torch.tanh(v)

class ConvNet(nn.Module):
    def __init__(self, observation_size:tuple[int, int], action_space_size:int, config:BaseNetConfig, device:torch.device='cpu'):
        # game params
        self.board_x, self.board_y = observation_size
        self.action_size = action_space_size
        self.config = config
        self.device = device

        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, config.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(config.num_channels, config.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(config.num_channels, config.num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(config.num_channels)
        self.bn2 = nn.BatchNorm2d(config.num_channels)
        self.bn3 = nn.BatchNorm2d(config.num_channels)
        self.bn4 = nn.BatchNorm2d(config.num_channels)
        
        linear_hidden = config.linear_hidden

        self.fc1 = nn.Linear(config.num_channels*(self.board_x-2)*(self.board_y-2), linear_hidden[0])
        self.fc_bn1 = nn.BatchNorm1d(linear_hidden[0])

        self.fc2 = nn.Linear(linear_hidden[0], linear_hidden[1])
        self.fc_bn2 = nn.BatchNorm1d(linear_hidden[1])

        self.policy_head = nn.Linear(linear_hidden[1], self.action_size)

        self.value_head = nn.Linear(linear_hidden[1], 1)
        
        self.to(device)

    def forward(self, s: torch.Tensor):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(-1, 1, self.board_x, self.board_y)                # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)

        s = s.view(-1, self.config.num_channels*(self.board_x-2)*(self.board_y-2))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.config.dropout, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.config.dropout, training=self.training)  # batch_size x 512

        pi = self.policy_head(s)                                                                         # batch_size x action_size
        v = self.value_head(s)                                                                          # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)

class MyNet(nn.Module):
    def __init__(self, observation_size:tuple[int, int], action_space_size:int, config:BaseNetConfig, device:torch.device='cpu'):
        super().__init__()
        self.config = config
        self.device = device
        ############################################################
        #                  TODO: Your Code Here                    #
        # Define your network here
        
        ############################################################
        self.to(device)
    
    def forward(self, s: torch.Tensor):
        ############################################################
        #                  TODO: Your Code Here                    #
        # Define your forward pass here
        
        return None, None
        ############################################################