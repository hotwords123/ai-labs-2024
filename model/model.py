import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple


class AlphaZeroNetConfig(NamedTuple):
    in_channels: int = 2
    num_filters: int = 64
    num_residual_blocks: int = 3
    policy_num_filters: int = 2
    value_num_filters: int = 1
    value_hidden_size: int = 256


class ResidualBlock(nn.Module):
    def __init__(self, num_filters: int = 64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += x
        out = F.relu(out)
        return out


class AlphaZeroNet(nn.Module):
    def __init__(
        self,
        observation_size: tuple[int, int],
        action_space_size: int,
        config: AlphaZeroNetConfig = AlphaZeroNetConfig(),
        device: torch.device = torch.device("cpu"),
    ):
        super(AlphaZeroNet, self).__init__()

        self.observation_size = observation_size
        self.action_space_size = action_space_size
        self.config = config
        self.device = device

        # Initial convolutional layer
        self.conv = nn.Conv2d(in_channels=config.in_channels, out_channels=config.num_filters, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(config.num_filters)

        # Residual blocks
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(num_filters=config.num_filters) for _ in range(config.num_residual_blocks)]
        )

        board_size = observation_size[0] * observation_size[1]

        # Policy head
        self.policy_conv = nn.Conv2d(in_channels=config.num_filters, out_channels=config.policy_num_filters, kernel_size=1, stride=1)
        self.policy_bn = nn.BatchNorm2d(config.policy_num_filters)
        self.policy_fc = nn.Linear(config.policy_num_filters * board_size, action_space_size)

        # Value head
        self.value_conv = nn.Conv2d(in_channels=config.num_filters, out_channels=config.value_num_filters, kernel_size=1, stride=1)
        self.value_bn = nn.BatchNorm2d(config.value_num_filters)
        self.value_fc1 = nn.Linear(config.value_num_filters * board_size, config.value_hidden_size)
        self.value_fc2 = nn.Linear(config.value_hidden_size, 1)

        self.to(device)

    def forward(self, x: torch.Tensor):
        # Initial convolutional layer
        x = F.relu(self.bn(self.conv(x)))

        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value)).squeeze(1)

        return {'policy': policy, 'value': value}
