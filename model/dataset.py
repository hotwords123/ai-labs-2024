from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class GameDataset(Dataset):
    def __init__(self, samples, transform=None):
        """
        Args:
            samples: a list of tuples containing (observation, policy, value)
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        observation, policy, value = self.samples[idx]
        if self.transform is not None:
            observation = self.transform(observation)
        policy = np.array(policy, dtype=np.float32)
        value = np.float32(value)
        return observation, {'policy': policy, 'value': value}
