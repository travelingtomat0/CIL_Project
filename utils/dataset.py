import torch
import numpy as np

from torch.utils.data import IterableDataset, Dataset, random_split, ConcatDataset

class InputDataset(Dataset):

    def __init__(self, matrix):
        super(InputDataset, self).__init__()
        self.X = matrix
        self.length = matrix.shape[1]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return torch.from_numpy(self.X[:, index]), torch.from_numpy(self.X[:, index])

