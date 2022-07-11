import torch
import numpy as np

from torch.utils.data import IterableDataset, Dataset, random_split, ConcatDataset


class InputDataset(Dataset):

    def __init__(self, matrix):
        super(InputDataset, self).__init__()
        self.X = matrix
        self.length = matrix.shape[1]
        self.masked_X = None

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.masked_X is None:
            return torch.from_numpy(self.X[:, index]), torch.from_numpy(self.X[:, index])
        else:
            return torch.from_numpy(self.masked_X[:, index]), torch.from_numpy(self.X[:, index])


class ValDataset(Dataset):

    def __init__(self, input_matrix, output_matrix):
        super(ValDataset, self).__init__()
        self.X = input_matrix
        self.Y = output_matrix
        self.length = input_matrix.shape[1]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return torch.from_numpy(self.X[:, index]), torch.from_numpy(self.Y[:, index])

