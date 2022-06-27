import argparse
import os

import torch
import pickle
# from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import nan_euclidean_distances
# from scanpy.external.pp import magic
# from scanpy.external.pp import dca
import magic
from anndata import AnnData

from utils.dataset import InputDataset
from utils.data_utils import *


# ------------------------------------------------
# Global Parameter Initialisation
# ------------------------------------------------

if __name__ == '__main__':
    if not os.path.exists(f'data_train.idx'):
        mat = read_data(path=f'data_train.csv')
        with open(f'data_train.idx', 'wb') as f:
            pickle.dump(mat, f)
    else:
        print(f"Loading old matrix from data_train.idx")
        mat = np.load(f'data_train.idx', allow_pickle=True)
    print(f'Sparsity: {np.count_nonzero(np.isnan(mat)) / (mat.shape[0]*mat.shape[1])}% NaNs in the data')
    print(f'Ouput matric of shape {mat.shape}')

    # data, val_data = torch.utils.data.random_split(InputDataset(mat), [6000, 4000])

    # Information Gathering
    # Threshold here: 1% of the data.
    thresh_num_ratings_user = 100 # Different for users and items?
    thresh_num_ratings_item = 10
    user_count = 0
    item_count = 0

    user_nans = []
    item_nans = []
    for c in range(mat.shape[1]):
        user_nans.append(np.count_nonzero(~np.isnan(mat[:, c])))
        if np.count_nonzero(~np.isnan(mat[:, c])) < thresh_num_ratings_user:
            user_count = user_count + 1
    for i in range(mat.shape[0]):
        item_nans.append(np.count_nonzero(~np.isnan(mat[i, :])))
        if np.count_nonzero(~np.isnan(mat[i, :])) < thresh_num_ratings_item:
            item_count = item_count + 1

    print(f"Number of users with <{thresh_num_ratings_user} recorded ratings: {user_count}")
    print(f"Number of items with <{thresh_num_ratings_item} recorded ratings: {item_count}")

    # Calculating affinities

    # TODO: Implement in methods