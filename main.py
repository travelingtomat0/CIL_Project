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

parser = argparse.ArgumentParser(description='Run Prediction Script.')
parser.add_argument('path', type=str, default='/home/phil/Documents/Studium/Courses/CIL/Project',
                    help='path to training data.')
parser.add_argument('--debug', action='store_true',
                    help='set --debug flag for lower computational impact.')
parser.add_argument('--overwrite', action='store_true',
                    help='set --overwrite flag to overwrite the matrix generation.')
args = parser.parse_args()
path = args.path

if __name__ == '__main__':
    if not os.path.exists(f'{path}/data_train.idx') or args.overwrite:
        mat = read_data(path=f'{path}/data_train.csv')
        with open(f'{path}/data_train.idx', 'wb') as f:
            pickle.dump(mat, f)
    else:
        print(f"Loading old matrix from {path}/data_train.idx")
        mat = np.load(f'{path}/data_train.idx', allow_pickle=True)
    print(f'Sparsity: {np.count_nonzero(np.isnan(mat)) / (mat.shape[0]*mat.shape[1])}% NaNs in the data')
    print(f'Ouput matric of shape {mat.shape}')

    data, val_data = torch.utils.data.random_split(InputDataset(mat), [6000, 4000])

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

    dist_mat = nan_euclidean_distances(X=mat.T)
    print(dist_mat)
    #adata_mat = AnnData(X=np.nan_to_num(x=mat))
    #print(adata_mat.X)
    # imputed = dca(adata_mat)
    # To use the below, run 'pip install --user magic-impute'
    # magic_operator = magic.MAGIC(knn=5)
    # print(np.nan_to_num(x=mat, nan=0))
    # X_magic = magic_operator.fit_transform(np.nan_to_num(x=mat, nan=0))
    # print(mat)
    # print(10*X_magic)
    # imputed = magic(adata_mat)
    # imputed = magic(adata_mat)
    # print(imputed.shape)
    # user_nans.sort()
    # item_nans.sort()
    # print(user_nans)
    # print(item_nans)

    print(f"Number of users with <{thresh_num_ratings_user} recorded ratings: {user_count}")
    print(f"Number of items with <{thresh_num_ratings_item} recorded ratings: {item_count}")

    # Reduce number of users (10% of the data) to save resources during debugging
    if args.debug:
        mat = mat[:, :100]

    # Calculating affinities

    # TODO: Implement in methods