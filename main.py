import os

import pickle

import numpy as np

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
    thresh_num_ratings_user = 10 # Different for users and items?
    thresh_num_ratings_item = 100
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

    if os.path.exists("wixxpisse123.txt"):
        from kakawasser import kaka
        kaka(mat)
    elif os.path.exists("its_me.txt"):
        pass
        import magic
        from kakawasser import kaka
        from scipy.spatial.distance import pdist, squareform
        from sklearn.decomposition import TruncatedSVD
        from sklearn.neighbors import NearestNeighbors
        from sklearn.metrics import mean_squared_error

        # TODO: Look at initial impute! Is there a better method?

        initial_impute = mean_user(mat).T
        svd = TruncatedSVD(n_components=10, n_iter=7, random_state=42)
        X_new = svd.fit_transform(initial_impute)

        nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(X_new)
        distances, indices = nbrs.kneighbors(X_new)

        count = 0
        error = 0
        """for i in range(X_new.shape[0]):
            for j in indices[i]:
                error = error + mean_squared_error(initial_impute[i, :], initial_impute[j, :])
                count = count + 1
        print(error / count)"""

        print("Impute from Neighbors")
        mat = mat.T
        for i in range(X_new.shape[0]):
            print(i/X_new.shape[0])
            for j in range(indices.shape[0]):
                mat[i, np.isnan(mat[i, :])] = mat[j, np.isnan(mat[i, :])]
            # initial_impute[i, :] = np.mean(np.vstack((initial_impute[indices[i], :], initial_impute[i, :])), axis=0)

        X_predict = initial_impute

        kaka(mat)
        df = prediction_data(mat)
    
    # Calculating affinities

    # TODO: Implement in methods