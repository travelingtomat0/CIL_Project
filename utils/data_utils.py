import pandas as pd
import numpy as np


# Returns np array of dimension (1.000, 10.000) where rows correspond to features and collumns correspond to
# user ratings.
def read_data(path='./', out_shape=(1000, 10000)):
    df = pd.read_csv(path)
    out_matrix = np.empty(out_shape)
    out_matrix[:] = np.nan
    # Build matrix for user preferences
    # Output matrix must be of dimension (10.000, 1.000)
    for i in range(df.shape[0]):
        user, item = df['Id'][i].split('_')
        user = int(user[1:])-1
        item = int(item[1:])-1
        # Catch erroneous user / item numbers?
        out_matrix[item, user] = df['Prediction'][i]
    return out_matrix


def in_question(path):
    df = pd.read_csv(path)
    res = []
    for i in range(df.shape[0]):
        user, item = df['Id'][i].split('_')
        user = int(user[1:]) - 1
        item = int(item[1:]) - 1
        res.append((item, user))
    return res


def prediction_data(matrix, num_items=10000, num_users=1000, out_path='./'):
    id_list = []
    pred_list = []
    for c in range(num_users):
        for i in range(num_items):
            # CAUTION: Add 1 because users & items start at 1, not 0.
            id_list.append(f'r{i+1}_c{c+1}')
            pred_list.append(matrix[num_items][num_users])
    df = pd.DataFrame(np.array([id_list, pred_list]).T, columns=['Id', 'Prediction'])
    return df


# An initial imputation, before running our models. This eliminates NaN's from the initial data, the optimal policy for
# initial imputation is a TODO.
def initial_impute():
    pass


def mean_user(mat):
    item_means = np.nanmean(mat, axis=1)
    print(item_means.shape)
    print(mat.shape)
    print(mat)
    for i in range(item_means.shape[0]):
        mat[i, :] = np.nan_to_num(x=mat[i, :], nan=round(item_means[i]))
    return mat
