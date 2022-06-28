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


# get the prediction id's that are in question from the sample csv
# returns tuples (item_id, user_id)
def in_question(path):
    df = pd.read_csv(path)
    res = []
    for i in range(df.shape[0]):
        user, item = df['Id'][i].split('_')
        user = int(user[1:]) - 1
        item = int(item[1:]) - 1
        res.append((item, user))
    return res


# Input shape of matrix: [num_users, num_items]
def prediction_data(matrix, num_items=10000, num_users=1000, out_path='prediction.csv'):
    assert matrix.shape[0] == 10000, f"Expected shape (10000, 1000), got: {matrix.shape}"
    ids = in_question('sampleSubmission.csv')
    id_list = []
    pred_list = []
    for (item, user) in ids:
        id_list.append(f'r{user+1}_c{item+1}')
        #pred_list.append(int(matrix[user][item]))
        pred_list.append(matrix[user][item])
    df = pd.DataFrame(np.array([id_list, pred_list]).T, columns=['Id', 'Prediction'])
    df.to_csv(out_path, index=False)
    return df


# An initial imputation, before running our models. This eliminates NaN's from the initial data, the optimal policy for
# initial imputation is a TODO.
def initial_impute():
    pass


def mean_user(mat):
    item_means = np.nanmean(mat, axis=1)
    for i in range(item_means.shape[0]):
        mat[i, :] = np.nan_to_num(x=mat[i, :], nan=round(item_means[i]))
    return mat


# input dimension [num_users, num_items]
def center_user(mat):
    assert mat.shape[0] == 10000, f"Expected shape (10000, 1000), got: {mat.shape}"
    user_means = np.nanmean(mat, axis=1)
    for i in range(mat.shape[0]):
        mat[i, :] = mat[i, :] - user_means[i]
    return user_means, mat
