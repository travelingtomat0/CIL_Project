import pandas as pd
import numpy as np


# Returns np array of dimension (10.000, 1.000) where rows correspond to features and collumns correspond to
# user ratings.
def read_data(path='./', out_shape=(10000, 1000)):
    df = pd.read_csv(path)
    out_matrix = np.empty(out_shape)
    out_matrix[:] = np.nan
    # Build matrix for user preferences
    # Output matrix must be of dimension (10.000, 1.000)
    for i in range(df.shape[0]):
        item, user = df['Id'][i].split('_')
        user = int(user[1:])-1
        item = int(item[1:])-1
        # Catch erroneous user / item numbers?
        out_matrix[item, user] = df['Prediction'][i]
    return out_matrix


def paste_data(matrix):
    pass
