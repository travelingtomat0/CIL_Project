import argparse
from utils.data_utils import *

# ------------------------------------------------
# Parameter Initialisation
# ------------------------------------------------

parser = argparse.ArgumentParser(description='Run Prediction Script.')
parser.add_argument('path', type=str, default='/home/phil/Documents/Studium/Courses/CIL/Project',
                    help='path to training data.')
parser.add_argument('--debug', action='store_true',
                    help='set --debug flag for lower computational impact.')
args = parser.parse_args()
path = args.path

if __name__ == '__main__':
    mat = read_data(path=f'{path}/data_train.csv')
    print(f'Sparsity: {np.count_nonzero(np.isnan(mat)) / (mat.shape[0]*mat.shape[1])}% NaNs in the data')

    # Reduce number of users (10% of the data)
    if args.debug:
        mat = mat[, :100]

    # Calculating affinities
    # TODO: Implement in methods