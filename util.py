import pandas as pd
import numpy as np
import sklearn.model_selection
import tqdm
import os.path

# Imports for cluster label generation:
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.impute import SimpleImputer


def load_indices(file):
    """
    Loads the indices from a file. If a .csv file is given, it parses the file, otherwise it assumes the indices are
    preprocessed and in binary .npy format.
    """
    if os.path.basename(file)[-3:] == "csv":
        data = pd.read_csv(file).to_numpy()
        results = []
        for i in tqdm.tqdm(range(data.shape[0])):
            row = data[i]
            a, b = row[0].split("_")
            m, n = int(a[1:]) - 1, int(b[1:]) - 1
            results.append([m, n, row[1]])
        return np.asarray(results)
    else:
        return np.load(file)


def indices_to_matrix(indices):
    """
    Converts the indices into a dense 10k x 1k matrix.
    """
    mat = np.zeros((10000, 1000))
    mat[indices[:, 0].astype(int), indices[:, 1].astype(int)] = indices[:, 2].astype(float)
    return mat


def export_matrix_to_csv(mat, file, export_train=False, export_submission=False, clip=False):
    """
    Exports the given data into a submittable CSV file.
    """
    to_export = []
    if export_train:
        to_export.append(load_indices("temp/idx_train.npy"))
    if export_submission:
        to_export.append(load_indices("temp/idx_sub.npy"))
    target_indices = np.concatenate(to_export)

    if clip:
        mat = np.clip(mat, 1, 5)

    with open(file, "w") as f:
        f.write("Id,Prediction\n")
        for t in target_indices:
            f.write(f"r{t[0] + 1}_c{t[1] + 1},{mat[t[0], t[1]]}\n")


def export_indices_to_csv(indices, values, file, clip=False):
    """
    Exports the given indices and predictions into a submittable CSV file.
    """
    if clip:
        values = np.clip(values, 1, 5)

    with open(file, "w") as f:
        f.write("Id,Prediction\n")
        for i, t in enumerate(indices):
            f.write(f"r{t[0] + 1}_c{t[1] + 1},{values[i]}\n")


def rmse(y_true, y_pred):
    """
    Computes the root-mean-squared-error.
    """
    return np.sqrt(np.mean(np.square(y_true - y_pred)))
