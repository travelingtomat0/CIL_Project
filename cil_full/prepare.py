import util
import numpy as np
import sklearn.model_selection


def get_crossval_data(indices):
    """
    Generates 10 folds on the given indices, each consisting of train and validation data.
    """
    kf = sklearn.model_selection.KFold(n_splits=10, shuffle=True, random_state=42)
    res = []
    for train_index, test_index in kf.split(indices):
        res.append((indices[train_index], indices[test_index]))
    return res


def run():
    """
    Prepares the index data for all further steps.
    """
    print("Loading training data...")
    idx = util.load_indices("input/data_train.csv")
    util.cluster_prep(idx, mean_impute=False)

    print("Generating folds...")
    for i, (train_idx, val_idx) in enumerate(get_crossval_data(idx)):
        np.save(f"temp/idx_cv_train_{i}", train_idx)
        np.save(f"temp/idx_cv_val_{i}", val_idx)

    np.save("temp/idx_train", idx)

    print("Loading submission data...")
    np.save("temp/idx_sub", util.load_indices("input/sampleSubmission.csv"))

    print("Done.")

if __name__ == '__main__':
    run()
