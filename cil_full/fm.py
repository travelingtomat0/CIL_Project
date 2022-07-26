from typing import Dict, List

import myfm
import numpy as np
import pandas as pd
from myfm import RelationBlock
from myfm.utils.callbacks.libfm import (
    OrderedProbitCallback,
    RegressionCallback,
)
from myfm.utils.encoders import CategoryValueToSparseEncoder
from scipy import sparse as sps
import util


def run_algorithm(algorithm, idx_train, idx_val, embedding_size=18, verbose=False):
    df_train = pd.DataFrame(idx_train, columns=["user_id", "movie_id", "rating"])
    df_test = pd.DataFrame(idx_val, columns=["user_id", "movie_id", "rating"])

    if algorithm == "oprobit":
        # interpret the rating (1, 2, 3, 4, 5) as class (0, 1, 2, 3, 4).
        for df_ in [df_train, df_test]:
            df_["rating"] -= 1
            df_["rating"] = df_.rating.astype(np.int32)

    implicit_data_source = df_train

    user_to_internal = CategoryValueToSparseEncoder[int](
        implicit_data_source.user_id.values
    )
    movie_to_internal = CategoryValueToSparseEncoder[int](
        implicit_data_source.movie_id.values
    )

    movie_vs_watched: Dict[int, List[int]] = dict()
    user_vs_watched: Dict[int, List[int]] = dict()

    for row in implicit_data_source.itertuples():
        user_id = row.user_id
        movie_id = row.movie_id
        movie_vs_watched.setdefault(movie_id, list()).append(user_id)
        user_vs_watched.setdefault(user_id, list()).append(movie_id)

    X_train, X_test = (None, None)

    # setup grouping
    feature_group_sizes = []

    feature_group_sizes.append(len(user_to_internal))  # user ids

    # all movies which a user watched
    feature_group_sizes.append(len(movie_to_internal))

    feature_group_sizes.append(len(movie_to_internal))  # movie ids

    feature_group_sizes.append(
        len(user_to_internal)  # all the users who watched a movies
    )

    grouping = [i for i, size in enumerate(feature_group_sizes) for _ in range(size)]

    def augment_user_id(user_ids: List[int]) -> sps.csr_matrix:
        X = user_to_internal.to_sparse(user_ids)
        data: List[float] = []
        row: List[int] = []
        col: List[int] = []
        for index, user_id in enumerate(user_ids):
            watched_movies = user_vs_watched.get(user_id, [])
            normalizer = 1 / max(len(watched_movies), 1) ** 0.5
            for mid in watched_movies:
                data.append(normalizer)
                col.append(movie_to_internal._get_index(mid))
                row.append(index)
        return sps.hstack(
            [
                X,
                sps.csr_matrix(
                    (data, (row, col)),
                    shape=(len(user_ids), len(movie_to_internal)),
                ),
            ],
            format="csr",
        )

    def augment_movie_id(movie_ids: List[int]):
        X = movie_to_internal.to_sparse(movie_ids)
        data: List[float] = []
        row: List[int] = []
        col: List[int] = []

        for index, movie_id in enumerate(movie_ids):
            watched_users = movie_vs_watched.get(movie_id, [])
            normalizer = 1 / max(len(watched_users), 1) ** 0.5
            for uid in watched_users:
                data.append(normalizer)
                row.append(index)
                col.append(user_to_internal._get_index(uid))
        return sps.hstack(
            [
                X,
                sps.csr_matrix(
                    (data, (row, col)),
                    shape=(len(movie_ids), len(user_to_internal)),
                ),
            ]
        )

    # Create RelationBlock.
    train_blocks: List[RelationBlock] = []
    test_blocks: List[RelationBlock] = []
    for source, target in [(df_train, train_blocks), (df_test, test_blocks)]:
        unique_users, user_map = np.unique(source.user_id, return_inverse=True)
        target.append(RelationBlock(user_map, augment_user_id(unique_users)))
        unique_movies, movie_map = np.unique(source.movie_id, return_inverse=True)
        target.append(RelationBlock(movie_map, augment_movie_id(unique_movies)))

    if algorithm == "regression":
        fm = myfm.MyFMRegressor(rank=embedding_size)
        callback = RegressionCallback(
            600,
            X_test,
            df_test.rating.values,
            X_rel_test=test_blocks,
            clip_min=0.5,
            clip_max=5.0,
            trace_path="temp/fm_trace.csv",
        )
    elif algorithm == "variational":
        fm = myfm.VariationalFMRegressor(rank=embedding_size)
        callback = RegressionCallback(
            600,
            X_test,
            df_test.rating.values,
            X_rel_test=test_blocks,
            clip_min=0.5,
            clip_max=5.0,
            trace_path="temp/fm_trace.csv",
        )
    else:
        fm = myfm.MyFMOrderedProbit(rank=embedding_size)
        callback = OrderedProbitCallback(
            600,
            X_test,
            df_test.rating.values,
            n_class=5,
            X_rel_test=test_blocks,
            trace_path="temp/fm_trace.csv",
        )

    fm.fit(
        X_train,
        df_train.rating.values,
        X_rel=train_blocks,
        grouping=grouping,
        n_iter=callback.n_iter,
        callback=callback if verbose else None
    )

    if algorithm == "oprobit":
        return np.dot(fm.predict_proba(X_test, X_rel=test_blocks), np.arange(1, 6))
    else:
        return fm.predict(X_test, X_rel=test_blocks)


def run(algorithm, embedding_size=18):
    ##############################
    # Cross-validation
    print("Cross validation...")

    with open(f"output/{algorithm}_fm_cv.txt", "w") as f:
        rmses = []
        for i in range(10):
            print(f"{i + 1} / 10:")

            idx_train = util.load_indices(f"temp/idx_cv_train_{i}.npy")
            idx_val = util.load_indices(f"temp/idx_cv_val_{i}.npy")

            predictions = run_algorithm(algorithm, idx_train, idx_val, embedding_size)
            print(predictions)

            rmse = util.rmse(idx_val[:, 2], predictions)
            print(f"RMSE: {rmse}")
            f.write(f"{rmse}\n")
            rmses.append(rmse)
        f.write(f"mean: {np.mean(rmses)}, std: {np.std(rmses)}")

    ##############################
    # Submission
    print("Submission...")

    idx_full = util.load_indices(f"temp/idx_train.npy")
    idx_sub = util.load_indices("temp/idx_sub.npy")

    predictions = run_algorithm(algorithm, idx_full, np.concatenate([idx_full, idx_sub]), embedding_size)
    util.export_indices_to_csv(idx_full, predictions[:idx_full.shape[0]], f"output/{algorithm}_fm_train.csv")
    util.export_indices_to_csv(idx_sub, predictions[idx_full.shape[0]:idx_full.shape[0] + idx_sub.shape[0]],
                              f"output/{algorithm}_fm_sub.csv", clip=True)

    print("Done.")


if __name__ == '__main__':
    print("--- Regression FM ---")
    run("regression")
