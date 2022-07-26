from scipy.sparse.linalg import svds
from sklearn.impute import SimpleImputer
import util
import numpy as np

# CV: 1.0303734482860514 (std: 0.0018637845093404358)
# Public score: 1.02982

def run():
    print("--- Mean movie impute ---")

    ##############################
    # Cross-validation
    print("Cross validation...")

    with open("output/mean_movie_cv.txt", "w") as f:
        rmses = []
        for i in range(10):
            print(f"{i + 1} / 10:")

            mat_train = util.indices_to_matrix(util.load_indices(f"temp/idx_cv_train_{i}.npy"))
            mat_val = util.indices_to_matrix(util.load_indices(f"temp/idx_cv_val_{i}.npy"))

            imp_mean = SimpleImputer(missing_values=0, strategy='mean')
            mat_train_imp = imp_mean.fit_transform(mat_train)
            predictions = mat_train_imp[mat_val > 0]
            rmse = util.rmse(mat_val[mat_val > 0], predictions)
            print(f"RMSE: {rmse}")
            f.write(f"{rmse}\n")
            rmses.append(rmse)
        f.write(f"mean: {np.mean(rmses)}, std: {np.std(rmses)}")

    ##############################
    # Submission
    print("Submission...")

    mat_full = util.indices_to_matrix(util.load_indices(f"temp/idx_train.npy"))
    imp_mean = SimpleImputer(missing_values=0, strategy='mean')
    mat_full_imp = imp_mean.fit_transform(mat_full)

    mat_full_imp[mat_full > 0] = 0
    imp_mean = SimpleImputer(missing_values=0, strategy='mean')
    mat_full_imp = imp_mean.fit_transform(mat_full_imp)

    util.export_matrix_to_csv(mat_full_imp, "output/mean_movie_train.csv", export_train=True)
    util.export_matrix_to_csv(mat_full_imp, "output/mean_movie_sub.csv", export_submission=True, clip=True)

    print("Done.")

if __name__ == '__main__':
    run()
