from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
from surprise import Trainset
from surprise import Dataset, Reader
from surprise.prediction_algorithms.matrix_factorization import SVD
import util
from tqdm import tqdm
import numpy as np
import pandas as pd

# CV: 1.0008940820926258 (std: 0.0016295044878154525)
# Public score: 0.99694

def in_question(path):
    df = pd.read_csv(path)
    res = []
    for i in range(df.shape[0]):
        user, item = df['Id'][i].split('_')
        user = int(user[1:]) - 1
        item = int(item[1:]) - 1
        res.append((item, user))
    return res

def run():
    print("--- Simon Funks SVD ---")
    
    """reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(input_df[['userID', 'itemID', 'rating']], reader)"""
    
    ##############################
    # Cross-validation
    print("Cross validation...")

    with open("output/funk_svd_cv.txt", "w") as f:
        rmses = []
        for i in range(10):
            print(f"{i + 1} / 10:")
            
            m = np.load(f"temp/idx_cv_train_{i}.npy")
            df_train = pd.DataFrame(m, columns=['userID', 'itemID', 'rating'])
            
            m = np.load(f"temp/idx_cv_val_{i}.npy")
            df_val = pd.DataFrame(m, columns=['userID', 'itemID', 'rating'])
            mat_val = util.indices_to_matrix(util.load_indices(f"temp/idx_cv_val_{i}.npy"))

            model = SVD(n_factors=12, verbose=True)
            reader = Reader(rating_scale=(1, 5))
            trainset = Dataset.load_from_df(df_train[['userID', 'itemID', 'rating']], reader)
            trainset = trainset.build_full_trainset()
            # testset = Dataset.load_from_df(df_val[['userID', 'itemID', 'rating']], reader)
            # testset = testset.build_full_trainset()
            #mat_train_imp = imp_mean.fit_transform(mat_train)
            model.fit(trainset)

            predictions = np.zeros((10000, 1000))
            for j in range(df_val.shape[0]):
                # print(df_val.iloc[j])
                predictions[df_val.iloc[j]['userID'], df_val.iloc[j]['itemID']] = model.predict(df_val.iloc[j]['userID'], df_val.iloc[j]['itemID'], verbose=False)[3]

            predictions = predictions[mat_val > 0]

            rmse = util.rmse(mat_val[mat_val > 0], predictions)
            print(f"RMSE: {rmse}")
            f.write(f"{rmse}\n")
            rmses.append(rmse)
        f.write(f"mean: {np.mean(rmses)}, std: {np.std(rmses)}")

    ##############################
    # Submission
    print("Submission...")

    mat_full = np.load(f"temp/idx_train.npy")
    df = pd.DataFrame(mat_full, columns=['userID', 'itemID', 'rating'])
    reader = Reader(rating_scale=(1, 5))
    
    data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
    data = data.build_full_trainset()
    # util.indices_to_matrix(util.load_indices(f"temp/idx_train.npy"))
    model = SVD(n_factors=12, verbose=False)
    
    model.fit(data)
    # mat_full_imp = imp_mean.fit_transform(mat_full)
    
    ids = in_question('input/sampleSubmission.csv')
    predicted_mat = np.zeros((10000, 1000))
    for itm, usr in ids:
        predicted_mat[usr, itm] = model.predict(usr, itm)[3]
    
    """for usr, itm in tqdm(zip(non_nan_users, non_nan_items), total=len(non_nan_users)):
        predicted_mat[itm, usr] = model.predict(usr,itm)[3]"""
    
    # util.export_matrix_to_csv(predicted_mat, "output/funk_svd_train.csv", export_train=True)
    util.export_matrix_to_csv(predicted_mat, "output/funk_svd_sub.csv", export_submission=True, clip=True)

    print("Done.")

if __name__ == '__main__':
    run()
