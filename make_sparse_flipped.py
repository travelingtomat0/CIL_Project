import scipy.sparse
import numpy as np

def preprocess_matrix(mat, file):
  rows = np.sum(mat > 0)

  mask = np.zeros_like(mat)
  mask[mat > 0] = 1

  user_rated_movies_indices = []
  user_rated_movies_data = []

  num_ratings_per_user = mask.sum(axis=1).astype(int) # user rated that many movies
  scalings_rating_per_user = 1 / np.sqrt(num_ratings_per_user) # normalization

  num_ratings_per_movie = mask.sum(axis=0).astype(int) # movie got rated by that many users
  scalings_rating_per_movie = 1 / np.sqrt(num_ratings_per_movie) # normalization


  for u in range(10000):
    user_rated_movies_indices.append(list(11000 + np.ravel(np.argwhere(mat[u] > 0))))
    user_rated_movies_data.append(list(np.ones(num_ratings_per_user[u]) * scalings_rating_per_user[u]))


  movie_got_rated_by_users_indices = []
  movie_got_rated_by_users_data = []


  for i in range(1000):
    movie_got_rated_by_users_indices.append(list(12000 + np.ravel(np.argwhere(mat[:, i] > 0))))
    movie_got_rated_by_users_data.append(list(np.ones(num_ratings_per_movie[i]) * scalings_rating_per_movie[i]))
    

  data = []
  row_ind = []
  col_ind = []

  import tqdm

  for num, (u, i) in tqdm.tqdm(enumerate(np.argwhere(mat))):
    # One-hot user
    row_ind.append(num)
    col_ind.append(u)
    data.append(1)

    # One-hot movie
    row_ind.append(num)
    col_ind.append(i + 10000)
    data.append(1)
    
    # bag of words: specific user rated movies
    row_ind.extend([num] * num_ratings_per_user[u])
    col_ind.extend(user_rated_movies_indices[u])
    data.extend(user_rated_movies_data[u])

    # bag of words: specific movie got rated by these users
    row_ind.extend([num] * num_ratings_per_movie[i])
    col_ind.extend(movie_got_rated_by_users_indices[i])
    data.extend(movie_got_rated_by_users_data[i])



  sparse_mat = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(rows, 22000))
  scipy.sparse.save_npz(file, sparse_mat)

#preprocess_matrix(mat_validation, "/content/drive/MyDrive/sparse_mat_val.npz")
