import scipy.sparse

def preprocess_matrix(mat, file):
  rows = np.sum(mat > 0)

  mask = np.zeros_like(mat)
  mask[mat > 0] = 1

  user_rated_movies_indices = []
  user_rated_movies_data = []

  num_ratings = mask.sum(axis=1).astype(int)
  scalings = 1 / np.sqrt(num_ratings)


  for u in range(10000):
    user_rated_movies_indices.append(list(11000 + np.ravel(np.argwhere(mat[u] > 0))))
    user_rated_movies_data.append(list(np.ones(num_ratings[u]) * scalings[u]))

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
    
    # Last shit
    row_ind.extend([num] * num_ratings[u])
    col_ind.extend(user_rated_movies_indices[u])
    data.extend(user_rated_movies_data[u])

  sparse_mat = scipy.sparse.csc_matrix((data, (row_ind, col_ind)), shape=(rows, 12000))
  scipy.sparse.save_npz(file, sparse_mat)

preprocess_matrix(mat_validation, "/content/drive/MyDrive/sparse_mat_val.npz")
