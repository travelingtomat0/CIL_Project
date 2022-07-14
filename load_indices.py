def load_indices(file):
  data = pd.read_csv(file).to_numpy()
  results = []
  for i in tqdm.tqdm(range(data.shape[0])):
      row = data[i]
      a, b = row[0].split("_")
      m, n = int(a[1:]) - 1, int(b[1:]) - 1
      results.append([m, n, row[1]])
  return np.asarray(results)
