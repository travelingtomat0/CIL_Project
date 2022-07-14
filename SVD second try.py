import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import activations
import tensorflow_probability as tfp
import numpy as np
from utils.data_utils import get_average_rating
import pandas as pd

path = './data'
print("path declared")

data = pd.read_csv("./data/data_train.csv").to_numpy()

mat_total = np.zeros((10000, 1000))
mat_train = np.zeros((10000, 1000))
mat_validation = np.zeros((10000, 1000))

np.random.seed(42)
tf.random.set_seed(42)

mask = np.random.rand(len(data[:, 0])) < 0.85

count_val_items = 0
for i, row in enumerate(data):
  a, b = row[0].split("_")
  m, n = int(a[1:]) - 1, int(b[1:]) - 1
  mat_total[m, n] = row[1]

  if mask[i]:
    mat_train[m, n] = row[1]
  else:
    mat_validation[m, n] = row[1]
    count_val_items+=1


average_rating = get_average_rating(mat_train)

inner_dimension = 100

item_vectors = np.zeros((1000, inner_dimension))
user_vectors = np.zeros((10000, inner_dimension))
user_bias = np.zeros(10000)
item_bias = np.zeros(1000)


epochs = 50
learning_rate = 0.00005
regularization_parameter = 0.0
for i in range(epochs):
    print(f"started epoch: {i}")
    if i>0: # and i%10==0:
        learning_rate *= 0.9

    for i in range(mat_train.shape[0]):
      for j in range(mat_train.shape[1]):
        if mat_train[i][j]!=0:
          user = i
          item = j
          rating = mat_train[user][item]
          prediction = average_rating + user_bias[user] + item_bias[item] + np.dot(user_vectors[user], item_vectors[item])
          error = rating - prediction

          item_bias[item] = item_bias[item] + learning_rate * (error - regularization_parameter * item_bias[item])
          user_bias[user] = user_bias[user] + learning_rate * (error - regularization_parameter * user_bias[user])
          user_vectors[user] = user_vectors[user] + learning_rate * (error - regularization_parameter * user_vectors[user])
          item_vectors[item] = item_vectors[item] + learning_rate * (error - regularization_parameter * item_vectors[item])

    #error calculation
    rmse_error = 0.0
    for i in range(mat_validation.shape[0]):
      for j in range(mat_validation.shape[1]):
        if mat_validation[i][j]!=0:
          user = i
          item = j
          rating = mat_validation[user][item]
          prediction = average_rating + user_bias[user] + item_bias[item] + np.dot(user_vectors[user], item_vectors[item])
          error = rating - prediction
          squared_pred_error = error * error
          #reg_error = regularization_parameter * (np.linalg.norm(item_vectors[item])**2 + np.linalg.norm(user_vectors[user])**2 + user_bias[user]**2 + item_bias[item]**2)
          rmse_error += squared_pred_error


    rmse_error /= count_val_items
    rmse_error = np.sqrt(rmse_error)
    print(f"VALIDATION RMSE {rmse_error}")

