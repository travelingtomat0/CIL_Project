import numpy as np
import pandas as pd
from  sklearn.impute import SimpleImputer

path = './data'
print("path declared")

data = pd.read_csv("./data/data_train.csv").to_numpy()

mat_total = np.zeros((10000, 1000))
mat_train = np.empty((10000, 1000))
mat_validation = np.zeros((10000, 1000))

mat_train.fill(np.nan)

np.random.seed(42)

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

#user_ratings_mean = np.mean(mat_train, axis = 1)
#R_demeaned = mat_train - user_ratings_mean.reshape(-1, 1)
#print(R_demeaned.shape)

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
mat_train = imp_mean.fit_transform(mat_train)


from scipy.sparse.linalg import svds
U, sigma, Vt = svds(mat_train, k = 12)
sigma = np.diag(sigma)

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) #+ user_ratings_mean.reshape(-1, 1)

a=5


rmse_error = 0.0
for i in range(mat_validation.shape[0]):
  for j in range(mat_validation.shape[1]):
    if mat_validation[i][j]!=0:
        user = i
        item = j
        rating = mat_validation[user][item]
        prediction = all_user_predicted_ratings[user][item]
        error = rating - prediction
        squared_pred_error = error * error
        #reg_error = regularization_parameter * (np.linalg.norm(item_vectors[item])**2 + np.linalg.norm(user_vectors[user])**2 + user_bias[user]**2 + item_bias[item]**2)
        rmse_error += squared_pred_error


rmse_error /= count_val_items
rmse_error = np.sqrt(rmse_error)
print(f"VALIDATION RMSE {rmse_error}")
print("STARTING AUTOENCODER")


import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import tensorflow_probability as tfp


tfk = tf.keras
tfkl = tf.keras.layers
tfd = tfp.distributions
tfpl = tfp.layers

def train_loss(y_true, y_pred):
  return tf.reduce_mean(((y_true - y_pred) ** 2) * tf.clip_by_value(y_true, 0, 1))

class UserBias(keras.layers.Layer):
  def __init__(self):
    super().__init__()
    self.bias = self.add_weight(shape=((10000,)), initializer="zeros", trainable=True)

class TrueValidationScore(keras.callbacks.Callback):
  def __init__(self, mat_total, mat_validation):
    super().__init__()
    self.mat_total = mat_total
    self.mat_validation = mat_validation
    self.ids = np.asarray([np.arange(10000)]).T

  def on_epoch_end(self, epoch, logs=None):
    pred = self.model.predict([self.mat_total, self.ids])
    mask = mat_validation > 0
    rmse = np.sqrt(np.mean(np.square(pred[mask] - mat_validation[mask])))
    print(f"epoch {epoch} - val_score: {rmse:.4f}")


input1 = keras.layers.Input(shape=(1000, ))
input2 = keras.layers.Input(shape=(1,), dtype=tf.int32)
x = layers.Dropout(0.5)(input1)
x = tfkl.Dense(12, "elu")(x)
x = tfkl.Dense(12, "elu")(x)
x = tfkl.Dense(12, "elu")(x)
x = tfkl.Dense(12, "elu")(x)
x = tfkl.Dense(12, "elu")(x)
x = tfkl.Dense(1000, "linear")(x)
x = UserBias()([x, input2])


model = keras.Model(inputs=[input1, input2], outputs=[x], name="AE")
model.add_loss(train_loss(input1, x))


lr = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=10.0,
    decay_steps=800,
    decay_rate=0.96)
ids = np.asarray([np.arange(10000)]).T

model.compile(optimizer=keras.optimizers.Adadelta(learning_rate=lr))
model.fit([all_user_predicted_ratings, ids], all_user_predicted_ratings, epochs=300, batch_size=500, callbacks=[TrueValidationScore(all_user_predicted_ratings, mat_validation)], #validation_data=([mat_validation, ids], mat_validation)
         shuffle=True, verbose = False, use_multiprocessing= True, workers=-1)