import tensorflow as tf
import tensorflow.keras as keras
from sklearn.impute import SimpleImputer
from tensorflow.keras import layers
from tensorflow.keras import activations
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression



path = './data'
print("path declared")

data = pd.read_csv("./data/data_train.csv").to_numpy()

mat_total = np.zeros((10000, 1000))
mat_train = np.zeros((10000, 1000))
mat_train_imputed = np.empty((10000, 1000))
mat_train_imputed.fill(np.nan)

mat_validation = np.zeros((10000, 1000))

np.random.seed(42)
tf.random.set_seed(42)

mask = np.random.rand(len(data[:, 0])) < 0.9

class UserFactors(keras.layers.Layer):
  def __init__(self, n, name=None, **kwargs):
    super().__init__(name=name)
    self.n = n
    self.bias = self.add_weight(shape=((10000, self.n)), initializer="zeros", trainable=True)

  def call(self, inputs):
    biases = tf.gather(self.bias, inputs[:, 0])
    return biases

  def get_config(self):
      config = super().get_config()
      config.update({
          "n": self.n
      })
      return config


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


class TrueValidationScore_modified(keras.callbacks.Callback):
  def __init__(self, mat_train_imputed, mat_train, mat_validation):
    super().__init__()
    self.mat_train_imputed = mat_train_imputed
    self.mat_train = mat_train
    self.mat_validation = mat_validation
    self.ids = np.asarray([np.arange(10000)]).T

  def on_epoch_end(self, epoch, logs=None):
    pred = self.model.predict([self.mat_train_imputed, self.ids, self.mat_train])
    mask = mat_validation > 0
    rmse = np.sqrt(np.mean(np.square(pred[mask] - mat_validation[mask])))
    print(f"epoch {epoch} - val_score: {rmse:.4f}")

for i, row in enumerate(data):
  a, b = row[0].split("_")
  m, n = int(a[1:]) - 1, int(b[1:]) - 1
  mat_total[m, n] = row[1]

  if mask[i]:
    mat_train[m, n] = row[1]
    mat_train_imputed[m, n] = row[1]
  else:
    mat_validation[m, n] = row[1]


###
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
mat_train_imputed = imp_mean.fit_transform(mat_train_imputed)


from scipy.sparse.linalg import svds
U, sigma, Vt = svds(mat_train_imputed, k = 12)
sigma = np.diag(sigma)

mat_train_svd = np.dot(np.dot(U, sigma), Vt)

print("MAT TRAIN COMPUTED")

###





tfk = tf.keras
tfkl = tf.keras.layers
tfd = tfp.distributions
tfpl = tfp.layers





def NLL_loss(y_true, y_pred):
  return -y_pred.log_prob(y_true)

def train_loss(y_true, y_pred):
  return tf.reduce_mean(((y_true - y_pred) ** 2) * tf.clip_by_value(y_true, 0, 1))

def train_loss_modified(y_true, y_pred, y_not_imputed):
  return tf.reduce_mean(((y_true - y_pred) ** 2) * tf.clip_by_value(y_not_imputed, 0, 1))

def RMSE_loss(y_true, y_pred):
  mask = tf.clip_by_value(y_true, 0, 1)
  return tf.sqrt(tf.reduce_sum(((y_true - y_pred) ** 2) * mask) / tf.reduce_sum(mask))

class UserBias(tf.keras.layers.Layer):
  def __init__(self):
    super().__init__()
    self.bias = self.add_weight(shape=((10000,)), initializer="zeros", trainable=True)

  def call(self, inputs):
    biases = tf.gather(self.bias, inputs[1][:, 0])
    return inputs[0] + tf.transpose([biases])


from autoencoders import stefan
print("STEFAN MODEL")
stefan_model = stefan(mat_train, mat_validation)


input1 = keras.layers.Input(shape=(1000, ))
input2 = keras.layers.Input(shape=(1,), dtype=tf.int32)
input3 = keras.layers.Input(shape=(1000, ))

x = layers.Dropout(0.5)(input1)
x = tfkl.Dense(12, "elu")(x)
x = tfkl.Dense(12, "elu")(x)
x = tfkl.Dense(12, "elu")(x)

encoded_size = 12
prior = tfd.Independent(tfd.Normal(loc=tf.zeros(encoded_size), scale=1), reinterpreted_batch_ndims=1)
x = tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(encoded_size), activation=None)(x)
x = tfpl.MultivariateNormalTriL(encoded_size, activity_regularizer=tfpl.KLDivergenceRegularizer(prior, weight=0.0))(x)


#x = tf.keras.layers.Concatenate(axis=1)([x, y])
x = tfkl.Dense(12, "elu")(x)
x = tfkl.Dense(12, "elu")(x)
x = tfkl.Dense(1000, "linear")(x)
x = UserBias()([x, input2])




model = keras.Model(inputs=[input1, input2], outputs=[x], name="VAE")
#model = keras.Model(inputs=[input1, input2, input3], outputs=[x], name="VAE")
model.add_loss(train_loss(input1, x))
#model.add_loss(NLL_loss(input1, x))
#model.add_loss(train_loss_modified(input1, x, input3))


lr = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=10.0,
    decay_steps=800,
    decay_rate=0.96)
ids = np.asarray([np.arange(10000)]).T

model.compile(optimizer=keras.optimizers.Adadelta(learning_rate=lr))

print("VAE")
model.fit([mat_train, ids], mat_train, epochs=400, batch_size=64, #callbacks=[TrueValidationScore(mat_train, mat_validation)], #validation_data=([mat_validation, ids], mat_validation)
         shuffle=True, verbose = True, use_multiprocessing= True, workers=-1)


print("AE")
from autoencoders import construct_autoencoder, calc_rmse, calc_rmse_array

AE = construct_autoencoder()
lrAE = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=10.0,
    decay_steps=800,
    decay_rate=0.96)
ids = np.asarray([np.arange(10000)]).T

AE.compile(optimizer=keras.optimizers.Adadelta(learning_rate=lrAE))

AE.fit([mat_train, ids], mat_train, epochs=400, batch_size=64, #callbacks=[TrueValidationScore(mat_train, mat_validation)],
         shuffle=True, verbose = True, use_multiprocessing= True, workers=-1)







VAEpreds_train = model.predict([mat_train, ids])
AEpreds_train = AE.predict([mat_train, ids])
stefan_pred = stefan_model.predict([mat_train, ids])




VAEpreds_train_rav = VAEpreds_train.ravel()
AEpreds_train_rav = AEpreds_train.ravel()
stefan_pred_rav = stefan_pred.ravel()

mat_svd_rav = mat_train_svd.ravel()

concat_predictions = np.vstack([VAEpreds_train_rav, AEpreds_train_rav, #mat_svd_rav,
                                stefan_pred_rav]).T
mat_train_rav = mat_train.ravel()

regressor = LinearRegression(fit_intercept=True)
X = concat_predictions[mat_train_rav!=0]
Y = mat_train_rav[mat_train_rav!=0]
regressor.fit(X, Y)

regressor_predictions = regressor.predict(concat_predictions)

print("ENSEMBLE ")
#print(calc_rmse_array(regressor_predictions, mat_validation.ravel()))
print(f"WEIGHTS {regressor.coef_} intercept {regressor.intercept_}")


reg_reshaped = regressor_predictions.reshape((10000, 1000))
data_sample = pd.read_csv("./data/sampleSubmission.csv").to_numpy()
target_indices = []
for d in data_sample:
    a, b = d[0].split("_")
    target_indices.append((int(a[1:]) - 1, int(b[1:]) - 1))

#pred = model.predict([mat_total,ids])

with open("submission.csv", "w") as f:
  f.write("Id,Prediction\n")
  for t in target_indices:
    f.write(f"r{t[0]+1}_c{t[1]+1},{reg_reshaped[t[0], t[1]]}\n")

import zipfile
zipfile.ZipFile("ENSEMBLEsubmission.zip", "w").write("submission.csv", compress_type=zipfile.ZIP_DEFLATED)

