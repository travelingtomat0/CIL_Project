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

def train_loss(y_true, y_pred):
  mask = tf.clip_by_value(y_true, 0.0, 1.0)
  return tf.reduce_mean(((y_true - y_pred) ** 2) * mask)


ratings = keras.Input((1000,), dtype=tf.float32)
user_id = keras.Input((1,), dtype=tf.int32)

w = 12

z = UserFactors(w)(user_id)

x = ratings

x = layers.Dropout(0.5)(x)

x = layers.Dense(w, "elu")(x)
x = layers.Dense(w, "elu")(x) + x
x = layers.Dense(w, "elu")(x) + x

x = x + z

x = layers.Dense(w, "elu")(x) + x
x = layers.Dense(w, "elu")(x) + x
x = layers.Dense(w, "elu")(x)

mu = layers.Dense(1000, "linear")(x)

model = keras.Model(inputs=[ratings, user_id], outputs=[mu])
model.add_loss(train_loss(ratings, mu))

lr = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=10.0,
    decay_steps=800,
    decay_rate=0.96)

model.compile(optimizer=keras.optimizers.Adadelta(learning_rate=lr))

ids = np.asarray([np.arange(10000)]).T

model.fit([mat_train, ids], mat_train, epochs=300, batch_size=64, shuffle=True, callbacks=[TrueValidationScore(mat_train, mat_validation)], verbose=False)

