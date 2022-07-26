import tensorflow as tf

tf.random.set_seed(42)
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import numpy as np
import util


class TrueValidationScore(keras.callbacks.Callback):
    def __init__(self, mat_total, mat_validation):
        super().__init__()
        self.mat_total = mat_total
        self.mat_validation = mat_validation
        self.ids = np.asarray([np.arange(10000)]).T

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            pred = self.model.predict([self.mat_total, self.ids])
            mask = self.mat_validation > 0
            rmse = util.rmse(self.mat_validation[mask], pred[mask])
            print(f"epoch {epoch + 1} - val_score: {rmse:.4f}")


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
    return tf.reduce_mean((((y_true - y_pred) ** 2) * mask))
