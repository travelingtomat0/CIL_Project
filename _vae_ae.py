import tensorflow as tf

tf.random.set_seed(42)
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import util
import tensorflow_probability as tfp
import common as common

tf.keras.backend.set_floatx('float32')


def run(embedding_size=12):
    print("--- Variational AE ---")

    def get_model():
      ratings = keras.Input((1000,), dtype=tf.float32)
      user_id = keras.Input((1,), dtype=tf.int32)

      x = ratings
      x = layers.Dropout(0.5)(x)
      x = layers.Dense(embedding_size, "elu")(x)
      x = layers.Dense(embedding_size, "elu")(x)
      x = layers.Dense(embedding_size, "elu")(x)

      tfk = tf.keras
      tfkl = tf.keras.layers
      tfd = tfp.distributions
      tfpl = tfp.layers

      prior = tfd.Independent(tfd.Normal(loc=tf.zeros(embedding_size), scale=1), reinterpreted_batch_ndims=1)
      x = tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(embedding_size), activation=None)(x)
      x = tfpl.MultivariateNormalTriL(embedding_size,
                                      activity_regularizer=tfpl.KLDivergenceRegularizer(prior, weight=0.0))(
          x)

      x = layers.Dense(embedding_size, "elu")(x)
      x = layers.Dense(embedding_size, "elu")(x)
      x = layers.Dense(1000, "linear")(x)
      z = common.UserFactors(1, "zeros")(user_id)
      x = x + z

      model = keras.Model(inputs=[ratings, user_id], outputs=[x])
      model.add_loss(common.train_loss(ratings, x))

      lr = keras.optimizers.schedules.ExponentialDecay(
          initial_learning_rate=10.0,
          decay_steps=800,
          decay_rate=0.96)

      model.compile(optimizer=keras.optimizers.Adadelta(learning_rate=lr))

      return model

    ids = np.asarray([np.arange(10000)]).T

    ##############################
    # Cross-validation
    print("Cross validation...")

    with open("output/vae_ae_cv.txt", "w") as f:
        rmses = []
        for i in range(10):
            print(f"{i + 1} / 10:")

            mat_train = util.indices_to_matrix(util.load_indices(f"temp/idx_cv_train_{i}.npy"))
            mat_val = util.indices_to_matrix(util.load_indices(f"temp/idx_cv_val_{i}.npy"))

            model = get_model()
            model.fit([mat_train, ids], mat_train, epochs=400, batch_size=64, shuffle=True,
                      callbacks=[common.TrueValidationScore(mat_train, mat_val)], verbose=False)
            val = model.predict([mat_train, ids])

            if i == 0:
                util.export_matrix_to_csv(val, "output/vae_ae_fold0.csv", export_train=True)

            rmse = util.rmse(mat_val[mat_val > 0], val[mat_val > 0])
            print(f"RMSE: {rmse}")
            f.write(f"{rmse}\n")
            rmses.append(rmse)
        f.write(f"mean: {np.mean(rmses)}, std: {np.std(rmses)}")

    ##############################
    # Submission
    print("Submission...")

    mat_train = util.indices_to_matrix(util.load_indices(f"temp/idx_train.npy"))

    model = get_model()

    model.fit([mat_train, ids], mat_train, epochs=400, batch_size=64, shuffle=True, verbose=True)
    predictions = model.predict([mat_train, ids])

    util.export_matrix_to_csv(predictions, "output/vae_ae_train.csv", export_train=True)
    util.export_matrix_to_csv(predictions, "output/vae_ae_sub.csv", export_submission=True, clip=True)

    print("Done.")


if __name__ == '__main__':
    run()
