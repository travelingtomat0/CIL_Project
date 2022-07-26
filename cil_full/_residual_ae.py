import tensorflow as tf

tf.random.set_seed(42)
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import util
import common as common

tf.keras.backend.set_floatx('float64')


def run(embedding_size=12):
    print("--- Residual AE ---")

    ratings = keras.Input((1000,), dtype=tf.float64)
    user_id = keras.Input((1,), dtype=tf.int32)

    x = ratings
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(embedding_size, "elu")(x)
    x = layers.Dense(embedding_size, "elu")(x) + x
    x = layers.Dense(embedding_size, "elu")(x) + x
    x = layers.Dense(embedding_size, "elu")(x) + x
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
    model.save("temp/residual_ae.hdf5")

    ids = np.asarray([np.arange(10000)]).T

    ##############################
    # Cross-validation
    print("Cross validation...")

    with open("output/residual_ae_cv.txt", "w") as f:
        rmses = []
        for i in range(10):
            print(f"{i + 1} / 10:")

            mat_train = util.indices_to_matrix(util.load_indices(f"temp/idx_cv_train_{i}.npy"))
            mat_val = util.indices_to_matrix(util.load_indices(f"temp/idx_cv_val_{i}.npy"))

            model = keras.models.load_model("temp/residual_ae.hdf5",
                                            custom_objects={'UserFactors': common.UserFactors,
                                                            'train_loss': common.train_loss})
            model.fit([mat_train, ids], mat_train, epochs=400, batch_size=64, shuffle=True,
                      callbacks=[common.TrueValidationScore(mat_train, mat_val)], verbose=False)
            val = model.predict([mat_train, ids])
            rmse = util.rmse(mat_val[mat_val > 0], val[mat_val > 0])
            print(f"RMSE: {rmse}")
            f.write(f"{rmse}\n")
            rmses.append(rmse)
        f.write(f"mean: {np.mean(rmses)}, std: {np.std(rmses)}")

    ##############################
    # Submission
    print("Submission...")

    mat_train = util.indices_to_matrix(util.load_indices(f"temp/idx_train.npy"))

    model = keras.models.load_model("temp/residual_ae.hdf5",
                                    custom_objects={'UserFactors': common.UserFactors, 'train_loss': common.train_loss})
    model.fit([mat_train, ids], mat_train, epochs=400, batch_size=64, shuffle=True, verbose=True)
    predictions = model.predict([mat_train, ids])

    util.export_matrix_to_csv(predictions, "output/residual_ae_train.csv", export_train=True)
    util.export_matrix_to_csv(predictions, "output/residual_ae_sub.csv", export_submission=True, clip=True)

    print("Done.")


if __name__ == '__main__':
    run()
