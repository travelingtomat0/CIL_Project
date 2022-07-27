import tensorflow as tf

tf.random.set_seed(42)
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import util
import common as common

tf.keras.backend.set_floatx('float64')

# Imports for cluster label generation
from sklearn.decomposition import PCA
import umap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.impute import SimpleImputer


class UserClusterBias(keras.layers.Layer):
    def __init__(self, cluster_map, name=None, **kwargs):
        super().__init__(name=name)
        self.cluster_map = cluster_map
        self.bias = self.add_weight(shape=((10000, len(np.unique(cluster_map)))), initializer="zeros", trainable=True)

    def call(self, inputs):
        cluster_map = tf.constant(self.cluster_map)
        biases = tf.transpose(tf.gather(self.bias, inputs[:, 0]))
        movie_biases = tf.transpose(tf.gather(biases, cluster_map))
        return movie_biases

    def get_config(self):
        config = super().get_config()
        config.update({
            "cluster_map": self.cluster_map
        })
        return config


def cluster_prep(mean_impute=False):
    """
    Computes item-labels for the cluster_AE model.
    """
    print("Prepare Labels for ClusterAE...")
    mat = util.indices_to_matrix(util.load_indices("input/data_train.csv")).T
    pca = PCA(n_components=25)

    mat = np.nan_to_num(mat)
    if mean_impute:
        imp_mean = SimpleImputer(missing_values=0, strategy='mean')
        mat = imp_mean.fit_transform(mat)
    reduced_mat = pca.fit_transform(mat)
    embedding = umap.UMAP(metric='correlation').fit_transform(reduced_mat)

    clustering = AgglomerativeClustering(n_clusters=2).fit_predict(embedding)

    # Save clustering plot for validation purposes.
    plt.figure(figsize=(10, 10))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=clustering, cmap='Spectral')
    plt.savefig("temp/item_clusters.jpg")
    np.save("input/cluster_map.npy", np.asarray(clustering))


def run(embedding_size=12):
    print("--- Cluster AE ---")

    # Generate and fetch cluster labels
    cluster_prep(mean_impute=False)
    clusters = np.load("input/cluster_map.npy").astype(int)

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
    z = UserClusterBias(clusters)(user_id)
    x = x + z

    model = keras.Model(inputs=[ratings, user_id], outputs=[x])
    model.add_loss(common.train_loss(ratings, x))

    lr = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=10.0,
        decay_steps=800,
        decay_rate=0.96)

    model.compile(optimizer=keras.optimizers.Adadelta(learning_rate=lr))
    model.save("temp/cluster_ae.hdf5")

    ids = np.asarray([np.arange(10000)]).T

    ##############################
    # Cross-validation
    print("Cross validation...")

    with open("output/cluster_ae_cv.txt", "w") as f:
        rmses = []
        for i in range(10):
            print(f"{i + 1} / 10:")

            mat_train = util.indices_to_matrix(util.load_indices(f"temp/idx_cv_train_{i}.npy"))
            mat_val = util.indices_to_matrix(util.load_indices(f"temp/idx_cv_val_{i}.npy"))

            model = keras.models.load_model("temp/cluster_ae.hdf5",
                                            custom_objects={'UserClusterBias': UserClusterBias,
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

    model = keras.models.load_model("temp/cluster_ae.hdf5",
                                    custom_objects={'UserClusterBias': UserClusterBias,
                                                    'train_loss': common.train_loss})
    model.fit([mat_train, ids], mat_train, epochs=400, batch_size=64, shuffle=True, verbose=True)
    predictions = model.predict([mat_train, ids])

    util.export_matrix_to_csv(predictions, "output/cluster_ae_train.csv", export_train=True)
    util.export_matrix_to_csv(predictions, "output/cluster_ae_sub.csv", export_submission=True, clip=True)

    print("Done.")


if __name__ == '__main__':
    run()
