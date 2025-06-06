import matplotlib.pyplot as plt
import numpy as np
import os
import sklearn

from .layers import Normalization

from .utils import onehot, train_test_split


def pca(x: np.ndarray, y: np.ndarray):
    fig = plt.figure()
    x = Normalization(x)(x)
    pca = sklearn.decomposition.PCA(n_components=3)
    x_pca, y_pca, z_pca = zip(*pca.fit_transform(x))
    ax = fig.add_subplot(projection="3d")
    scatter = ax.scatter(x_pca, y_pca, z_pca, c=(y == "M").astype(int), marker="o")
    legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
    ax.add_artist(legend1)
    plt.tight_layout()
    plt.show()


def split(args):
    data = np.loadtxt(args.file, delimiter=",", dtype=str)
    x, y = data[:, 2:].astype(float), data[:, 1]

    # pca(x, y)

    y = onehot(y)
    x_train, y_train, x_val, y_val = train_test_split(x, y, 0.2)

    os.makedirs("data", exist_ok=True)
    np.save("data/x_train.npy", x_train)
    np.save("data/y_train.npy", y_train)
    np.save("data/x_eval.npy", x_val)
    np.save("data/y_eval.npy", y_val)

    print("Data file loaded and saved to {x_train, y_train, x_val, y_val}.npy")
