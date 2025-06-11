import numpy as np
import matplotlib.pyplot as plt

from .initializers import heNormal
from .layers import Dense, Input, Normalization
from .models import Sequential
from .optimizers import RMSProp
from .utils import train_test_split
from .activations import relu, softmax


def train(args):
    x = np.load("data/x_train.npy")
    y = np.load("data/y_train.npy")
    x_train, y_train, x_test, y_test = train_test_split(x, y, 0.3)

    model = Sequential(
        [
            Input(x_train.shape[1]),
            Normalization(x_train),
            Dense(64, activation=relu, initializer=heNormal),
            Dense(32, activation=relu, initializer=heNormal),
            Dense(16, activation=relu, initializer=heNormal),
            Dense(y_train.shape[1], activation=softmax),
        ]
    )

    history = model.fit(
        x_train,
        y_train,
        validation=(x_test, y_test),
        optimizer=RMSProp(learning_rate=0.0001),
        epochs=1000,
        batch_size=4,
        early_stop=("val_accuracy", 0.001, 32),
    )

    model.save(args.file)
    plot(history)


def plot(history):
    plt.subplot(211)
    plt.title("loss")
    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    plt.legend(["loss", "val_loss"])
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.subplot(212)
    plt.title("accuracy")
    plt.plot(history["accuracy"])
    plt.plot(history["val_accuracy"])
    plt.legend(["accuracy", "val_accuracy"])
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.show()
