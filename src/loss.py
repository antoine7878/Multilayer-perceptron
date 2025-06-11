import numpy as np


def cross_entropy(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    loss = -(y * np.log(y_pred))
    return np.apply_along_axis(sum, 1, loss)


def cross_entropy_softmax_1(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return y_pred - y

