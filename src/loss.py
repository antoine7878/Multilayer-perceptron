import numpy as np


def mse(y, y_pred):
    return np.sum((y_pred - y) ** 2) / len(y)


def mse_1(y, y_pred):
    return np.sum(y_pred - y)


def cross_entropy(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return loss


def cross_entropy_softmax_1(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return y_pred - y


LOSSES_DERIV = {mse: mse_1}
