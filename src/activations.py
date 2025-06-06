import numpy as np


def linear(x: np.ndarray) -> np.ndarray:
    return x


def linear_prime(x: np.ndarray) -> np.ndarray:
    return np.ones(x.shape)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x: np.ndarray) -> np.ndarray:
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(x) / np.sum(np.exp(x))


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def tanh_prime(x: np.ndarray) -> np.ndarray:
    return 1 - tanh(x) ** 2


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def relu_prime(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(int)


ACTIVATIONS_DERIV = {
    sigmoid: sigmoid_prime,
    linear: linear_prime,
    relu: relu_prime,
    tanh: tanh_prime,
    softmax: None,
}
