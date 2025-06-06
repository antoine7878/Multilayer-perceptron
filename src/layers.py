from typing import Callable
from abc import ABC, abstractmethod
import numpy as np

from .activations import ACTIVATIONS_DERIV, linear, sigmoid
from .initializers import normal


class Layer(ABC):
    def __init__(self, size: int, in_size: int, activation: Callable[..., np.ndarray]) -> None:
        assert activation in ACTIVATIONS_DERIV, "unknown activation function"
        self.size = size
        self.in_size = in_size
        self.activation = activation
        self.activation_1 = ACTIVATIONS_DERIV[activation]
        self.activation_str = activation

        self.trainable = False
        self.weights = np.array([])
        self.bias = np.array([])
        self.grad_b = np.array([])
        self.grad_w = np.array([])
        self.mom_w = np.array([])
        self.mom_b = np.array([])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.z = np.matmul(x, self.weights) + self.bias
        self.act = np.apply_along_axis(
            func1d=self.activation, axis=1, arr=self.z)
        return self.act

    def parameter_count(self) -> int:
        return self.bias.shape[0] * self.bias.shape[1] + self.weights.shape[0] * self.weights.shape[1]

    @abstractmethod
    def connect(self, in_size: int) -> None:
        pass


class Dense(Layer):
    def __init__(
        self,
        size: int,
        activation: Callable[..., np.ndarray] = sigmoid,
        initializer: Callable[[tuple[int, int], float], np.ndarray] = normal,
    ) -> None:
        super().__init__(size, 0, activation)
        self.initializer = initializer
        self.trainable = True

    def connect(self, in_size: int) -> None:
        self.weights = self.initializer((in_size, self.size), in_size)
        self.bias = self.initializer((1, self.size), in_size)
        self.mom_w = np.zeros_like(self.weights)
        self.mom_b = np.zeros_like(self.bias)
        self.in_size = in_size


class Input(Layer):
    def __init__(self, size: int) -> None:
        super().__init__(size, size, linear)
        self.weights = np.eye(size)
        self.bias = np.zeros((1, size))

    def connect(self, in_size: int) -> None:
        self.in_size = in_size


class Normalization(Layer):
    def __init__(self, x_train: np.ndarray) -> None:
        size = x_train.shape[1]
        super().__init__(size, size, linear)
        self.weights = np.eye(size)
        self.bias = np.zeros((1, size))
        self.fit(x_train)

    def fit(self, x: np.ndarray):
        means = []
        stds = []
        for i in range(x.shape[1]):
            means.append(np.mean(x[:, i]))
            stds.append(np.std(x[:, i]))
        means = np.array(means)
        stds = np.array(stds)
        self.bias -= means / stds
        self.weights /= stds

    def connect(self, in_size: int) -> None:
        self.in_size = in_size


class Manual(Layer):
    def __init__(self, weights: np.ndarray, bias: np.ndarray, activation: Callable[..., np.ndarray]) -> None:
        super().__init__(weights.shape[0], weights.shape[1], activation)
        self.weights = weights
        self.bias = bias

    def connect(self, in_size: int) -> None:
        self.in_size = in_size
