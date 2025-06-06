from abc import ABC, abstractmethod

import numpy as np

from .layers import Layer


class Optimizer(ABC):
    def __init__(self, learning_rate: float = 0.001) -> None:
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, layer: Layer, batch_size: int) -> None:
        pass


class SGD(Optimizer):
    def update(self, layer: Layer, batch_size: int) -> None:
        rate = self.learning_rate / batch_size
        layer.weights -= rate * layer.grad_w
        layer.bias -= rate * layer.grad_b


class RMSProp(Optimizer):
    eps = 0.000000000000001

    def __init__(self, learning_rate: float = 0.001, decay_rate: float = 0.90) -> None:
        super().__init__(learning_rate)
        self.decay_rate = decay_rate

    def update(self, layer: Layer, batch_size: int) -> None:
        batch_size = batch_size
        layer.mom_w = self.decay_rate * layer.mom_w + (1 - self.decay_rate) * layer.grad_w**2
        layer.mom_b = self.decay_rate * layer.mom_b + (1 - self.decay_rate) * layer.grad_b**2
        rate_w = self.learning_rate / np.sqrt(layer.mom_w + self.eps)
        rate_b = self.learning_rate / np.sqrt(layer.mom_b + self.eps)
        layer.weights -= rate_w * layer.grad_w
        layer.bias -= rate_b * layer.grad_b
