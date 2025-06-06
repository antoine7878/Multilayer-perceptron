import numpy as np
import pickle

from .layers import Layer, Manual
from .loss import cross_entropy, cross_entropy_softmax_1
from .metrics import accuracy
from .optimizers import SGD, Optimizer
from .utils import shuffle


class Sequential:
    def __init__(self, layers: list[Layer]) -> None:
        self.layers = layers
        self.depth = len(self.layers)
        self.in_size = layers[0].size
        self.loss = cross_entropy
        self.loss_1 = cross_entropy_softmax_1
        for i in range(1, self.depth):
            self.layers[i].connect(self.layers[i - 1].size)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        validation: tuple[np.ndarray, np.ndarray],
        epochs: int = 16,
        batch_size: int = 16,
        early_stop: tuple = ("", 0, 0),
        optimizer: Optimizer = SGD(0.01),
    ) -> dict:
        self.history = {"loss": [], "accuracy": [],
                        "val_loss": [], "val_accuracy": []}
        batch_count = int(len(x) / batch_size)
        x_test = validation[0]
        y_test = validation[1]
        self.add_history(y, self(x), y_test, self(x_test))
        lst = [f" - {key}: {value[-1]:#.4g}" for key,
               value in self.history.items()]
        print(f"Epoch 0/{epochs} {''.join(lst)}")
        for epoch_i in range(epochs):
            x_suffle, y_suffle = shuffle(x,y)


            for batch_i in range(batch_count):
                x_batch = x_suffle[batch_i *
                                   batch_size: min((batch_i + 1) * batch_size, len(x))]
                y_batch = y_suffle[batch_i *
                                   batch_size: min((batch_i + 1) * batch_size, len(x))]
                self.__update(x_batch, y_batch, optimizer)

            self.add_history(y, self(x), y_test, self(x_test))
            lst = [f" - {key}: {value[-1]:#.4g}" for key,
                   value in self.history.items()]
            print(f"Epoch {epoch_i + 1}/{epochs} {''.join(lst)}")

            if self.early_stop(epoch_i, early_stop):
                return self.history

        return self.history

    def early_stop(self, i: int, early_stop: tuple) -> bool:
        (metric, delta, patience) = early_stop
        if not metric or patience < 1 or i <= patience:
            return False
        if (
            all(
                [
                    acc + delta > self.history[metric][-1] for acc in self.history[metric][-patience:-1]
                ]
            )
        ):
            return True
        return False

    def add_history(self, y: np.ndarray, y_pred: np.ndarray, y_test: np.ndarray, y_test_pred: np.ndarray) -> None:
        self.history["accuracy"].append(accuracy(y, y_pred))
        self.history["loss"].append(self.loss(y, y_pred).mean())
        self.history["val_accuracy"].append(accuracy(y_test, y_test_pred))
        self.history["val_loss"].append(self.loss(y_test, y_test_pred).mean())

    def __update(self, x: np.ndarray, y: np.ndarray, optimizer: Optimizer) -> None:
        self.__backprop(x, y)
        for layer in self.layers:
            if not layer.trainable:
                continue
            optimizer.update(layer, len(x))

    def __backprop(self, x: np.ndarray, y: np.ndarray) -> None:
        act = x
        for layer in self.layers:
            act = layer(act)

        out_l = self.layers[-1]
        delta = self.loss_1(y, out_l.act)
        out_l.grad_b = np.sum(delta, axis=0, keepdims=True)
        out_l.grad_w = np.matmul(self.layers[-2].act.T, delta)

        for l_id in range(2, self.depth):
            layer = self.layers[-l_id]
            sp = np.apply_along_axis(
                func1d=layer.activation_1, axis=1, arr=layer.z)
            delta = np.matmul(delta, self.layers[-l_id + 1].weights.T) * sp
            layer.grad_b = np.sum(delta, axis=0, keepdims=True)
            layer.grad_w = np.matmul(self.layers[-l_id - 1].act.T, delta)

    def save(self, filename):
        model_dump = {"weights": [], "biases": [], "activations": []}
        for layer in self.layers:
            model_dump["weights"].append(layer.weights)
            model_dump["biases"].append(layer.bias)
            model_dump["activations"].append(layer.activation_str)
        with open(filename, "wb") as file:
            pickle.dump(model_dump, file)

    def parameter_count(self) -> int:
        return sum([layer.parameter_count() for layer in self.layers])

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as file:
            model_dump = pickle.load(file)
        layers = []
        for weight, bias, activation in zip(model_dump["weights"], model_dump["biases"], model_dump["activations"]):
            layers.append(Manual(weight, bias, activation))
        return cls(layers)
