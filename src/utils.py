import numpy as np

onehot_encoding = {"M": [1, 0], "B": [0, 1]}


def onehot(x: np.ndarray) -> np.ndarray:
    return np.array([onehot_encoding[row] for row in x])


def dehot(x: np.ndarray) -> np.ndarray:
    return np.array([int(row[0] == 1) for row in x])


def train_test_split(
    x: np.ndarray, y: np.ndarray, test_size: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y = shuffle(x, y)
    limit = int(len(x) * test_size)
    return x[limit:], y[limit:], x[:limit], y[:limit]


def shuffle(x, y):
    c = np.hstack([x, y])
    c = np.random.permutation(c)
    return c[:, : x.shape[1]], c[:, x.shape[1] :]
