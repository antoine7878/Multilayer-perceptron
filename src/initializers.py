import numpy as np
import math


def heNormal(size: tuple[int, int], in_size: float) -> np.ndarray:
    return np.random.normal(scale=math.sqrt(2 / in_size), size=size)


def normal(size: tuple[int, int], in_size: float) -> np.ndarray:
    return np.random.normal(size=size)
