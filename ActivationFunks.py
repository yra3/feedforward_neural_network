from Funk import Funk
import numpy as np


class Sigmoid(Funk):
    def func(self, x):
        return 1 / (1 + np.exp(-x))

    def prime_func(self, x):
        return self.func(x) * (1 - self.func(x))


class ReLU(Funk):
    def func(self, x):
        return np.maximum(0,x)

    def prime_func(self, x):
        x = np.where(x <= 0, x, 1)
        x = np.where(x >= 0, x, 0)
        return x


class NoneFunc(Funk):
    def func(self, x):
        return x

    def prime_func(self, x):
        return np.ones(x.shape)
