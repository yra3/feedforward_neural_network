from Funk import Funk
import numpy as np


class Sigmoid(Funk):
    def funk(self, x):
        return 1 / (1 + np.exp(-x))

    def prime_funk(self, x):
        return self.funk(x) * (1 - self.funk(x))


class ReLU(Funk):
    def funk(self, x):
        return np.maximum(0,x)

    def prime_funk(self, x):
        x = np.where(x <= 0, x, 1)
        x = np.where(x >= 0, x, 0)
        return x


class NoneFunc(Funk):
    def funk(self, x):
        return x

    def prime_funk(self, x):
        return np.ones(x.shape)
