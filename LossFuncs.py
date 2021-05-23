from LossFunc import LossFunc
import numpy as np


class L2(LossFunc):
    def loss(self, y, y_pred):
        return np.square(y - y_pred).sum()/y.size

    def grad(self, y, y_pred):
        return 2 * (y_pred - y)


class L1(LossFunc):
    def loss(self, y, y_pred):
        return np.abs(y-y_pred).sum()/y.size

    def grad(self, y, y_pred):
        grad = np.ones_like(y_pred)
        grad[(y-y_pred)<0] = -1 #TODO test
        return -grad


class SoftMax(LossFunc):
    def loss(self, y, y_pred):
        loss = 0.0
        count_y = len(y)
        answers = y_pred[range(count_y), y].reshape(count_y, 1)
        sum_j = np.sum(np.exp(y_pred), axis=1).reshape((count_y, 1))
        return np.sum(-1 * answers + np.log(sum_j)) / count_y

    def grad(self, y, y_pred):
        delta = 1.0
        count_y = len(y)
        answers = y_pred[range(count_y), y].reshape(count_y, 1)
        grad = np.ones_like(y_pred)
        grad[y_pred - answers + delta < 0] = 0
        grad[range(count_y), y] = 0
        grad[range(count_y), y] = -grad.sum(axis=1)
        return grad


class Svm(LossFunc):
    def loss(self, y, y_pred):
        delta = 1.0
        count_y = len(y)
        answers = y_pred[range(count_y), y].reshape(count_y, 1)
        margins = np.maximum(0, y_pred - answers + delta)
        margins[range(count_y), y] = 0
        return np.sum(margins)/count_y

    def grad(self, y, y_pred):
        delta = 1.0
        count_y = len(y)
        answers = y_pred[range(count_y), y].reshape(count_y, 1)
        grad = np.ones_like(y_pred)
        grad[y_pred - answers + delta < 0] = 0
        grad[range(count_y), y] = 0
        grad[range(count_y), y] = -grad.sum(axis=1)
        return grad


