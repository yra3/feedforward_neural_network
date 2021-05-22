import numpy as np

def ReLU(x):
    return np.maximum(x, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(z):# Производная сигмоидальной функции
    return sigmoid(z)*(1-sigmoid(z))


def ReLU(z):
    return np.maximum(0,z)


def ReLU_prime(z):
    # z[np.where(z < 0)] = 0

    z = np.where(z <= 0, z, 1)
    z = np.where(z >= 0, z, 0)
    return z


def sigpriozv(x):
    return np.exp(-x)/np.square(1 + np.exp(x))


def nonef_prime(x):
    return np.ones(x.shape)


def x_funk(x):
    return x


class TestNetwork:
    def __init__(self, sizes, activation_func):
        self.sizes = sizes
        self.count_input = sizes[0]
        self.count_output = sizes[-1]
        self.functions = activation_func
        self.num_layers = len(sizes)  # число слоев
        self.learning_speed = 1000
        self.loss_func = None
        self.weights = [np.random.randn(y + 1, x) for x, y in zip(sizes[1:], sizes[:-1])]

    def feedforward(self, a):
        for w, func in zip(self.weights, self.functions):
            b = np.ones((a.shape[0], 1))
            a = np.concatenate((a, b), axis=1)
            a = func(a.dot(w))
        return a

    def backward(self, a, y):
        au = a
        for i in range(self.learning_speed):
            a_without_b_save = []
            a_with_b_save = []
            a_with_b_multiplied_w_save = []
            a_after_funk_save = []
            # grads_w = [np.zeros(w.shape) for w in self.weights]
            a = au
            for w, func in zip(self.weights, self.functions):

                a_without_b_save.append(a)
                b = np.ones((a.shape[0], 1))
                a = np.concatenate((a, b), axis=1)
                a_with_b_save.append(a)
                a = a.dot(w)
                a_with_b_multiplied_w_save.append(a)
                a = func(a)
                a_after_funk_save.append(a)

            y_pred = a
            loss = (y_pred - y)
            y_pred = loss
            is_first = True
            # for i in range(1, len(self.sizes)):
            #     f_p = funk_prime(a_with_b_multiplied_w_save[-i])

            for w, funk_prime, save, a_after_funk, a_with_b_w, j in zip(self.weights[::-1], [nonef_prime][::-1]
                    , a_with_b_save[::-1], a_after_funk_save[::-1], a_with_b_multiplied_w_save[::-1], range(self.num_layers-1, 0, -1)):
                z = funk_prime(a_with_b_w)
                y_pred = y_pred * z
                if is_first:
                    is_first = False
                else:
                    y_pred = np.delete(y_pred, (y_pred.shape[1]-1), axis=1)
                delta_w = (save.T.dot(y_pred))
                self.weights[-j] -= delta_w / float(self.learning_speed)
                y_pred = y_pred.dot(w.T)


