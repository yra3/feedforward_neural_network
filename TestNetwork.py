import numpy as np
from LossFuncs import L2
# def ReLU(x):
#     return np.maximum(x, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(z):# Производная сигмоидальной функции
    return sigmoid(z)*(1-sigmoid(z))

#
# def ReLU(z):
#     return np.maximum(0,z)


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
    def __init__(self, sizes, activation_funcs, loss_func, learning_speed):
        self.sizes = sizes
        self.count_input = sizes[0]
        self.count_output = sizes[-1]
        self.functions = activation_funcs
        self.num_layers = len(sizes)  # число слоев
        self.learning_speed = learning_speed
        self.loss_func = loss_func
        self.weights = [np.random.randn(y + 1, x) for x, y in zip(sizes[1:], sizes[:-1])]

    def feedforward(self, a):
        for w, func in zip(self.weights, self.functions):
            b = np.ones((a.shape[0], 1))
            a = np.concatenate((a, b), axis=1)
            a = func.func(a.dot(w))
        return a

    def backward(self, a, y, count_epoch, count_iterations_in_epoch):
        au = a
        loss_values = []
        for epoch in range(count_epoch):
            weights_speed = [np.zeros_like(w) for w in self.weights]
            for i in range(count_iterations_in_epoch):
                a_without_b_save = []
                a_with_b_save = []
                a_with_b_multiplied_w_save = []
                a_after_funk_save = []
                a = au
                for w, func in zip(self.weights, self.functions):
                    a_without_b_save.append(a)
                    b = np.ones((a.shape[0], 1))
                    a = np.concatenate((a, b), axis=1)
                    a_with_b_save.append(a)
                    a = a.dot(w)
                    a_with_b_multiplied_w_save.append(a)
                    a = func.func(a)
                    a_after_funk_save.append(a)
                y_pred = a


                total_loss = self.loss_func.loss(y, y_pred)
                loss_values.append(total_loss)

                loss = self.loss_func.grad(y, y_pred)


                y_pred = loss
                is_first = True
                for w, func, save, a_after_funk, a_with_b_w, j in zip(self.weights[::-1], self.functions[::-1]
                        , a_with_b_save[::-1], a_after_funk_save[::-1], a_with_b_multiplied_w_save[::-1],
                                                                            range(self.num_layers - 2, -1, -1)):
                    z = func.prime_func(a_with_b_w)
                    if is_first:
                        is_first = False
                    else:
                        y_pred = np.delete(y_pred, (y_pred.shape[1] - 1), axis=1)
                    y_pred = y_pred * z
                    delta_w = (save.T.dot(y_pred))
                    weights_speed[j] *= 0.4
                    weights_speed[j] += delta_w / float(self.learning_speed)
                    w_temp = w
                    regularization = 2 * w * 0.000003
                    self.weights[j] -= weights_speed[j]
                    # self.weights[j] -= delta_w / float(self.learning_speed)
                    y_pred = y_pred.dot(w_temp.T)
        return loss_values


