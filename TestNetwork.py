import numpy as np


class TestNetwork:
    def __init__(self, sizes, activation_funcs, loss_func, learning_speed=10000000):
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

    def backward(self, a, y, count_epoch, count_iterations_in_epoch, learning_speed, friction):
        self.learning_speed = learning_speed
        au = a
        loss_values = []
        for epoch in range(count_epoch):
            # biny_batch = np.random.au
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
                    weights_speed[j] *= friction
                    weights_speed[j] += delta_w / float(self.learning_speed)
                    w_temp = w
                    # regularization = 2 * w * 0.000003
                    self.weights[j] -= weights_speed[j]  # + regularization
                    # self.weights[j] -= delta_w / float(self.learning_speed)
                    y_pred = y_pred.dot(w_temp.T)
        return loss_values

