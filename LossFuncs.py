from LossFunc import LossFunc
import numpy as np


class L2(LossFunc):
    def loss(self, y, y_pred):
        return np.square(y - y_pred).sum()

    def grad(self, y, y_pred):
        return 2 * (y_pred - y)


class L1(LossFunc):
    def loss(self, y, y_pred):
        return np.abs(y-y_pred)

    def grad(self, y, y_pred):
        grad = np.ones_like(y_pred)
        grad[(y-y_pred)<0] = -1 #TODO test
        return grad


# class SoftMax(LossFunc):
#     def grad(self, y, y_pred):
#         e_x = np.exp(y_pred - np.max(y_pred, axis=0))
#         return e_x / e_x.sum(axis=0)
#
#     def softmax_loss_vectorized(self, y, y_pred):
#         reg = 0.03
#         loss = 0.0
#         num_classes = W.shape[1]
#         num_train = X.shape[0]
#         scores = X.dot(W)  # 1
#
#         correct_class_scores = scores[range(num_train), y].reshape((num_train, 1))
#         sum_j = np.sum(np.exp(scores), axis=1).reshape((num_train, 1))
#         loss = np.sum(-1 * correct_class_scores + np.log(sum_j)) / num_train + reg * np.sum(W * W)
#
#         correct_matrix = np.zeros(scores.shape)
#         correct_matrix[range(num_train), y] = 1
#
#         dW = X.T.dot(np.exp(scores) / sum_j) - X.T.dot(correct_matrix)
#         dW = dW / num_train + W * reg
#
#         return loss, dW


class Svm(LossFunc):
    def loss(self, y, y_pred):
        delta = 1.0
        count_y = len(y)
        answers = y_pred[range(count_y), y].reshape(count_y, 1)
        margins = np.maximum(0, y_pred - answers + delta)
        margins[range(count_y), y] = 0
        return np.sum(margins)
    def grad(self, y, y_pred):
        delta = 1.0
        count_y = len(y)
        answers = y_pred[range(count_y), y].reshape(count_y, 1)
        grad = np.ones_like(y_pred)
        grad[y_pred - answers + delta < 0] = 0
        grad[range(count_y), y] = 0
        grad[range(count_y), y] = -grad.sum(axis=1)
        return grad

    # def svm_loss_and_grad_vectorized(W, X, y, reg):
    #     loss = 0.0
    #     delta = 1.0
    #     dW = np.zeros(W.shape)
    #     num_train = X.shape[0]
    #     scores = X.dot(W)
    #
    #     correct_class_scores = scores[range(num_train), y].reshape((num_train, 1))
    #     margins = np.maximum(0.0, scores - correct_class_scores + delta)
    #     margins[range(num_train), y] = 0.0
    #     loss = (np.sum(margins) / num_train) + reg * np.sum(W * W)
    #     margins[margins > 0] = 1.0
    #     margins[margins < 0] = 0.0
    #     margins[range(num_train), y] = -1.0 * np.sum(margins, axis=1)
    #
    #     dW = (X.T.dot(margins) / num_train) + W * reg
    #
    #     return loss, dW

