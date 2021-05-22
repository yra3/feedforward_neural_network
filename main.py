import numpy as np


def findedfunk(x):
    return np.power(x, 2)


# def generateregr(count, func, is_linear=False):
#     x = np.arange(0, count, 1).reshape(count, 1) if is_linear else np.random.randn(count, 1)
#     y = func(x)
#     noize = np.random.random(x.shape) * 2 - 1
#     noize_y = y + noize
#     xy = np.array([x, noize_y])
#     c = xy.T
#     return x, noize_y
#
#
# if __name__ == '__main__':
#     from network import *
#     # x, y = generateregr(10, findedfunk, True)
#     # x = x.reshape(x.shape[0], 1)
#     # y = y.reshape(x.shape[0], 1)
#     count = 100
#     x = np.random.rand(1, count)
#     # for i in range(count):
#     #     x[1][i] = 1
#     y = findedfunk(x[0])
#
#     net = Network(


def generateregr(count, func, is_linear=False):
    x = np.arange(0, count, 1).reshape(count, 1) if is_linear else np.random.randn(count, 1)
    y = func(x)
    noize = np.random.random(x.shape) * 2 - 1
    noize_y = y + noize
    xy = np.array([x, noize_y])
    c = xy.T
    return x, noize_y


# if __name__ == '__main__':
#     from network import *
#     # x, y = generateregr(10, findedfunk, True)
#     # x = x.reshape(x.shape[0], 1)
#     # y = y.reshape(x.shape[0], 1)
#     count = 3
#     x = np.random.rand(1, count)
#     # for i in range(count):
#     #     x[1][i] = 1
#     y = findedfunk(x[0])
#     x = np.array([[1, 2, 3]])
#     net = Network([1, 5, 1], [ReLU, x_funk])
#     y1 = y.reshape(count, 1)
#     y2 = np.array([[1], [4], [9]])
#     net.backward(x.T, y2)
#     loss = net.feedforward(x.T)
#     print(x.T)
#     print(y.reshape(count, 1))
#     print(loss)

class TestNetwork:
    pass


if __name__ == '__main__':
    from network import *
    from TestNetwork import *

    net = TestNetwork([1, 1], [x_funk])
    x = np.array([[4, 1.0, 2]])
    
    y2 = np.array([[2], [0.5], [1]])
    loss = net.feedforward(x.T)
    print(loss-y2)
    net.backward(x.T, y2, 10)
    loss = net.feedforward(x.T)
    print(loss-y2)


