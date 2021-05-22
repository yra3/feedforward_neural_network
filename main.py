import numpy as np


def findedfunk1(x):
    y = x * x
    return y


def findedfunk2(x):
    y = np.sin(x)
    return y


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

def plot_draw_funks(x, y, y_pred):

    arrs = np.concatenate([x,y,y_pred], axis=1)
    arrs = arrs[arrs[:,0].argsort()]
    x = arrs[:, 0]
    y = arrs[:, 1]
    y_pred = arrs[:, 2]
    plot.plot(x, y)
    plot.plot(x, y_pred)
    plot.show()


def plot_draw_loss(loss):
    loss = np.array(loss)
    loss = loss[loss > 10000]
    iterations = list(range(len(loss)))
    plot.plot(iterations, loss)
    plot.show()


if __name__ == '__main__':
    from ActivationFunks import *
    from TestNetwork import *
    from matplotlib import pyplot as plot
    from LossFuncs import L2
    np.random.seed(1)
    funcs = [ReLU(), NoneFunc()]
    lossf = L2()
    net = TestNetwork([1, 20, 1], funcs, lossf, 100)

    count = 20
    x = np.random.rand(1, count)*10-5
    # x = np.arange(0, 10, 1)
    y = findedfunk1(x)
    x = x.reshape(count, 1)
    y = y.reshape(count, 1)
    print(x)
    print(y)
    loss_values = net.backward(x, y, 30, 100)
    y_pred = net.feedforward(x)
    print(y_pred)

    plot_draw_funks(x, y, y_pred)
    plot_draw_loss(loss_values)


