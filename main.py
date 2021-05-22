import numpy as np


def findedfunk(x):
    y = x * x
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

def plot_draw(x, y, y_pred):

    arrs = np.concatenate([x,y,y_pred], axis=1)
    arrs = arrs[arrs[:,0].argsort()]
    x = arrs[:, 0]
    y = arrs[:, 1]
    y_pred = arrs[:, 2]
    plot.plot(x, y)
    plot.plot(x, y_pred)
    plot.show()


if __name__ == '__main__':
    from network import *
    from TestNetwork import *
    from matplotlib import pyplot as plot
    net = TestNetwork([1, 100, 1], [ReLU, x_funk])
    count = 6
    x = np.random.rand(1, count)*4-1
    # x = np.arange(0, 10, 1)
    y = findedfunk(x)
    x = x.reshape(count, 1)
    y = y.reshape(count, 1)
    loss = net.feedforward(x)
    print(x)
    print(y)
    net.backward(x, y, 10)
    y_pred = net.feedforward(x)
    print(loss)
    # plot.plot(np.array([[3], [4]]), np.array([[7],[8]]))
    # plot.show()
    # arrs = np.concatenate([x,y,y_pred], axis=1)
    # arrs = arrs[arrs[:,0].argsort()]
    # # x = np.sort(x, axis=0).reshape(count, )
    # # y = np.sort(y, axis=0).reshape(count, )
    # # loss = np.sort(loss, axis=0).reshape(count, )
    # x = arrs[:, 0]
    # y = arrs[:, 1]
    # y_pred = arrs[:, 2]
    # plot.plot(x, y)
    # plot.plot(x, y_pred)
    # plot.show()
    plot_draw(x, y, y_pred)


