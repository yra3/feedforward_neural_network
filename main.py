import numpy as np


def findedfunk1(x):
    y = x * x
    return y


def findedfunk2(x):
    y = np.sin(x)
    return y


def findedfunk3(x):
    return np.sqrt(np.abs(x))-1
    Y = 2.0*x*x*x -x*x +x-2


def generateregr(count, func, is_linear=False):
    x = np.arange(0, count, 1).reshape(count, 1) if is_linear else np.random.randn(count, 1)
    y = func(x)
    noize = np.random.random(x.shape) * 2 - 1
    noize_y = y + noize
    xy = np.array([x, noize_y])
    c = xy.T
    return x, noize_y



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


def test_regression(net, count, finded_func, loss):
    x = np.random.rand(1, count) * 10 - 5
    y = finded_func(x)
    x = x.reshape(count, 1)
    y = y.reshape(count, 1)
    y_pred = net.feedforward(x)
    print('loss: ', end='')
    print(loss.loss(y, y_pred))

    plot_draw_funks(x, y, y_pred)


if __name__ == '__main__':
    from ActivationFunks import *
    from TestNetwork import *
    from matplotlib import pyplot as plot
    from LossFuncs import L2
    from PlotDraws import plot_draw_loss
    # np.random.seed(1)
    funcs = [ReLU(), NoneFunc()]
    lossf = L2()
    net = TestNetwork([1, 100, 1], funcs, lossf)

    count = 20
    x = np.random.rand(1, count)*10-5
    # x = np.arange(0, 10, 1)
    finded_func = findedfunk1
    y = finded_func(x)
    x = x.reshape(count, 1)
    y = y.reshape(count, 1)
    print(x)
    print(y)
    loss_values = net.backward(x, y, 30, 100, 50000, 0.3)
    y_pred = net.feedforward(x)
    print(y_pred)
    print('loss: ', end='')
    print(loss_values[-1])

    plot_draw_funks(x, y, y_pred)
    plot_draw_loss(loss_values)
    test_regression(net, 1000, finded_func, lossf)


