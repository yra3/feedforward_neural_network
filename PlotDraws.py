from matplotlib import pyplot as plot
import numpy as np


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
    loss = loss[loss < 10000]
    iterations = list(range(len(loss)))
    plot.plot(iterations, loss)
    plot.show()