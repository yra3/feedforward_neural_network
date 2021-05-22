def findedfunk1(x):
    y = x * x
    return y


def findedfunk2(x):
    y = np.sin(x)
    return y

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
    from ActivationFunks import *
    from TestNetwork import *
    from matplotlib import pyplot as plot

    np.random.seed(1)
    funcs = [ReLU(), NoneFunc()]
    net = TestNetwork([2, 20, 2], funcs, 10000)
    count = 5

    points = np.random.rand(count, 2)*2-1
    answers = np.zeros(count)
    funk_estimate = findedfunk1(points.T[0])
    y = points.T[1]
    answers[y > funk_estimate] = 1
    net.backward(points, y, 100, 1000)
    y_pred = net.feedforward(x)
    print(y_pred)

    plot_draw(x, y, y_pred)