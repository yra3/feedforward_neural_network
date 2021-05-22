def findedfunk1(x):
    y = x * x
    return y


def findedfunk2(x):
    y = np.sin(x)
    return y

if __name__ == '__main__':
    from ActivationFunks import *
    from TestNetwork import *
    from matplotlib import pyplot as plot
    from LossFuncs import Svm
    from PlotDraws import plot_draw_loss

    np.random.seed(1)
    funcs = [ReLU(), NoneFunc()]
    lossf = Svm()
    net = TestNetwork([2, 20, 2], funcs, lossf, 1000)
    count = 100

    points = np.random.rand(count, 2)*2-1
    answers = np.zeros(count, int)
    funk_border = findedfunk1(points.T[0])
    y = points.T[1]
    answers[y > funk_border] = 1
    loss_values = net.backward(points, answers, 5, 100)
    print("loss: ", end='')
    print(loss_values[-1])
    y_pred = net.feedforward(points)
    print(np.concatenate((y_pred, answers.reshape(count, 1)), axis=1))

    plot_draw_loss(loss_values)
