def findedfunk1(x):
    y = x * x
    return y


def findedfunk2(x):
    y = np.sin(x)
    return y


def test_classifier(net, count, finded_funk):
    points = np.random.rand(count, 2) * 2 - 1
    answers = np.zeros(count, int)
    funk_border = finded_funk(points.T[0])
    y = points.T[1]
    answers[y > funk_border] = 1
    f_pred = net.feedforward(points)
    net_answers = []
    for i in range(len(f_pred)):
        if f_pred[i, 0] > f_pred[i, 1] :
            net_answers.append(0)
        else:
            net_answers.append(1)
    count_right_answers = 0
    for i in range(len(f_pred)):
        if answers[i] == net_answers[i]:
            count_right_answers += 1

    return float(count_right_answers)/len(net_answers)


if __name__ == '__main__':
    from ActivationFunks import *
    from TestNetwork import *
    from matplotlib import pyplot as plot
    from LossFuncs import Svm
    from PlotDraws import plot_draw_loss

    finded_funk = findedfunk1
    np.random.seed(1)
    funcs = [ReLU(), NoneFunc()]
    lossf = Svm()
    net = TestNetwork([2, 20, 2], funcs, lossf, 1000)
    count = 100

    points = np.random.rand(count, 2)*2-1
    answers = np.zeros(count, int)
    funk_border = finded_funk(points.T[0])
    y = points.T[1]
    answers[y > funk_border] = 1
    loss_values = net.backward(points, answers, 5, 100)
    print("loss: ", end='')
    print(loss_values[-1])
    y_pred = net.feedforward(points)
    print(np.concatenate((y_pred, answers.reshape(count, 1)), axis=1))
    count_right_answers = test_classifier(net, 200, finded_funk)
    print(count_right_answers)
    plot_draw_loss(loss_values)
