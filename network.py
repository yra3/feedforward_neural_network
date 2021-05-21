import numpy as np

'''Реализуйте полносвязную нейронную сеть. На вход сеть должна принимать матрицу размерности 𝑁×𝐷 элементов, 
где 𝑁 – количество элементов в одном пакете данных, 𝐷 – количество элементов в каждом из 𝑁 векторов, описывающих данные. 
Возвращать сеть должна матрицу размера 𝑁×С, где 𝐶 – размер выходных данных для каждого вектора. 
На данном этапе можете взять для 𝐶 любое небольшое число (5-10 элементов).

Сеть должна включать в себя также фу�import numpy as np

Реализуйте полносвязную нейронную сеть. На вход сеть должна принимать матрицу размерности 𝑁×𝐷 элементов, 
где 𝑁 – количество элементов в одном пакете данных, 𝐷 – количество элементов в каждом из 𝑁 векторов, описывающих данные. 
Возвращать сеть должна матрицу размера 𝑁×С, где 𝐶 – размер выходных данных для каждого вектора. 
На данном этапе можете взять для 𝐶 любое небольшое число (5-10 элементов).

Сеть должна включать в себя также функции активации: ReLU – обязательно, другие на выбор.

Сеть должна быть реализована таким образом, чтобы тип слоя, количество слоев и функций активации можно было задавать. 
В этом вам может помочь наследование. Например, вы можете сделать базовый класс (или абстрактный класс, или интерфейс, это зависит от языка, на котором вы пишете) 
и реализовывать все компоненты, слои и функции активации, как наследники этого базового класса.'''

N = 2
D = 2


def ReLU(x):
    return np.maximum(x, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(z):# Производная сигмоидальной функции
    return sigmoid(z)*(1-sigmoid(z))

def ReLU(z):
    return np.maximum(0,z)

def ReLU_prime(z):
    z = np.where(z > 0, z, 1)
    return np.where(z <= 0, z, 0)

def sigpriozv(x):
    return np.exp(-x)/np.square(1 + np.exp(x))

def nonef_prime(x):
    return x;


def x_funk(x):
    return x


class Network:
    def __init__(self, sizes, activation_func):
        self.sizes = sizes
        self.count_input = sizes[0]
        self.count_output = sizes[-1]
        self.functions = activation_func
        self.num_layers = len(sizes)  # число слоев
        self.learning_speed = 1000
        self.loss_func = None
        self.biases = [np.random.randn(1, y) for y in sizes[1:]]
        self.weights = [np.random.randn(y+1, x) for x, y in zip(sizes[1:], sizes[:-1])]

    def feedforward(self, a):
        for w, b, func in zip(self.weights, self.biases, self.functions):
            b = np.ones((a.shape[0], 1))
            a = np.concatenate((a, b), axis=1)
            a = func(a.dot(w)+b)
        return a

    def backward(self, a, y):
        for i in range(self.learning_speed):
            a_without_b_save = []
            a_with_b_save = []
            a_with_b_multiplied_w_save = []
            a_after_funk_save = []
            # grads_w = [np.zeros(w.shape) for w in self.weights]

            for w, b, func in zip(self.weights, self.biases, self.functions):

                a_without_b_save.append(a)
                b = np.ones((a.shape[0], 1))
                a = np.concatenate((a, b), axis=1)
                a_with_b_save.append(a)
                a_with_b_multiplied_w_save.append(a.dot(w))
                a = func(a.dot(w))
                a_after_funk_save.append(a)

            y_pred = a
            loss = np.square(y_pred - y)
            y_pred = a_after_funk_save[-1] = loss
            is_first = True
            # for i in range(1, len(self.sizes)):
            #     f_p = funk_prime(a_with_b_multiplied_w_save[-i])

            for w, funk_prime, save, a_after_funk, a_with_b_w in zip(self.weights[::-1], [ReLU_prime, nonef_prime][::-1]
                    , a_with_b_save[::-1], a_after_funk_save[::-1], a_with_b_multiplied_w_save[::-1]):
                f_p = funk_prime(a_with_b_w)
                y_pred = funk_prime(y_pred)
                if is_first:
                    is_first = False
                else:
                    y_pred = np.delete(y_pred, (y_pred.shape[1]-1), axis=1)
                delta_w = (save.T.dot(y_pred))
                w -= delta_w / self.learning_speed
                y_pred = y_pred.dot(w.T)





def rndGenerate(N, D):
    return np.random.rand(N, D)


if __name__ == "__main__":
    matrix = np.random.rand(N,D)

    nw = Network([2, 6, 5], [sigmoid, x_funk])
    nw.feedforward(matrix)
