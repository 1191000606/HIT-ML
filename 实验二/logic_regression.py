import copy

import numpy as np

from data_guass import get_data_guass
from data_uci import get_data_uci

gradient_regulation_radio = 0
newton_regulation_radio = 0


def sigmoid_trick(m):
    for i in range(len(m)):
        a = m[i]
        if a >= 0:
            m[i] = 1 / (1 + np.exp(-1 * a))
        else:
            temp = np.exp(a)
            m[i] = temp / (1 + temp)
    return m


def gradient_descent(train_X, train_Y, validation_X, validation_Y, regulation_radio=0.0):
    dimension_num = len(train_X[0])

    # adam中的超参数，一般使用以下值
    adam_beta_1 = 0.9
    adam_beta_2 = 0.99
    inf_small = 1e-8
    learning_rate = 0.01

    # 计算过程中的累计值
    m = np.zeros(dimension_num)
    v = np.zeros(dimension_num)
    accumulative_adam_beta_1 = adam_beta_1
    accumulative_adam_beta_2 = adam_beta_2

    w = np.array([0.1 for _ in range(dimension_num)])
    temp_w = copy.deepcopy(w)
    accuracy = test(validation_X, w, validation_Y)
    accuracy_list = [accuracy]
    iteration_times = 0

    while True:
        for i in range(10000):
            gradient = np.dot(train_X.T, sigmoid_trick(-1 * np.dot(train_X, w)) - train_Y) - regulation_radio * w

            m = adam_beta_1 * m + (1 - adam_beta_1) * gradient
            v = adam_beta_2 * v + (1 - adam_beta_2) * (gradient ** 2)
            M = m / (1 - accumulative_adam_beta_1)
            V = v / (1 - accumulative_adam_beta_2)
            accumulative_adam_beta_1 *= adam_beta_1
            accumulative_adam_beta_2 *= adam_beta_2
            w += learning_rate * M / (V ** (1 / 2) + inf_small)

        iteration_times += 1
        accuracy = test(validation_X, w, validation_Y)

        if accuracy < accuracy_list[-1] or iteration_times >= 15:
            break
        else:
            temp_w = copy.deepcopy(w)
            accuracy_list.append(accuracy)

    return temp_w, iteration_times


def Newton_method(train_X, train_Y, validation_X, validation_Y, regulation_radio=0.0):
    dimension_num = len(train_X[0])
    w = np.array([0.1 for _ in range(dimension_num)])
    temp_w = copy.deepcopy(w)
    accuracy = test(validation_X, w, validation_Y)
    accuracy_list = [accuracy]
    iteration_times = 0

    while True:
        for i in range(10000):
            sigmoid = sigmoid_trick(-1 * np.dot(train_X, w))
            a = np.dot(train_X.T, sigmoid - train_Y)
            b = train_X.T @ (np.reshape(sigmoid * (1 - sigmoid), (len(sigmoid), 1)) * train_X)
            delta_w = np.dot(np.linalg.pinv(b), a) - w * regulation_radio
            w += delta_w

        iteration_times += 1
        accuracy = test(validation_X, w, validation_Y)

        if iteration_times >= 15 or accuracy < accuracy_list[-1]:
            break
        else:
            temp_w = copy.deepcopy(w)
            accuracy_list.append(accuracy)

    return temp_w, iteration_times


def test(X, w, Y):
    X = np.array(X)
    Y = np.array(Y)

    predict_result = np.dot(X, w)
    right_num = 0
    for i in range(len(X)):
        if (predict_result[i] >= 0 and Y[i] == 0) or (predict_result[i] < 0 and Y[i] == 1):
            right_num += 1

    return float(right_num) / len(X)


if __name__ == '__main__':
    train_X_list, train_Y_list, validation_X_list, validation_Y_list, test_X_list, test_Y_list = get_data_guass(100, 20, 100, 0.7, False)
    # train_X_list, train_Y_list, validation_X_list, validation_Y_list, test_X_list, test_Y_list = get_data_uci("iris")
    # train_X_list, train_Y_list, validation_X_list, validation_Y_list, test_X_list, test_Y_list = get_data_uci("adult")

    train_X = np.array(train_X_list)
    train_Y = np.array(train_Y_list)
    validation_X = np.array(validation_X_list)
    validation_Y = np.array(validation_Y_list)
    test_X = np.array(test_X_list)
    test_Y = np.array(test_Y_list)

    w1, iteration_times_1 = gradient_descent(train_X, train_Y, validation_X, validation_Y, gradient_regulation_radio)
    print((test(test_X_list, w1, test_Y_list)), w1, iteration_times_1)

    w2, iteration_times_1 = Newton_method(train_X, train_Y, validation_X, validation_Y, newton_regulation_radio)
    print(test(test_X, w2, test_Y), w2, iteration_times_1)
