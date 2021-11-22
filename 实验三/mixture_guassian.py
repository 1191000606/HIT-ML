import copy
import random

import matplotlib.pyplot as plt
import pandas as pd

from graph import init_graph, draw_graph, font_title, font_label
import numpy as np

dimension_num = 0
k = 0
iteration_times_K_Means = 40
iteration_times_GMM = 40


def get_mixture_guassian(size_list, mean_list, cov_matrix_list):
    total_data = [[0.0] * dimension_num] * sum(size_list)
    temp_index = 0
    seed = []
    for i in range(k):
        temp_list = np.random.multivariate_normal(mean_list[i], cov_matrix_list[i], size_list[i])
        total_data[temp_index:temp_index + size_list[i]] = [temp.tolist() + [i] for temp in temp_list]
        seed.append(total_data[temp_index])
        temp_index += size_list[i]

    random.shuffle(total_data)
    return [data[:-1] for data in total_data], [data[-1] for data in total_data], seed


def get_iris():
    field_list = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    label_dict = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
    raw_data = pd.read_csv("iris.data", names=field_list + ["label"], nrows=150)

    processed_data = []
    processed_data_label = []

    for i in range(len(raw_data)):
        processed_sample = []
        raw_sample = raw_data.iloc[i]

        try:
            for field in field_list:
                processed_sample.append(float(raw_sample.loc[field]))
        except KeyError:
            continue
        except ValueError:
            continue

        processed_data_label.append(label_dict[raw_sample.loc["label"]])
        processed_data.append(processed_sample)

    seed = [processed_data[0] + [0], processed_data[50] + [1], processed_data[100] + [2]]
    return processed_data, processed_data_label, seed


def K_Means(X_list, seed):
    U = np.array([temp[:-1] for temp in seed])
    label = [temp[-1] for temp in seed]
    X = np.array(X_list)
    Y = [0] * len(X_list)

    for i in range(iteration_times_K_Means):
        for j in range(len(X_list)):
            min_distance = np.dot(X[j] - U[0], X[j] - U[0])
            for m in range(k):
                temp_distance = np.dot(X[j] - U[m], X[j] - U[m])
                if temp_distance < min_distance:
                    min_distance = temp_distance
                    Y[j] = label[m]

        U = np.array([[0.0] * dimension_num] * k)
        Num = np.zeros(k)
        for j in range(len(X_list)):
            U[Y[j]] += X[j]
            Num[Y[j]] += 1

        for m in range(k):
            U[m] /= Num[m]
    return Y


def GMM(X_list, seed):
    length = len(X_list)
    U = np.array([temp[:-1] for temp in seed])
    label = [temp[-1] for temp in seed]
    X = np.array(X_list)
    Cov = [np.diag([1] * dimension_num)] * k
    P = [1 / k] * k
    result = [[1 / k] * k] * length
    likelihood_list = [0.0] * iteration_times_GMM

    for t in range(iteration_times_GMM):
        for j in range(length):
            for i in range(k):
                result[j][i] = P[i] * (1 / np.sqrt(np.linalg.det(Cov[i]))) * np.exp(
                    -0.5 * np.dot((np.dot((X[j] - U[i]).T, np.linalg.pinv(Cov[i]))), X[j] - U[i]))
            temp = sum(result[j])
            likelihood_list[t] += np.log(temp)
            result[j] = [result[j][i] / temp for i in range(k)]

        U = np.array([[0.0] * dimension_num] * k)
        for i in range(k):
            temp = sum(result[j][i] for j in range(length))
            U[i] = sum([X[j] * result[j][i] for j in range(length)]) / temp
            Cov[i] = sum([np.dot((X[j] - U[i]).reshape(dimension_num, 1), (X[j] - U[i]).reshape(1, dimension_num)) * result[j][i] for j in
                          range(length)]) / temp
            P[i] = sum([result[j][i] for j in range(length)]) / length

    Y = [result[j].index(max(result[j])) for j in range(length)]
    Y = [label[Y[j]] for j in range(length)]
    return Y, likelihood_list


def test(predict_Y_list, Y_list):
    correct_num = 0
    total_num = len(Y_list)
    for i in range(len(Y_list)):
        if predict_Y_list[i] == Y_list[i]:
            correct_num += 1
    return correct_num / total_num


def show(X_list, raw_Y_list, K_Means_Y_list, GMM_Y_list, likelihood_list):
    init_graph(plt)
    if dimension_num==2:
        total_Y_list = [raw_Y_list, K_Means_Y_list, GMM_Y_list]
        title_list = ["生成结果", "K_Means", "GMM", "似然值"]
        X_list_one = [x[0] for x in X_list]
        X_list_two = [x[1] for x in X_list]

        for i in range(3):
            plt.subplot(2, 2, i + 1)
            plt.title(title_list[i], font=font_title)
            plt.scatter(X_list_one, X_list_two, c=total_Y_list[i], marker="x")

        plt.subplot(2, 2, 4)

    plt.xlabel(fontdict=font_label, xlabel="迭代次数")
    plt.ylabel(fontdict=font_label, ylabel="似然值")
    plt.plot([i + 1 for i in range(len(likelihood_list))], likelihood_list, linewidth=1)
    draw_graph(plt)


if __name__ == '__main__':
    size_list = [400, 100, 200]
    mean_list = [[0, 0], [20, -20], [15, 10]]
    cov_matrix_list = [[[5, 0.2], [0.2, 40]], [[10, 0.5], [0.5, 20]], [[10, 0.3], [0.3, 15]]]
    k = 3
    dimension_num = 2

    X_list, Y_list, seed = get_mixture_guassian(size_list, mean_list, cov_matrix_list)
    # X_list, Y_list, seed = get_iris()
    # k = 3
    # dimension_num = 4

    K_Means_Y_list = K_Means(X_list, seed)
    GMM_Y_list, likelihood_list = GMM(X_list, seed)

    print("K_Means:", test(K_Means_Y_list, Y_list))
    print("GMM:", test(GMM_Y_list, Y_list))
    show(X_list, Y_list, K_Means_Y_list, GMM_Y_list,likelihood_list)
