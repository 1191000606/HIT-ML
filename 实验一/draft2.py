import copy
import math

import matplotlib.pyplot as plt
import numpy as np
import random
import time
from graph import init_graph, draw_graph

# 数据集大小
num = 100
# 多项式次数
order = 50
# 梯度下降学习率
learning_rate = 0.01
# 梯度下降停止条件
gradient_descent_epsilon = 1e-7
# 使用Adam加速的梯度下降停止条件
gradient_descent_adam_epsilon = 1e-7
# 共轭梯度停止判断条件
conjugate_gradient_epsilon = 1e-10


# 解析法
def analytic_method(X, y):
    return np.dot(np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.transpose(X)), y)


# 梯度下降
def gradient_descent_batch(X, y):
    w = np.array([1.0 for _ in range(order + 1)])
    while True:
        gradient = np.dot(np.transpose(X), np.dot(X, w) - y) / num
        if np.dot(np.transpose(gradient), gradient) < gradient_descent_epsilon:
            break
        for i in range(10000):
            gradient = np.dot(np.transpose(X), np.dot(X, w) - y) / num
            w -= learning_rate * gradient
    return w


# 梯度下降Adam方法
def gradient_descent_adam(X, y):
    # adam中的超参数，一般使用以下值
    adam_beta_1 = 0.9
    adam_beta_2 = 0.99
    inf_small = 1e-8

    # 计算过程中的累计值
    m = np.zeros(order + 1)
    v = np.zeros(order + 1)
    accumulative_adam_beta_1 = adam_beta_1
    accumulative_adam_beta_2 = adam_beta_2

    w = np.array([1.0 for _ in range(order + 1)])

    while True:
        gradient = np.dot(np.transpose(X), np.dot(X, w) - y) / num
        if np.dot(np.transpose(gradient), gradient) < gradient_descent_adam_epsilon:
            break

        for i in range(10000):
            gradient = np.dot(np.transpose(X), np.dot(X, w) - y) / num
            m = adam_beta_1 * m + (1 - adam_beta_1) * gradient
            v = adam_beta_2 * v + (1 - adam_beta_2) * (gradient ** 2)
            M = m / (1 - accumulative_adam_beta_1)
            V = v / (1 - accumulative_adam_beta_2)
            accumulative_adam_beta_1 *= adam_beta_1
            accumulative_adam_beta_2 *= adam_beta_2
            w -= learning_rate * M / (V ** (1 / 2) + inf_small)

    return w


# 参考https://en.wikipedia.org/wiki/Conjugate_gradient_method中伪代码
# 共轭梯度
def conjugate_gradient_method(X, y):
    w = np.array([1.0 for _ in range(order + 1)])
    A = np.matmul(np.transpose(X), X)
    b = np.dot(np.transpose(X), y)
    r = b - np.dot(A, w)

    if np.dot(np.transpose(r), r) < conjugate_gradient_epsilon:
        return w

    p = copy.deepcopy(r)

    while True:
        alpha = np.dot(np.transpose(r), r) / np.dot(np.dot(np.transpose(p), A), p)
        w += alpha * p
        temp = copy.deepcopy(r)
        r -= alpha * np.dot(A, p)

        if np.dot(np.transpose(r), r) < conjugate_gradient_epsilon:
            break

        beta = np.dot(np.transpose(r), r) / np.dot(np.transpose(temp), temp)
        p = r + beta * p

    return w


x = np.linspace(0, 1, num, endpoint=True, dtype=float)
y_raw = np.array(np.sin(2 * np.pi * x))
# 加入噪声
y = np.array([y_raw[i] + random.gauss(0, 1) / 10 for i in range(num)])
# 得到设计的矩阵X
X = np.array([x[i // (order + 1)] ** (i % (order + 1)) for i in range(num * (order + 1))]).reshape(num, order + 1)

init_graph("x", "y", "拟合曲线", dpi=600)
plt.plot(x, y_raw, linewidth=1, label="标准sin函数")

# 求多项式参数
w1 = analytic_method(X, y)
t1 = time.perf_counter()
w3 = gradient_descent_batch(X, y)
t2 = time.perf_counter()

t3 = time.perf_counter()
w4 = gradient_descent_adam(X, y)
t4 = time.perf_counter()
w5 = conjugate_gradient_method(X, y)

# 求拟合曲线
y1 = np.dot(X, w1)
y3 = np.dot(X, w3)
y4 = np.dot(X, w4)
y5 = np.dot(X, w5)

plt.plot(x, y1, linewidth=1, label="解析法")
plt.plot(x, y3, linewidth=1, label="梯度下降")
plt.plot(x, y4, linewidth=1, label="梯度下降Adam优化")
plt.plot(x, y5, linewidth=1, label="共轭梯度")

print(str(order) + "&" + "%.4f" % math.sqrt(np.dot(np.transpose(y1 - y_raw), y1 - y_raw) * 2 / num)
      + "&" + "%.4f" % math.sqrt(np.dot(np.transpose(y3 - y_raw), y3 - y_raw) * 2 / num)
      + "&" + "%.4f" % math.sqrt(np.dot(np.transpose(y4 - y_raw), y4 - y_raw) * 2 / num)
      + "&" + "%.4f" % math.sqrt(np.dot(np.transpose(y5 - y_raw), y5 - y_raw) * 2 / num))

filename = "n" + str(num) + "o" + str(order) + ".jpg"
draw_graph(legend=True, save=True, filename=filename)
print((t2 - t1) * 1000, (t4 - t3) * 1000)
