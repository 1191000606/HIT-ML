import copy

import numpy as np
import cv2

from graph import init_graph, font_title, draw_graph
import matplotlib.pyplot as plt


def generate_data(size, raw_dimension_num):
    mean_list = [5.0] * raw_dimension_num
    cov_list = [[0.0] * i + [2.0] + [0.0] * (raw_dimension_num - i - 1) for i in range(raw_dimension_num)]
    cov_list[0][0] = 0.5

    data = np.random.multivariate_normal(mean_list, cov_list, size)
    return data


def PCA(data, compress_dimension_num):
    raw_dimension_num = len(data[0])
    mean_array = np.mean(data, axis=0)
    for i in range(len(data)):
        data[i] -= mean_array

    cov_matrix = (data.T @ data) / len(data)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sort_index = np.argsort(eigenvalues)
    # 变换矩阵 or 未移位的基向量
    compress_matrix = np.array([[0.0] * compress_dimension_num] * raw_dimension_num)
    for i in range(compress_dimension_num):
        index = sort_index[-1 - i]
        compress_matrix[:, i] = eigenvectors[:, index]

    # 压缩后的矩阵
    compressed_data = data @ compress_matrix

    # 解压后的矩阵
    decompressed_data = compressed_data @ compress_matrix.T
    for i in range(len(data)):
        # 加上中心值
        decompressed_data[i] += mean_array

    return mean_array, compress_matrix, compressed_data, decompressed_data


def draw(data, decompressed_data, mean_array, compress_matrix):
    raw_dimension_num = len(data[0])

    if raw_dimension_num == 2:
        init_graph(plt, dpi=400)
        plt.title("二维压缩为一维", font=font_title)
        plt.scatter(data[:, 0], data[:, 1], c='g', marker='.')
        plt.scatter(decompressed_data[:, 0], decompressed_data[:, 1], c='r', marker='.')
        plt.plot([mean_array[0] - 2.5 * compress_matrix[0][0], mean_array[0] + 2.5 * compress_matrix[0][0]],
                 [mean_array[1] - 2.5 * compress_matrix[1][0], mean_array[1] + 2.5 * compress_matrix[1][0]], linewidth=1)
        draw_graph(plt)
    elif raw_dimension_num == 3:
        init_graph(plt, dpi=400)
        ax = plt.axes(projection='3d')
        plt.title("三维压缩为二维", font=font_title)
        ax.scatter3D(data[:, 0], data[:, 1], data[:, 2], c='g', marker='.')
        ax.scatter3D(decompressed_data[:, 0], decompressed_data[:, 1], decompressed_data[:, 2], c='r', marker='.')
        ax.plot3D([mean_array[0] - 2.5 * compress_matrix[0][0], mean_array[0] + 2.5 * compress_matrix[0][0]],
                  [mean_array[1] - 2.5 * compress_matrix[1][0], mean_array[1] + 2.5 * compress_matrix[1][0]],
                  [mean_array[2] - 2.5 * compress_matrix[2][0], mean_array[2] + 2.5 * compress_matrix[2][0]], linewidth=1)
        ax.plot3D([mean_array[0] - 2.5 * compress_matrix[0][1], mean_array[0] + 2.5 * compress_matrix[0][1]],
                  [mean_array[1] - 2.5 * compress_matrix[1][1], mean_array[1] + 2.5 * compress_matrix[1][1]],
                  [mean_array[2] - 2.5 * compress_matrix[2][1], mean_array[2] + 2.5 * compress_matrix[2][1]], linewidth=1)
        draw_graph(plt)
    else:
        print("Matplotlib can't draw if dimension num>3")


def PSNR(img1, img2):
    img0 = img1 - img2
    return 10 * np.log10(65025 * len(img0) * len(img0[0]) / np.sum(img0 ** 2))


if __name__ == '__main__':
    # raw_dimension_num = 3
    # data = generate_data(size=100, raw_dimension_num=raw_dimension_num)
    # mean_array, compress_matrix, compressed_data, decompressed_data = PCA(copy.deepcopy(data), raw_dimension_num - 1)
    # draw(data, decompressed_data, mean_array, compress_matrix)

    img = cv2.imread("./figures/biden.jpg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.subplot(3, 3, 1)
    plt.title("原图", font=font_title)
    plt.imshow(img_gray)
    img_gray_float = np.array(img_gray, dtype=np.float32, copy=True)
    dimension_list = [2, 5, 10, 15, 20, 50, 100, 164]  # biden
    # dimension_list = [2, 5, 10, 15, 20, 50, 100, 193]  # trump
    for i in range(8):
        plt.subplot(3, 3, i + 2)
        mean_array, compress_matrix, compressed_data, decompressed_data = PCA(img_gray_float, dimension_list[i])
        decompressed_data_int = np.array(decompressed_data, dtype=np.int32, copy=True)
        plt.title(str(dimension_list[i]) + "维,PSNR=" + str(PSNR(img_gray_float, np.array(decompressed_data, dtype=np.float32, copy=True))),
                  font=font_title)
        plt.imshow(decompressed_data_int)
    plt.show()
