import numpy as np


def get_data_guass(train_size, validation_size, test_size, positive_sample_radio, naive=True):
    if naive:
        cov12 = 0.0
    else:
        cov12 = 0.2

    mean_array_positive = [1, 1]
    cov_matrix_positive = [[0.4, cov12], [cov12, 0.3]]

    mean_array_negative = [-1, -1]
    cov_matrix_negative = [[0.4, cov12], [cov12, 0.3]]

    train_X_list, train_Y_list = get_guass_distribution(train_size, int(train_size * positive_sample_radio), mean_array_positive, cov_matrix_positive,
                                                        mean_array_negative, cov_matrix_negative)
    validation_X_list, validation_Y_list = get_guass_distribution(validation_size, int(validation_size * positive_sample_radio), mean_array_positive,
                                                                  cov_matrix_positive, mean_array_negative, cov_matrix_negative)
    test_X_list, test_Y_list = get_guass_distribution(test_size, int(test_size * positive_sample_radio), mean_array_positive, cov_matrix_positive,
                                                      mean_array_negative, cov_matrix_negative)
    return train_X_list, train_Y_list, validation_X_list, validation_Y_list, test_X_list, test_Y_list


def get_guass_distribution(total_size, positive_size, mean_array_positive, cov_matrix_positive, mean_array_negative, cov_matrix_negative):
    X_list = [(0.0, 0.0)] * total_size
    X_list[:positive_size] = np.random.multivariate_normal(mean_array_positive, cov_matrix_positive, size=positive_size)
    X_list[positive_size:] = np.random.multivariate_normal(mean_array_negative, cov_matrix_negative, size=total_size - positive_size)
    X_list = [list(sample) for sample in X_list]
    Y_list = [1] * positive_size + [0] * (total_size - positive_size)
    return X_list, Y_list
