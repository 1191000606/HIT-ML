import copy

import numpy as np
import pandas as pd

from data.config.adult_config import adult_config
from data.config.iris_config import iris_config


def get_data_uci(uci_data_name):
    if uci_data_name == "iris":
        data_config = iris_config
        train_nrows = 100
        test_nrows = 100
        validation_index = [25, 76]
    elif uci_data_name == "adult":
        data_config = adult_config
        train_nrows = 1000
        test_nrows = 1000
        validation_index = [0, 500]
    else:
        print("尚无此数据集")
        return None

    train_filename = data_config[0]
    test_filename = data_config[1]
    field_list = data_config[2]
    discrete_field_dict = data_config[3]
    label_dict = data_config[4]
    get_log_list = data_config[5]

    train_X_list, train_Y_list = read_uci_data(train_filename, train_nrows, field_list, discrete_field_dict, label_dict, get_log_list)
    if test_filename == train_filename:
        test_X_list, test_Y_list = train_X_list, train_Y_list
    else:
        test_X_list, test_Y_list = read_uci_data(test_filename, test_nrows, field_list, discrete_field_dict, label_dict, get_log_list)
    validation_X_list = test_X_list[validation_index[0]:validation_index[1]]
    validation_Y_list = test_Y_list[validation_index[0]:validation_index[1]]

    return train_X_list, train_Y_list, validation_X_list, validation_Y_list, test_X_list, test_Y_list


def read_uci_data(filename, nrows, field_list, discrete_field_dict, label_dict, get_log_list):
    columns = copy.deepcopy(field_list)
    columns.append("label")
    raw_data = pd.read_csv(filename, names=columns, nrows=nrows)

    processed_data = []
    processed_data_label = []

    for i in range(len(raw_data)):
        processed_sample = []
        raw_sample = raw_data.iloc[i]

        try:
            for field in field_list:
                if field in discrete_field_dict:
                    processed_sample.append(discrete_field_dict[field][raw_sample.loc[field]])
                else:
                    processed_sample.append(float(raw_sample.loc[field]))
        except KeyError:
            continue
        except ValueError:
            continue

        processed_data_label.append(label_dict[raw_sample.loc["label"]])
        processed_data.append(processed_sample)

    if len(get_log_list) != 0:
        for sample in processed_data:
            for index in get_log_list:
                if sample[index] <= 10:
                    sample[index] = 1
                else:
                    sample[index] = np.log10(sample[index])

            sample.insert(0, 1)
    else:
        for sample in processed_data:
            sample.insert(0, 1)

    return processed_data, processed_data_label
