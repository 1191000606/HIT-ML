train_filename = "./data/iris/iris.data"
test_filename = "./data/iris/iris.data"
field_list = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
discrete_field_dict = {}
label_dict = {"Iris-setosa": 0, "Iris-versicolor": 1}
get_log_list = []

iris_config = [train_filename, test_filename, field_list, discrete_field_dict, label_dict, get_log_list]
