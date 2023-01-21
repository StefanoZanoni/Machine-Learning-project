import numpy as np


# Also known as linear scaling
# x' = (x - min(x)) / (max(x) - min(x))
# where x is data and x' is the normalized data
def min_max_scaling(*args):
    training_input = np.array(args[0])
    normalized_dataset = []

    if len(args) == 2:
        columns = args[1]
        min_val = np.amin(training_input[:, columns], axis=0)
        max_val = np.amax(training_input[:, columns], axis=0)
        normalized_dataset = (training_input[:, columns] - min_val) / (max_val - min_val)
        for column, i in zip(range(normalized_dataset.shape[1]), columns):
            training_input[:, i] = normalized_dataset[:, column]
        normalized_dataset = training_input
    else:
        min_val = np.amin(training_input, axis=0)
        max_val = np.amax(training_input, axis=0)
        normalized_dataset = (training_input - min_val) / (max_val - min_val)

    return normalized_dataset


# def clipping_scaling(training_input, max_value, min_value):
    # greater_boolean_matrix = np.greater(training_input, max_value)
    # less_boolean_matrix = np.less(training_input, min_value)
    # training_input[greater_boolean_matrix] = max
    # training_input[less_boolean_matrix] = min

# x' = (x - mean) / standard deviation
# x is the dataset
# x' is the normalized data set
def z_score_scaling(*args):
    training_input = np.array(args[0])
    normalized_dataset = []

    if len(args) == 2:
        columns = args[1]
        mu = np.mean(training_input[:, columns], axis=0)
        sigma = np.std(training_input[:, columns], axis=0)
        normalized_dataset = np.divide(np.subtract(training_input[:, columns], mu), sigma)
        for column, i in zip(range(normalized_dataset.shape[1]), columns):
            training_input[:, i] = normalized_dataset[:, column]
        normalized_dataset = training_input
    else:
        mu = np.mean(training_input, axis=0)
        sigma = np.std(training_input, axis=0)
        normalized_dataset = np.divide(np.subtract(training_input, mu), sigma)

    return normalized_dataset


def one_hot_encoding(input_data):
    max_values = np.max(input_data, axis=0)
    dictionaries = []
    # [{}, {}, {}]
    for value in max_values:
        encoding = np.zeros(value, dtype=max_values.dtype)
        dictionary_entry = {}
        for i in range(1, value + 1):
            tmp = encoding.copy()
            tmp[i - 1] = 1
            dictionary_entry[i] = tmp
        dictionaries.append(dictionary_entry)

    result_data = []
    for row in input_data:
        new_row = []
        for column, i in zip(row, range(len(row))):
            dic = dictionaries[i]
            new_row = np.concatenate((new_row, dic[column]), axis=None)
        result_data.append(new_row)
        # [[0, 0 ,1] , [1, 1, 0]]

    return np.array(result_data, dtype=max_values.dtype)


def shuffle_data(input_data):
    np.random.seed(0)
    data = np.copy(input_data)
    np.random.shuffle(data)
    return data
