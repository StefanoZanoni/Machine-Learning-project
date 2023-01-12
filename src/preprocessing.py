import numpy as np

# axis = 0 in numpy (column wise)
# |
# V

# axis = 1 (row wise)
#  -->

# Also known as linear scaling
# x' = (x - min(x)) / (max(x) - min(x))
# where x is data and x' is the normalized data
# AS


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


def shuffle_data(input_data):
    data = np.copy(input_data)
    np.random.shuffle(data)
    return data
