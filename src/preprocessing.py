import numpy as np


def compute_mean_and_variance(data):
    mean = np.mean(data, axis=0)
    variance = np.var(data, axis=0)
    return mean, variance


# This function reorders randomly the input data
def shuffle_data(input_data):
    # The seed is fixed in order to have always the same reorder
    # This is only for testing purposes
    np.random.seed(0)

    data = np.copy(input_data)
    np.random.shuffle(data)

    return data


# Also known as linear scaling
# x' = (x - min(x)) / (max(x) - min(x))
# where x is data and x' is the normalized data
#
# The function can have both 1 or 2 arguments:
#   - With only 1 argument the scaling is applied to all
#     the columns of the training data (1st argument)
#   - With 2 arguments the scaling is applied to a subset
#     of columns. The first argument is the training data,
#     the second is the subset of columns to scale
#
# In case the function has more than 2 arguments, the default
# behaviour is to scale all the columns
def min_max_scaling(*args):
    training_input = np.array(args[0])
    normalized_dataset = []

    if len(args) == 2:
        columns = args[1]

        # Retrieves the minimum and maximum value in the columns
        min_val = np.amin(training_input[:, columns], axis=0)
        max_val = np.amax(training_input[:, columns], axis=0)

        # Normalizes the dataset
        normalized_dataset = (training_input[:, columns] - min_val) / (max_val - min_val)

        # Substitute in each scaled column the new values
        for column, i in zip(range(normalized_dataset.shape[1]), columns):
            training_input[:, i] = normalized_dataset[:, column]

        normalized_dataset = training_input
    else:
        # Retrieves the minimum and maximum value in the columns
        min_val = np.amin(training_input, axis=0)
        max_val = np.amax(training_input, axis=0)

        # Normalizes the dataset
        normalized_dataset = (training_input - min_val) / (max_val - min_val)

    return normalized_dataset


# x' = (x - mean) / standard deviation
# x is the dataset
# x' is the normalized data set
#
# The function can have both 1 or 2 arguments:
#   - With only 1 argument the scaling is applied to all
#     the columns of the training data (1st argument)
#   - With 2 arguments the scaling is applied to a subset
#     of columns. The first argument is the training data,
#     the second is the subset of columns to scale
#
# In case the function has more than 2 arguments, the default
# behaviour is to scale all the columns
def z_score_scaling(*args):
    training_input = np.array(args[0])
    normalized_dataset = []

    if len(args) == 2:
        columns = args[1]

        # Computes the mean over the affected columns
        mu = np.mean(training_input[:, columns], axis=0)

        # Computes the standard deviation over the affected columns
        sigma = np.std(training_input[:, columns], axis=0)

        # Normalizes the dataset
        normalized_dataset = np.divide(np.subtract(training_input[:, columns], mu), sigma)

        # Substitute in each scaled column the new values
        for column, i in zip(range(normalized_dataset.shape[1]), columns):
            training_input[:, i] = normalized_dataset[:, column]

        normalized_dataset = training_input
    else:
        # Computes the mean over the affected columns
        mu = np.mean(training_input, axis=0)

        # Computes the standard deviation over the affected columns
        sigma = np.std(training_input, axis=0)

        # Normalizes the dataset
        normalized_dataset = np.divide(np.subtract(training_input, mu), sigma)

    return normalized_dataset


# This function implements the one-hot encoding on the input data
#
# Note that this function works only when the input data is composed
# only by integers
def one_hot_encoding(input_data):
    # Computes the maximum value on each column in the input data
    max_values = np.max(input_data, axis=0)

    dictionaries = []

    for value in max_values:
        # Creates an array filled with zero of length 'value'
        encoding = np.zeros(value, dtype=max_values.dtype)

        dictionary_entry = {}

        # Fills the dictionary entry with all the encodings
        # of values in the range [1, value]
        for i in range(1, value + 1):
            tmp = encoding.copy()
            tmp[i - 1] = 1
            dictionary_entry[i] = tmp

        # Appends the dictionary entry to the list
        dictionaries.append(dictionary_entry)

    result_data = []
    # For each row of the input data,
    # encodes the values
    for row in input_data:
        new_row = []

        # Creates new row by concatenating the encoding of each value in the row
        for column, i in zip(row, range(len(row))):
            # Takes the correct encoding for that column
            dic = dictionaries[i]

            new_row = np.concatenate((new_row, dic[column]), axis=None)

        result_data.append(new_row)

    return np.array(result_data, dtype=max_values.dtype)
