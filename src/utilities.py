import json
import os
import numpy as np
import pandas as pd


# This function reads the dataset from the file for the MONK problem
def read_monk_data_set(problem):
    # Reads training dataframe from the input file
    training_df = pd.read_csv("../Monks_problem/monks-" + str(problem) + ".train", index_col=False, sep=" ",
                              names=["", "output", "a1", "a2", "a3", "a4", "a5", "a6", "class"])

    # Reads testing dataframe from the input file
    testing_df = pd.read_csv("../Monks_problem/monks-" + str(problem) + ".test", index_col=False, sep=" ",
                             names=["", "output", "a1", "a2", "a3", "a4", "a5", "a6", "class"])

    # Drops the first column, which is filled with NaN, from each dataframe
    training_df = training_df.dropna(axis=1)
    testing_df = testing_df.dropna(axis=1)
    training_df.drop_duplicates(inplace=True)
    training_df = training_df.sample(frac=1, random_state=1).reset_index()

    # Converts to numpy array the output training data and the output testing data
    training_output = np.array(training_df["output"])
    testing_output = np.array(testing_df["output"])

    # Converts to numpy array the input training data and the input testing data
    training_input = np.array(training_df[["a1", "a2", "a3", "a4", "a5", "a6"]])
    testing_input = np.array(testing_df[["a1", "a2", "a3", "a4", "a5", "a6"]])

    return training_input, training_output, testing_input, testing_output


# This function reads the dataset from file for the ML CUP
def read_ml_cup_data_set():
    # Reads training dataframe from the input file
    training_df = pd.read_csv('../MLcup_problem/ML-CUP22-TR.csv',
                              names=['id', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'o1', 'o2'])
    # Drops rows, which is filled with NaN, from each dataframe
    training_df = training_df.dropna(axis=0)
    training_df.drop_duplicates(inplace=True)
    # shuffle the data to avoid injecting some bias into the test set during the splitting
    training_df = training_df.sample(frac=1, random_state=1).reset_index()

    # Reads testing dataframe from the input file
    blind_testing_df = pd.read_csv('../MLcup_problem/ML-CUP22-TS.csv',
                                   names=['id', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9'])
    # Drops the first column, which is filled with NaN, from each dataframe
    blind_testing_df = blind_testing_df.dropna(axis=0)

    # Converts to a numpy array the input and output data
    input_data = np.array(training_df[['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9']])
    output_data = np.array(training_df[['o1', 'o2']])

    # Converts to a numpy array the input blind data
    blind_testing_input = np.array(blind_testing_df[['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9']])

    return input_data, output_data, blind_testing_input


# This function splits the input data in training data and test data
def split_input_data(input_data, output_data, percentage):

    # Keeps the 'percentage'% of input and output data for training purposes
    training_input = input_data[:(input_data.shape[0] / 100 * percentage).__ceil__(), :]
    training_output = output_data[:(output_data.shape[0] / 100 * percentage).__ceil__(), :]

    # Keeps the remaining percentage of input and output data for testing purposes
    test_input = input_data[(input_data.shape[0] / 100 * percentage).__ceil__():, :]
    test_output = output_data[(output_data.shape[0] / 100 * percentage).__ceil__():, :]

    return training_input, training_output, test_input, test_output


# Function that takes:
#   - performances of the network on validation data
#   - list of hyperparameters
#   - name of the file where to dump the json
#   - boolean flag to distinguish between classification and regression models
# and saves on a json file the model with its performances on validation data
def dump_on_json(performance, hyper_parameters, filename, is_classification):
    # List that stores the name of the activation function and its gradient
    activation_functions = []

    # Retrieves from the list of all the activation functions used,
    # the name of the function and the name of its gradient, and then
    # they are added to the list
    for functions in hyper_parameters[1]:
        activation_functions.append((
            (str(functions[0])).split(" ")[1],
            (str(functions[1])).split(" ")[1]
        ))

    # Creates a tuple with the name of the error function and its gradient
    error_functions = (
        (str(hyper_parameters[2][0])).split(" ")[1],
        (str(hyper_parameters[2][1])).split(" ")[1]
    )

    if is_classification:
        # Creates a dictionary with 'accuracy' as performance metric
        # and all the other hyperparameters
        model = {
            "accuracy": performance,
            "structure": hyper_parameters[0],
            "activation_function": activation_functions,
            "error_function": error_functions,
            "hyper_parameters": hyper_parameters[3],
            "gradient_descent_technique": hyper_parameters[4],
            "mini_batch_size": hyper_parameters[5],
            "regularization_technique": hyper_parameters[6]
        }
    else:
        # Creates a dictionary with 'error' as performance metric
        # and all the other hyperparameters
        model = {
            "error": performance,
            "structure": hyper_parameters[0],
            "activation_function": activation_functions,
            "error_function": error_functions,
            "hyper_parameters": hyper_parameters[3],
            "gradient_descent_technique": hyper_parameters[4],
            "mini_batch_size": hyper_parameters[5],
            "regularization_technique": hyper_parameters[6]
        }

    models = []

    # Checks if the dump file already exists and if it is empty
    file_exists = os.path.exists(filename)
    is_file_empty = file_exists and os.stat(filename).st_size == 0

    if file_exists and not is_file_empty:
        # Reads the content of the dump file
        with open(filename, "r") as open_file:
            read_models = json.load(open_file)

            # For each model in the dump file,
            # it's appended to the list of models
            for single_model in read_models:
                models.append(single_model)
            open_file.close()

        # The new model is appended to the list of models, and the whole
        # models list is dumped as json to the file
        with open(filename, "w") as open_file:
            models.append(model)
            json_object = json.dumps(models, indent=4)
            open_file.write(json_object)
            open_file.close()
    else:
        # Dumps directly the model to the file
        # since it's empty or not exists
        with open(filename, "w+") as open_file:
            models.append(model)
            json_object = json.dumps(models, indent=4)
            open_file.write(json_object)
            open_file.close()
