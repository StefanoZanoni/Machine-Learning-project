import numpy as np
import pandas as pd

from src import activation_functions
from src import error_functions
from src import kfold

if __name__ == '__main__':

    # (input, output) type
    dt = object

    training_df = pd.read_csv('../MLcup_problem/ML-CUP22-TR.csv',
                              names=['id', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'o1', 'o2'])
    training_df = training_df.dropna(axis=0)
    blind_testing_df = pd.read_csv('../MLcup_problem/ML-CUP22-TS.csv',
                                   names=['id', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9'])
    blind_testing_df = blind_testing_df.dropna(axis=0)

    input_data = np.array(training_df[['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9']])
    output_data = np.array(training_df[['o1', 'o2']])
    training_input = input_data[:(input_data.shape[0] / 100 * 80).__ceil__(), :]
    training_output = output_data[:(output_data.shape[0] / 100 * 80).__ceil__(), :]
    test_input = input_data[(input_data.shape[0] / 100 * 80).__ceil__():, :]
    test_output = output_data[(output_data.shape[0] / 100 * 80).__ceil__():, :]

    blind_testing_input = np.array(blind_testing_df[['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9']])

    activation_functions = [
                            [(activation_functions.tanh, activation_functions.tanh_gradient),
                             (activation_functions.tanh, activation_functions.tanh_gradient),
                             (activation_functions.swish, activation_functions.swish_gradient),
                             (activation_functions.linear, activation_functions.linear_gradient)]
                            ]
    error_function = (error_functions.mee, error_functions.mee_gradient)

    # Defines the list of hyperparameters to try
    # Each element of the list is a list that contains a tuple (or more than one if we are using activation functions
    # that requires an additional parameters) with the name of the parameter and the value to try
    hyper_parameters = [[('learning_rate', 0.005)]]

    # Define a list of regularization to try. Each element of the list is a tuple that contains the name of the
    # regularization technique and then its value.
    # If there's no need to try a regularization technique is sufficient to add the tuple ("None", 0)
    regularization_techniques = [('None', 0)]

    best_model = kfold.k_fold_cross_validation(training_input, training_output,
                                               [("structures", [[9, 12, 12, 8, 2]]),
                                                ("activation_functions",
                                                 activation_functions),
                                                ("error_functions",
                                                 [error_function]),
                                                ("hyper_parameters",
                                                 hyper_parameters),
                                                ("gradient_descent_techniques",
                                                 ["None"]),
                                                ("mini_batch_sizes", [4]),
                                                ("regularization_techniques",
                                                 regularization_techniques)],
                                               128, False, "../MLcup_models.json", False, dt)

    performance = best_model.compute_performance(test_input, test_output)
    print('performance on the test set: ' + str(performance))

    output_x = []
    output_y = []
    for input in blind_testing_input:
        output = best_model.forward(input)
        output_x.append(output[0][0])
        output_y.append(output[1][0])

    blind_len = blind_testing_input.shape[0]
    ids = [i for i in range(1, blind_len + 1)]

    df = pd.DataFrame(np.array([ids, output_x, output_y]).T, columns=['id', 'output_x', 'output_y'])
    # Converts the first column, the ids, in integers
    df[df.columns[0]] = df[df.columns[0]].astype(int)

    # Writes the dataframe to a .csv file
    df.to_csv('../output/ZDS_ML-CUP22-TS.csv', index=False)
