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
                            [(activation_functions.parametric_relu, activation_functions.parametric_relu_gradient),
                             (activation_functions.parametric_relu, activation_functions.parametric_relu_gradient),
                             (activation_functions.linear, activation_functions.linear_gradient)],
                            [(activation_functions.selu, activation_functions.selu_gradient),
                             (activation_functions.selu, activation_functions.selu_gradient),
                             (activation_functions.linear, activation_functions.linear_gradient)]
                            ]
    error_function = (error_functions.mee, error_functions.mee_gradient)
    hyper_parameters = [[('learning_rate', 0.1), ('PReLU_hp', 0.1)], [('learning_rate', 0.01), ('PReLU_hp', 0.2)],
                        [('learning_rate', 0.05), ('PReLU_hp', 0.1)], [('learning_rate', 0.2), ('PReLU_hp', 0.2)]]
    regularization_techniques = [("None", 0)]

    best_model = kfold.k_fold_cross_validation(training_input, training_output,
                                               [("structures", [[9, 10, 6, 2], [9, 4, 4, 2],
                                                                [9, 6, 6, 2], [9, 8, 8, 2],
                                                                [9, 10, 10, 2], [9, 10, 4, 2],
                                                                [9, 8, 6, 2], [9, 8, 4, 2]]),
                                                ("activation_functions",
                                                 activation_functions),
                                                ("error_functions",
                                                 [error_function]),
                                                ("hyper_parameters",
                                                 hyper_parameters),
                                                ("gradient_descent_techniques",
                                                 ["None", "NesterovM",
                                                  "RMSprop", "AdaGrad"]),
                                                ("mini_batch_sizes", [1, 2, 4]),
                                                ("regularization_techniques",
                                                 regularization_techniques)],
                                               4, False, "../MLcup_models.json", False, dt)

    performance = best_model.compute_performance(test_input, test_output)
    # print('performance on the test set: ' + str(performance))
