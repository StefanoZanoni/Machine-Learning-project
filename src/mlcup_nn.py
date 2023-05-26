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
                             (activation_functions.selu, activation_functions.selu_gradient),
                             (activation_functions.linear, activation_functions.linear_gradient)]
                            ]
    error_function = (error_functions.mee, error_functions.mee_gradient)
    hyper_parameters = [[('learning_rate', 0.005)]]
    regularization_techniques = [("L1", 0.001)]

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
    df2 = df.copy()
    df2.loc[:, 'id'] = df2['id'].apply(int)
    df2.to_csv('../output/ZDS_ML-CUP22-TS.csv', index=False)

