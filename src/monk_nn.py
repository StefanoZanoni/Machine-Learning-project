import numpy as np
import pandas as pd

from src import activation_functions
from src import error_functions
from src import preprocessing
from src import holdout


def read_data_set(problem):
    training_df = pd.read_csv("../Monks_problem/monks-" + str(problem) + ".train", index_col=False, sep=" ",
                              names=["", "output", "a1", "a2", "a3", "a4", "a5", "a6", "class"])
    testing_df = pd.read_csv("../Monks_problem/monks-" + str(problem) + ".test", index_col=False, sep=" ",
                             names=["", "output", "a1", "a2", "a3", "a4", "a5", "a6", "class"])
    training_df = training_df.dropna(axis=1)
    testing_df = testing_df.dropna(axis=1)

    training_output = np.array(training_df["output"])
    testing_output = np.array(testing_df["output"])

    training_input = np.array(training_df[["a1", "a2", "a3", "a4", "a5", "a6"]])
    testing_input = np.array(testing_df[["a1", "a2", "a3", "a4", "a5", "a6"]])

    return training_input, training_output, testing_input, testing_output


if __name__ == '__main__':

    dt = object

    # MONK1
    data = read_data_set(1)
    training_input1 = data[0]
    training_output1 = data[1]
    testing_input1 = data[2]
    testing_output1 = data[3]

    activation_functions1 = [[(activation_functions.tanh, activation_functions.tanh_gradient),
                              (activation_functions.tanh, activation_functions.tanh_gradient),
                              (activation_functions.sigmoid, activation_functions.sigmoid_gradient)]]
    error_function1 = (error_functions.bce, error_functions.bce_gradient)
    hyper_parameters = [[('learning_rate', 0.1)], [('learning_rate', 0.2)]]
    regularization_techniques1 = [("None", 0)]

    training_input1 = preprocessing.one_hot_encoding(training_input1)
    testing_input1 = preprocessing.one_hot_encoding(testing_input1)

    optimal_model = holdout.holdout_selection(training_input1, training_output1, [("structures",
                                                                                   [[17, 4, 4, 1]]),
                                                                                  ("activation_functions",
                                                                                   activation_functions1),
                                                                                  ("error_functions",
                                                                                   [error_function1]),
                                                                                  ("hyper_parameters",
                                                                                   hyper_parameters),
                                                                                  ("gradient_descend_techniques",
                                                                                   ["None", "NesterovM"]),
                                                                                  ("mini_batch_sizes",
                                                                                   [1, 2, 4, 8, 16, 32, 64]),
                                                                                  ("regularization_techniques",
                                                                                   regularization_techniques1)],
                                              70, False, "../Monk1_models.json", True, dt)

    performance = optimal_model.compute_performance(testing_input1, testing_output1)
    print('performance on the test set: ' + str(performance))

    # # MONK2
    # data = read_data_set(2)
    # training_input2 = data[0]
    # training_output2 = data[1]
    # testing_input2 = data[2]
    # testing_output2 = data[3]
    #
    # activation_functions2 = [[(activation_functions.relu, activation_functions.relu_gradient),
    #                           (activation_functions.sigmoid, activation_functions.sigmoid_gradient)]]
    # error_function2 = (error_functions.bce, error_functions.bce_gradient)
    # hyper_parameters2 = [[('learning_rate', 0.3)]]
    # regularization_techniques2 = [("None", 0)]
    #
    # training_input2 = preprocessing.one_hot_encoding(training_input2)
    # testing_input2 = preprocessing.one_hot_encoding(testing_input2)
    #
    # optimal_model = holdout.holdout_selection(training_input2, training_output2, [("structures", [[17, 2, 1]]),
    #                                                                               (
    #                                                                                   "activation_functions",
    #                                                                                   activation_functions2),
    #                                                                               ("error_functions", [error_function2]),
    #                                                                               ("hyper_parameters", hyper_parameters2),
    #                                                                               (
    #                                                                                   "gradient_descend_techniques",
    #                                                                                   ["NesterovM"]),
    #                                                                               ("mini_batch_sizes", [4]),
    #                                                                               ("regularization_techniques",
    #                                                                                regularization_techniques2)],
    #                                           70, False, "../Monk2_models.json", True, dt)
    #
    # performance = optimal_model.compute_performance(testing_input2, testing_output2)
    # print('performance on the test set: ' + str(performance))
    #
    # # MONK3
    # data = read_data_set(3)
    # training_input3 = data[0]
    # training_output3 = data[1]
    # testing_input3 = data[2]
    # testing_output3 = data[3]
    #
    # activation_functions3 = [[(activation_functions.relu, activation_functions.relu_gradient),
    #                           (activation_functions.sigmoid, activation_functions.sigmoid_gradient)]]
    # error_function3 = (error_functions.bce, error_functions.bce_gradient)
    # hyper_parameters3 = [[('learning_rate', 0.1)]]
    # regularization_techniques3 = [("None", 0)]
    #
    # training_input3 = preprocessing.one_hot_encoding(training_input3)
    # testing_input3 = preprocessing.one_hot_encoding(testing_input3)
    #
    # optimal_model = holdout.holdout_selection(training_input3, training_output3, [("structures", [[17, 5, 1]]),
    #                                                                               (
    #                                                                                   "activation_functions",
    #                                                                                   activation_functions3),
    #                                                                               ("error_functions", [error_function3]),
    #                                                                               ("hyper_parameters", hyper_parameters3),
    #                                                                               (
    #                                                                                   "gradient_descend_techniques",
    #                                                                                   ["NesterovM"]),
    #                                                                               ("mini_batch_sizes", [2]),
    #                                                                               ("regularization_techniques",
    #                                                                                regularization_techniques3)],
    #                                           70, False, "../Monk3_models.json", True, dt)
    #
    # performance = optimal_model.compute_performance(testing_input3, testing_output3)
    # # print('performance on the test set: ' + str(performance))
