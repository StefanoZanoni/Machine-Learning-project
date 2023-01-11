import numpy as np
import pandas as pd
import activationfunctions as af
import errorfunctions as ef
import preprocessing as pp
import validationtechniques as vt


# np.random.seed(0)

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


# MONK1
data = read_data_set(1)
training_input1 = data[0]
training_output1 = data[1]
testing_input1 = data[2]
testing_output1 = data[3]

activation_functions = [[(af.leaky_relu, af.leaky_relu_gradient), (af.sigmoid, af.sigmoid_gradient)],
                        [(af.relu, af.relu_gradient), (af.sigmoid, af.sigmoid_gradient)],
                        [(af.linear, af.linear_gradient), (af.sigmoid, af.sigmoid_gradient)]]
error_function1 = (ef.bce, ef.bce_derivative)
hyper_parameters = [[('leaky_relu_hp', 0.1), ('learning_rate', 0.1), ('huber_loss_hp', 0.1)],
                    [('leaky_relu_hp', 0.2), ('learning_rate', 0.01), ('huber_loss_hp', 0.01)],
                    [('leaky_relu_hp', 0.25), ('learning_rate', 0.001), ('huber_loss_hp', 0.01)],
                    [('leaky_relu_hp', 0.3), ('learning_rate', 0.0001), ('huber_loss_hp', 0.01)]]

training_input1 = pp.min_max_scaling(training_input1)
vt.holdout_validation(training_input1, training_output1, [("structures", [[6, 4, 1], [6, 2, 1], [6, 3, 1], [6, 1, 1]]),
                                                          ("activation_functions", activation_functions),
                                                          ("error_functions", [error_function1]),
                                                          ("hyper_parameters", hyper_parameters),
                                                          ("gradient_descend_techniques",
                                                           ["AdaGrad", "RMSProp", "SGD"]),
                                                          ("mini_batch_sizes", [1, 85, 5, 17])], 70, True, "../Monk1_models.txt")

# # MONK2
# data = read_data_set(2)
# training_input2 = data[0]
# training_output2 = data[1]
# testing_input2 = data[2]
# testing_output2 = data[3]
#
# activation_functions2 = [(af.leaky_relu, af.leaky_relu_gradient), (af.sigmoid, af.sigmoid_gradient)]
# error_function2 = (ef.bce, ef.bce_derivative)
# hyper_parameters2 = [('leaky_relu_hp', 0.1), ('learning_rate', 0.01), ('huber_loss_hp', 0.1)]
#
# training_input2 = pp.z_score_scaling(training_input2)
#
# # MONK3
# data = read_data_set(3)
# training_input3 = data[0]
# training_output3 = data[1]
# testing_input3 = data[2]
# testing_output3 = data[3]
#
# activation_functions3 = [(af.leaky_relu, af.leaky_relu_gradient), (af.sigmoid, af.sigmoid_gradient)]
# error_function3 = (ef.bce, ef.bce_derivative)
# hyper_parameters3 = [('leaky_relu_hp', 0.1), ('learning_rate', 0.01), ('huber_loss_hp', 0.1)]
#
# training_input3 = pp.z_score_scaling(training_input3)
