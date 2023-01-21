import numpy as np
import pandas as pd
import activation_functions as af
import error_functions as ef
import preprocessing as pp

from src import holdout


np.random.seed(0)
# (input, output) type
dt = object

training_df = pd.read_csv('../MLcup_ploblem/ML-CUP22-TR.csv', names=['id', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'o1', 'o2'])
training_df = training_df.dropna(axis=0)
blind_testing_df = pd.read_csv('../MLcup_ploblem/ML-CUP22-TS.csv', names=['id', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9'])
blind_testing_df = blind_testing_df.dropna(axis=0)

training_input = np.array(training_df[['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9']])
training_output = np.array(training_df[['o1', 'o2']])

blind_testing_input = np.array(blind_testing_df[['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9']])

activation_functions = [[(af.leaky_relu, af.leaky_relu_gradient), (af.leaky_relu, af.leaky_relu_gradient),
                         (af.relu, af.relu_gradient)]]
error_function = (ef.mse, ef.mse_derivative)
hyper_parameters = [[('learning_rate', 0.1), ('leaky_hp', 0.1)]]
regularization_techniques = [("None", 0)]

performance = holdout.holdout_selection_assessment(training_input, training_output, [("structures", [[9, 30, 10, 2]]),
                                                                              ("activation_functions", activation_functions),
                                                                              ("error_functions", [error_function]),
                                                                              ("hyper_parameters", hyper_parameters),
                                                                              ("gradient_descent_techniques", ["NesteroVM"]),
                                                                              ("mini_batch_sizes", [1]),
                                                                              ("regularization_techniques", regularization_techniques)],
                                                                               70, 70, False, "../MLcup_models.json", False, dt)
