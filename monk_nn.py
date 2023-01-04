from timeit import default_timer as timer

import numpy as np
import pandas as pd
import activationfunctions as af
import errorfunctions as ef
import network as nt

np.random.seed(0)

for i in range(1, 4):
    # read text file into pandas DataFrame
    training_df = pd.read_csv("./Monks_problem/monks-" + str(i) + ".train", index_col=False, sep=" ",
                              names=["", "output", "a1", "a2", "a3", "a4", "a5", "a6", "class"])
    testing_df = pd.read_csv("./Monks_problem/monks-" + str(i) + ".test", index_col=False, sep=" ",
                             names=["", "output", "a1", "a2", "a3", "a4", "a5", "a6", "class"])
    training_df = training_df.dropna(axis=1)
    testing_df = testing_df.dropna(axis=1)

    training_output = training_df["output"].array
    testing_output = testing_df["output"].array

    training_input = training_df[["a1", "a2", "a3", "a4", "a5", "a6"]]
    testing_input = testing_df[["a1", "a2", "a3", "a4", "a5", "a6"]]

    activation_functions = [(af.leaky_relu, af.leaky_relu_gradient), (af.sigmoid, af.sigmoid_gradient)]
    error_function = (ef.bce, ef.bce_derivative)
    hyper_parameters = [('leaky_relu_hp', 0.1), ('learning_rate', 0.01), ('huber_loss_hp', 0.1)]

    network = nt.Network([6, 4, 1], activation_functions, error_function, hyper_parameters)
    start = timer()
    network.train(network.stop, training_input, training_output, len(training_output))
    print("training time in seconds:", np.ceil(timer() - start))
    network.plot_learning_rate(i)