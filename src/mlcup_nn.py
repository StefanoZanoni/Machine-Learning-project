import numpy as np
import pandas as pd

from src import activation_functions
from src import error_functions
from src import kfold
from src import utilities
from src import network

# ML CUP
if __name__ == '__main__':
    # Reads the dataset
    input_data, output_data, blind_testing_input = utilities.read_ml_cup_data_set()

    # Splits the input and output data in training input/output and testing input/output data
    training_input, training_output, test_input, test_output = utilities.split_input_data(input_data, output_data, 80)

    # Defines the list of activation functions to try
    # Each element of the list is a list that contains the activation functions for the model.
    # Each element of the internal list contains tuples that are assigned to each layer of the model.
    # A single tuple is of the form (function, gradient of that function)
    activation_functions = [
        [(activation_functions.tanh, activation_functions.tanh_gradient),
         (activation_functions.tanh, activation_functions.tanh_gradient),
         (activation_functions.selu, activation_functions.selu_gradient),
         (activation_functions.linear, activation_functions.linear_gradient)]
    ]

    # Defines the tuple for the error function. The form of the tuple is
    # (error function, gradient of that error function)
    error_function = (error_functions.mee, error_functions.mee_gradient)

    # Defines the list of hyperparameters to try
    # Each element of the list is a list that contains a tuple (or more than one if we are using activation functions
    # that requires an additional parameters) with the name of the parameter and the value to try
    hyper_parameters = [[('learning_rate', 0.002)]]

    # Define a list of regularization to try. Each element of the list is a tuple that contains the name of the
    # regularization technique and then its value.
    # If there's no need to try a regularization technique, it is sufficient to add the tuple ("None", 0)
    regularization_techniques = [('None', 0)]

    # Model selection through k-fold technique
    # K = 128
    # The first False let us do an exhaustive search through all possible combinations of hyperparameters
    # The second False let us define that it's a regression problem
    best_model, max_epoch, best_mini_batch, training_error_means, validation_error_means = \
        kfold.k_fold_cross_validation(training_input, training_output,
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
                                      128, False, "../MLcup_models.json", False)

    # Retrains on the whole training set
    best_model.train(training_input, training_output, best_mini_batch, best_model.stop, max_epoch, test_input,
                     test_output)
    best_model.validation_errors_means = validation_error_means
    best_model.training_errors_means = training_error_means
    # Plots the learning rate
    best_model.plot_learning_rate()

    # With the optimal model found in the k-fold selection, we compute the performance on the testing data
    performance = best_model.compute_performance(test_input, test_output)
    print('best performance on the test set: ' + str(performance))

    final_model = network.Network(best_model.structure, best_model.activation_functions, best_model.error_function,
                                  best_model.hyper_parameters, best_model.is_classification, best_model.regularization,
                                  "None")
    final_model.W = best_model.initialized_weights
    # re-train the model over all the data before predicting on the blind test set
    final_model.train(input_data, output_data, best_mini_batch, final_model.stop, max_epoch)

    output_x = []
    output_y = []
    # Computes the predictions on the blind test set
    for input in blind_testing_input:
        # Computes the prediction on 'input'
        output = best_model.forward(input)

        # Saves the two results since it has two outputs
        output_x.append(output[0][0])
        output_y.append(output[1][0])

    # Generates a progressive id for each row
    blind_len = blind_testing_input.shape[0]
    ids = [i for i in range(1, blind_len + 1)]

    # Creates the Pandas DataFrame
    df = pd.DataFrame(np.array([ids, output_x, output_y]).T, columns=['id', 'output_x', 'output_y'])
    # Converts the first column, the ids, in integers
    df[df.columns[0]] = df[df.columns[0]].astype(int)

    # Writes the dataframe to a .csv file
    df.to_csv('../output/ZDS_ML-CUP22-TS.csv', index=False)
