from random import random

from src import network


# Function that creates the network with the specified parameters.
# Training on the training data, evaluation of performances on validation data.
# Returns performance, parameters of the network and the network built
def training(arguments):
    print(f"training model: {arguments[:7]}\n...\n")

    # Retrieving arguments to build the network
    structure = arguments[0]
    activation_functions = arguments[1]
    error_function = arguments[2]
    hyper_parameters = arguments[3]
    gradient_descent_technique = arguments[4]
    mini_batch_size = arguments[5]
    regularization_technique = arguments[6]
    is_classification = arguments[7]

    # Retrieving training and validation data
    training_set = arguments[8]
    output_training_set = arguments[9]
    validation_set = arguments[10]
    output_validation_set = arguments[11]

    # Creation of the network
    net = network.Network(structure, activation_functions, error_function, hyper_parameters, is_classification,
                          regularization_technique,
                          gradient_descent_technique)

    # Training of the model and then evaluation on validation data
    net.train(training_set, output_training_set, mini_batch_size, net.early_stopping,
              validation_set, output_validation_set)

    # Retrieving performance of the network on the validation set
    performance = net.best_validation_errors_means

    return performance, arguments, net


# Creates a list of sets of hyperparameters to try to build the network
def get_hyper_parameters(hyper_parameters_set, randomized_search, is_classification):
    # Retrieving arguments to all the possible hyperparameters
    structures = hyper_parameters_set[0][1]
    activation_functions_list = hyper_parameters_set[1][1]
    error_functions = hyper_parameters_set[2][1]
    hyper_parameters_list = hyper_parameters_set[3][1]
    gradient_descent_techniques = hyper_parameters_set[4][1]
    mini_batch_sizes = hyper_parameters_set[5][1]
    regularization_techniques = hyper_parameters_set[6][1]

    if randomized_search:
        hps = randomized_grid_search(structures, activation_functions_list, error_functions, hyper_parameters_list,
                                     gradient_descent_techniques, mini_batch_sizes, regularization_techniques,
                                     is_classification)
    else:
        hps = exhaustive_grid_search(structures, activation_functions_list, error_functions, hyper_parameters_list,
                                     gradient_descent_techniques, mini_batch_sizes, regularization_techniques,
                                     is_classification)
    return hps


# Given a set of arguments to build the network on,
# returns 20 combinations of hyperparameters randomly chosen
def randomized_grid_search(structures, activation_functions_list, error_functions, hyper_parameters_list,
                           gradient_descent_techniques, mini_batch_sizes, regularization_techniques, is_classification):
    # List of combinations of hyperparameters
    hps = []

    for i in range(20):
        structure = structures[random.randint(0, len(structures) - 1)]

        activation_functions = activation_functions_list[random.randint(0, len(activation_functions_list) - 1)]

        error_function = error_functions[random.randint(0, len(error_functions) - 1)]

        hyper_parameters = hyper_parameters_list[random.randint(0, len(hyper_parameters_list) - 1)]

        gradient_descent_technique = gradient_descent_techniques[
            random.randint(0, len(gradient_descent_techniques) - 1)]

        mini_batch_size = mini_batch_sizes[random.randint(0, len(mini_batch_sizes) - 1)]

        regularization_technique = regularization_techniques[random.randint(0, len(regularization_techniques) - 1)]

        # Append the new combination to the list of combinations chosen
        hps.append([structure, activation_functions, error_function, hyper_parameters, gradient_descent_technique,
                    mini_batch_size, regularization_technique, is_classification])

    return hps


# Given a set of arguments to build the network on,
# returns all the possible combinations of hyperparameters
def exhaustive_grid_search(structures, activation_functions_list, error_functions, hyper_parameters_list,
                           gradient_descent_techniques, mini_batch_sizes, regularization_techniques, is_classification):
    # List of possible combinations of hyperparameters
    hps = []

    for structure in structures:
        for activation_functions in activation_functions_list:
            for error_function in error_functions:
                for hyper_parameters in hyper_parameters_list:
                    for gradient_descent_technique in gradient_descent_techniques:
                        for mini_batch_size in mini_batch_sizes:
                            for regularization_technique in regularization_techniques:
                                hps.append([structure, activation_functions, error_function, hyper_parameters,
                                            gradient_descent_technique, mini_batch_size, regularization_technique,
                                            is_classification])
    return hps
