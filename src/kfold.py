import sys

import numpy as np
import validation_utilities
from multiprocessing.pool import Pool
from timeit import default_timer as timer
from src import network, preprocessing
from src.validation_utilities import training
from src import utilities


# This function implements the k fold cross validation technique. It takes:
#   - input data
#   - output data
#   - set of all hyperparameters
#   - boolean flag to determine if the search is randomized
#   - name of the file where to dump
#   - boolean flag to determine if is classification or regression task
def k_fold_cross_validation(data_set, output_data_set, hyper_parameters_set, k, randomized_search, filename,
                            is_classification):
    dt = object

    # Maps inputs to outputs
    temp_data = np.array([[inp, out] for inp, out in zip(data_set, output_data_set)], dtype=dt)
    # Shuffles the data
    temp_data = preprocessing.shuffle_data(temp_data)
    # Divides data between inputs and outputs
    data_set = temp_data[:, 0]
    output_data_set = temp_data[:, 1]

    # Gets combinations of hyperparameters in a list to try on the network
    hps = validation_utilities.get_hyper_parameters(hyper_parameters_set, randomized_search, is_classification)

    # Sets the neutral result for successive comparisons
    if is_classification:
        best_performance = 0
    else:
        best_performance = sys.float_info.max

    # Starts the timer to evaluate the time needed to do the model selection
    start = timer()

    # For every possible hyperparameters list does a k fold
    for hp in hps:
        # Applies the k fold and takes the performance achieved on the validation set
        performance, net = cross_validation_inner(data_set, output_data_set, hp, k, is_classification)

        # If the performances achieved on the validation set are better than the best performance achieved until
        # now, we update the value of best performance, the list of best hyperparameters and the best model found
        if is_classification:
            # In case of classification we want a better accuracy. Bigger is better
            if performance > best_performance:
                best_performance = performance
                best_hp = hp
                best_model = net
        else:
            # In case of regression we want a low error. Lower is better
            if performance < best_performance:
                best_performance = performance
                best_hp = hp
                best_model = net

    # Dumps on file the best model found
    utilities.dump_on_json(best_performance, best_hp, filename, is_classification)
    # Plots the learning rate
    best_model.plot_learning_rate()

    # Prints the best accuracy/error found on validation set with the list of the best hyperparameters
    if is_classification:
        print("Best accuracy on validation set: " + str(best_performance) + " | List of hyperparameters used: " + str(
            best_hp))
    else:
        print("Best error on validation set: " + str(best_performance) + " | List of hyperparameters used: " + str(
            best_hp))

    # We build a new network with the best hyperparameters found
    model = network.Network(best_hp[0], best_hp[1], best_hp[2], best_hp[3], is_classification,
                            best_hp[6], best_hp[4])
    max_epoch = best_model.epoch
    model.train(data_set, output_data_set, best_hp[5], model.stop, max_epoch)

    stop = timer()

    print('model selection in seconds: ' + str(np.ceil(stop - start)))

    return model


def cross_validation_inner(data_set, output_data_set, parameters, k, is_classification):
    data_set_len = data_set.shape[0]
    proportions = int(np.ceil(data_set_len / k))
    parameters_and_data = []

    pool = Pool()

    for validation_start in range(0, data_set_len, proportions):
        validation_end = validation_start + proportions
        validation_set = data_set[validation_start:validation_end]
        output_validation_set = output_data_set[validation_start:validation_end]
        training_set1 = data_set[:validation_start]
        training_set2 = data_set[validation_end:]
        training_set = np.concatenate((training_set1, training_set2))
        output_training_set = np.concatenate((output_data_set[:validation_start], output_data_set[validation_end:]))

        parameters_and_data.append(parameters + [training_set, output_training_set, validation_set,
                                                 output_validation_set, proportions])

    if is_classification:
        best_performance = 0
    else:
        best_performance = sys.float_info.max
    performance_sum = 0
    for result in pool.map(training, parameters_and_data):
        performance = result[0]
        performance_sum += performance
        if is_classification:
            if performance > best_performance:
                best_performance = performance
                best_model = result[2]
        else:
            if performance < best_performance:
                best_performance = performance
                best_model = result[2]

    performance_mean = performance_sum / k

    return performance_mean, best_model
