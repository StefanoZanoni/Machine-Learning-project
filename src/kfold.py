import sys
import numpy as np
import validation_utilities
from multiprocessing.pool import Pool
from timeit import default_timer as timer

from src import network, preprocessing
from src.validation_utilities import training
from src import utilities


# This function applies the k fold cross-validation technique to a list of combinations of hyperparameters.
# It takes:
#   - input data
#   - output data
#   - set of all hyperparameters
#   - boolean flag to determine if the search is randomized
#   - name of the file a where to dump
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
            # In case of regression, we want a low error. Lower is better
            if performance < best_performance:
                best_performance = performance
                best_hp = hp
                best_model = net

    # Dumps on file the best model found
    utilities.dump_on_json(best_performance, best_hp, filename, is_classification)

    # Prints the best accuracy/error found an on validation set with the list of the best hyperparameters
    if is_classification:
        print("Best accuracy on validation set: " + str(best_performance) + " | List of hyperparameters used: " + str(
            best_hp))
    else:
        print("Best error on validation set: " + str(best_performance) + " | List of hyperparameters used: " + str(
            best_hp))

    # Builds a new network with the best hyperparameters found
    model = network.Network(best_hp[0], best_hp[1], best_hp[2], best_hp[3], is_classification,
                            best_hp[6], best_hp[4])

    stop = timer()
    print('model selection in seconds: ' + str(np.ceil(stop - start)))

    # Sets the max epoch he can reach based on the maximum epoch reached by the best model
    max_epoch = best_model.epoch

    return model, max_epoch, best_hp[5], best_model.training_errors_means, best_model.validation_errors_means


# This function implements the k-fold validation technique.
# It takes:
#   - The input data set
#   - The output data set
#   - A combination of hyperparameters to try
#   - k, the factor of the k-fold
#   - a boolean flag to decide if it's a classification or regression technique
# The execution of the k training is parallelized
def cross_validation_inner(data_set, output_data_set, parameters, k, is_classification):
    # Computes the dimension of the validation set
    data_set_len = data_set.shape[0]
    proportions = int(np.ceil(data_set_len / k))
    parameters_and_data = []

    # Creates a Thread Pool to parallelize the k training process
    pool = Pool()

    # Builds the list of tasks to pass to the thread pool
    for validation_start in range(0, data_set_len, proportions):
        # Computes the end of the validation set in the input data set
        validation_end = validation_start + proportions

        # Takes the input and output data for the validation set
        validation_set = data_set[validation_start:validation_end]
        output_validation_set = output_data_set[validation_start:validation_end]

        # Takes the remaining input and output data for the training set
        training_set1 = data_set[:validation_start]
        training_set2 = data_set[validation_end:]
        training_set = np.concatenate((training_set1, training_set2))
        output_training_set = np.concatenate((output_data_set[:validation_start], output_data_set[validation_end:]))

        # Appends the task of hyperparameters to try and the data set
        parameters_and_data.append(parameters + [training_set, output_training_set, validation_set,
                                                 output_validation_set, proportions])

    # Sets the neutral result for successive comparisons
    if is_classification:
        best_performance = 0
    else:
        best_performance = sys.float_info.max

    performance_sum = 0
    # Gives to the thread pool the task and the list of data to apply to that task
    # Result is composed as follows:
    #   - Result[0] is the performance
    #   - Result[1] is the list of hyperparameters used
    #   - Result[2] is the model
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

    # At the end of the k trainings computes the mean over the performances
    performance_mean = performance_sum / k

    return performance_mean, best_model
