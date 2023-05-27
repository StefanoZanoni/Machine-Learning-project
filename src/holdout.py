import sys
import numpy as np
from multiprocessing.pool import Pool
from timeit import default_timer as timer

from src import preprocessing
from src import network
from src import validation_utilities
from src.validation_utilities import training
from src import utilities


def holdout_selection(data_set, output_data_set, hyper_parameters_set, split_percentage, randomized_search, filename,
                      is_classification):
    dt = object

    # compute validation set length
    validation_set_len = int(np.ceil((100 - split_percentage) * data_set.shape[0] / 100))
    # create an array with input and output coupled
    temp_data = np.array([[inp, out] for inp, out in zip(data_set, output_data_set)], dtype=dt)
    # shuffle all the data
    temp_data = preprocessing.shuffle_data(temp_data)
    # create the validation input set
    validation_set = temp_data[:validation_set_len, 0]
    # create the training input set
    training_set = temp_data[validation_set_len:, 0]
    # create the validation output set
    output_validation_set = temp_data[:validation_set_len, 1]
    # create the training output set
    output_training_set = temp_data[validation_set_len:, 1]

    # get all combinations of hyperparameters to test
    hps = validation_utilities.get_hyper_parameters(hyper_parameters_set, randomized_search, is_classification)
    # attach to each hyperparameter combination the training and validation set
    for i in range(len(hps)):
        hps[i] = hps[i] + [training_set, output_training_set, validation_set, output_validation_set]

    start = timer()

    # search the best model over all the hyperparameter configurations
    best_model, mini_batch_size, max_epoch = search_best_model(hps, filename, is_classification)
    # retrain the best model over the whole dataset
    best_model.train(temp_data[:, 0], temp_data[:, 1], mini_batch_size, best_model.stop, max_epoch)

    stop = timer()

    print('model selection in seconds: ' + str(np.ceil(stop - start)))

    return best_model


def holdout_selection_assessment(data_set, output_data_set, hyper_parameters_set, selection_split_percentage,
                                 training_split_percentage, randomized_search, filename, is_classification, dt):
    # compute selection set length
    selection_set_len = int(np.ceil(selection_split_percentage * data_set.shape[0] / 100))
    # create an array with input and output coupled
    temp_data = np.array([[inp, out] for inp, out in zip(data_set, output_data_set)], dtype=dt)
    # shuffle all the data
    temp_data = preprocessing.shuffle_data(temp_data)
    # create the selection input set
    selection_set = temp_data[:selection_set_len, 0]
    # create the test input set
    test_set = temp_data[selection_set_len:, 0]
    # create the selection output set
    output_selection_set = temp_data[:selection_set_len, 1]
    # create the test output set
    output_test_set = temp_data[selection_set_len:, 1]

    # compute holdout selection on the selection set
    best_model = holdout_selection(selection_set, output_selection_set, hyper_parameters_set,
                                   training_split_percentage, randomized_search, filename, is_classification, dt)

    # compute the performance on the test set
    performance = best_model.compute_performance(test_set, output_test_set)

    return performance


def search_best_model(parameters, filename, is_classification):

    # create a process pool
    pool = Pool()

    if is_classification:
        take_opposite = False
        max_accuracy_achieved = 0
        min_accuracy_achieved = 100
        best_hyper_parameters_found_max = []
        best_hyper_parameters_found_min = []

        # compute, in parallel, the training phase with a different hyperparameter combination
        # with the processes of the pool
        for result in pool.map(training, parameters):
            accuracy = result[0]
            # update the max accuracy found so far
            if accuracy > max_accuracy_achieved:
                max_accuracy_achieved = accuracy
                best_hyper_parameters_found_max = result[1]
                network_max = result[2]
            # update the min accuracy found so far
            if accuracy < min_accuracy_achieved:
                min_accuracy_achieved = accuracy
                best_hyper_parameters_found_min = result[1]
                network_min = result[2]

        # check if it is convenient to consider max accuracy or 100 - min accuracy
        if (100 - min_accuracy_achieved) > max_accuracy_achieved:
            max_accuracy_achieved = (100 - min_accuracy_achieved)
            best_hyper_parameters_found_max = best_hyper_parameters_found_min
            network_max = network_min
            take_opposite = True

        # update the best model and hyperparameters found
        best_network = network_max
        best_hyper_parameters_found = best_hyper_parameters_found_max
    else:
        error_min = sys.float_info.max
        best_hyper_parameters_found = []

        for result in pool.map(training, parameters):
            performance = result[0]
            # update the best performance and network found so far
            if performance < error_min:
                error_min = performance
                best_hyper_parameters_found = result[1]
                best_network = result[2]

    best_network.plot_learning_rate()

    if is_classification:
        print("Best accuracy on the validation set: " + str(max_accuracy_achieved) +
              " | List of hyperparameters used: " + str(best_hyper_parameters_found[:7]))
        utilities.dump_on_json(max_accuracy_achieved, best_hyper_parameters_found, filename, is_classification)
    else:
        print("Best error on the validation set: " + str(error_min) + " | List of hyperparameters used: " +
              str(best_hyper_parameters_found[:7]))
        utilities.dump_on_json(error_min, best_hyper_parameters_found, filename, is_classification)

    # build a new model with the best hyperparameters found
    nn_model = network.Network(best_hyper_parameters_found[0], best_hyper_parameters_found[1],
                               best_hyper_parameters_found[2], best_hyper_parameters_found[3],
                               is_classification, best_hyper_parameters_found[6], best_hyper_parameters_found[4])

    if is_classification:
        nn_model.take_opposite = take_opposite

    return nn_model, best_hyper_parameters_found[5], best_network.epoch
