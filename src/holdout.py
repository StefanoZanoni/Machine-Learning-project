import sys
import numpy as np
import multiprocessing
from multiprocessing.pool import ThreadPool
from timeit import default_timer as timer

from src import preprocessing
from src import network
from src import validation_utilities


def holdout_selection(data_set, output_data_set, hyper_parameters_set, split_percentage, randomized_search, filename,
                      is_classification, dt):
    validation_set_len = int(np.ceil((100 - split_percentage) * data_set.shape[0] / 100))
    temp_data = np.array([[inp, out] for inp, out in zip(data_set, output_data_set)], dtype=dt)
    temp_data = preprocessing.shuffle_data(temp_data)
    validation_set = temp_data[:validation_set_len, 0]
    training_set = temp_data[validation_set_len:, 0]
    output_validation_set = temp_data[:validation_set_len, 1]
    output_training_set = temp_data[validation_set_len:, 1]

    hp = validation_utilities.get_hyper_parameters(hyper_parameters_set, randomized_search, is_classification)
    for i in range(len(hp)):
        hp[i] = hp[i] + [training_set, output_training_set, validation_set, output_validation_set]

    start = timer()
    best_model, mini_batch_size = search_best_model(hp, filename, is_classification)
    best_model.train(best_model.stop, data_set, output_data_set, mini_batch_size)
    stop = timer()
    print('model selection in seconds: ' + str(np.ceil(stop - start)))
    best_model.plot_learning_rate('green')

    return best_model


def holdout_selection_assessment(data_set, output_data_set, hyper_parameters_set, selection_split_percentage,
                                 training_split_percentage, randomized_search, filename, is_classification, dt):
    selection_set_len = int(np.ceil(selection_split_percentage * data_set.shape[0] / 100))
    temp_data = np.array([[inp, out] for inp, out in zip(data_set, output_data_set)], dtype=dt)
    temp_data = preprocessing.shuffle_data(temp_data)
    selection_set = temp_data[:selection_set_len, 0]
    test_set = temp_data[selection_set_len:, 0]
    output_selection_set = temp_data[:selection_set_len, 1]
    output_test_set = temp_data[selection_set_len:, 1]

    best_model = holdout_selection(selection_set, output_selection_set, hyper_parameters_set,
                                   training_split_percentage, randomized_search, filename, is_classification, dt)

    performance = best_model.compute_performance(test_set, output_test_set)

    return performance


def search_best_model(parameters, filename, is_classification):
    core_count = multiprocessing.cpu_count()
    pool = ThreadPool(processes=core_count)

    if is_classification:
        take_opposite = False
        max_accuracy_achieved = 0
        min_accuracy_achieved = 100
        best_hyper_parameters_found_max = []
        best_hyper_parameters_found_min = []
        for result in pool.map(training, parameters):
            accuracy = result[0]
            if accuracy > max_accuracy_achieved:
                max_accuracy_achieved = accuracy
                best_hyper_parameters_found_max = result[1]
                network_max = result[2]
            if accuracy < min_accuracy_achieved:
                min_accuracy_achieved = accuracy
                best_hyper_parameters_found_min = result[1]
                network_min = result[2]

        if (100 - min_accuracy_achieved) > max_accuracy_achieved:
            max_accuracy_achieved = (100 - min_accuracy_achieved)
            best_hyper_parameters_found_max = best_hyper_parameters_found_min
            network_max = network_min
            take_opposite = True

        best_network = network_max
        best_hyper_parameters_found = best_hyper_parameters_found_max
    else:
        error_min = sys.float_info.max
        best_hyper_parameters_found = []
        for result in pool.map(training, parameters):
            performance = result[0]
            if performance < error_min:
                error_min = performance
                best_hyper_parameters_found = result[1]
                best_network = result[2]

    best_network.plot_learning_rate('red')

    if is_classification:
        print("Best accuracy: " + str(max_accuracy_achieved) + " | List of hyperparameters used: " + str(
            best_hyper_parameters_found[:7]))
        validation_utilities.dump_on_json(max_accuracy_achieved, best_hyper_parameters_found, filename,
                                          is_classification)
    else:
        print("Best error: " + str(error_min) + " | List of hyperparameters used: " + str(
            best_hyper_parameters_found[:7]))
        validation_utilities.dump_on_json(error_min, best_hyper_parameters_found, filename,
                                          is_classification)

    nn_model = network.Network(best_hyper_parameters_found[0], best_hyper_parameters_found[1],
                               best_hyper_parameters_found[2], best_hyper_parameters_found[3],
                               is_classification, best_hyper_parameters_found[6], best_hyper_parameters_found[4])

    if is_classification:
        nn_model.take_opposite = take_opposite

    return nn_model, best_hyper_parameters_found[5]


def training(arguments):
    structure = arguments[0]
    activation_functions = arguments[1]
    error_function = arguments[2]
    hyper_parameters = arguments[3]
    gradient_descent_technique = arguments[4]
    mini_batch_size = arguments[5]
    regularization_technique = arguments[6]
    is_classification = arguments[7]
    training_set = arguments[8]
    output_training_set = arguments[9]
    validation_set = arguments[10]
    output_validation_set = arguments[11]

    net = network.Network(structure, activation_functions, error_function, hyper_parameters, is_classification,
                          regularization_technique,
                          gradient_descent_technique)
    net.train(net.stop, training_set, output_training_set, mini_batch_size)
    performance = net.compute_performance(validation_set, output_validation_set)

    return performance, arguments, net
