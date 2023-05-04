import sys

import numpy as np
import validation_utilities
from multiprocessing.pool import Pool
from timeit import default_timer as timer
from src import network, preprocessing
from src.validation_utilities import training


def k_fold_cross_validation(data_set, output_data_set, hyper_parameters_set, k, randomized_search, filename,
                            is_classification):
    hps = validation_utilities.get_hyper_parameters(hyper_parameters_set, randomized_search, is_classification)

    if is_classification:
        best_performance = 0
    else:
        best_performance = sys.float_info.max

    start = timer()

    # grid search over k
    for hp in hps:
        performance, net = cross_validation_inner(data_set, output_data_set, hp, k)
        if is_classification:
            if performance > best_performance:
                best_performance = performance
                best_hp = hp
                best_model = net
        else:
            if performance < best_performance:
                best_performance = performance
                best_hp = hp
                best_model = net

    validation_utilities.dump_on_json(performance, best_hp, filename, is_classification)
    best_network.plot_learning_rate()
    if is_classification:
        print("Best accuracy: " + str(best_performance) + " | List of hyperparameters used: " + str(best_hp))
    else:
        print("Best error: " + str(best_performance) + " | List of hyperparameters used: " + str(best_hp))

    best_model = network.Network(best_hp[0], best_hp[1], best_hp[2], best_hp[3], is_classification,
                                 best_hp[6], best_hp[4])
    best_model.train(data_set, output_data_set, best_hp[5], best_model.stop, best_model.epoch)

    stop = timer()

    print('model selection in seconds: ' + str(np.ceil(stop - start)))

    return best_model


def cross_validation_inner(data_set, output_data_set, parameters, k):
    data_set_len = data_set.shape[0]
    proportions = int(np.ceil(data_set_len / k))
    performance_sum = 0
    for validation_start in range(0, data_set_len, proportions):
        validation_end = validation_start + proportions
        validation_set = data_set[validation_start:validation_end]
        output_validation_set = output_data_set[validation_start:validation_end]
        training_set1 = data_set[:validation_start]
        training_set2 = data_set[validation_end:]
        training_set = np.concatenate((training_set1, training_set2))
        output_training_set = np.concatenate((output_data_set[:validation_start], output_data_set[validation_end:]))

        hyperparameters = parameters + [training_set, output_training_set, validation_set, output_validation_set,
                                        proportions]

    best_performance = 0
    performance_sum = 0
    for result in pool.map(training, parameters_and_data):
        performance = result[0]
        performance_sum += performance
        if performance > best_performance:
            best_performance = performance
            best_model = result[2]

    performance_mean = performance_sum / k

    return performance, network
