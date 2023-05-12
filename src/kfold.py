import sys

import numpy as np
import validation_utilities
from multiprocessing.pool import Pool
from timeit import default_timer as timer
from src import network, preprocessing
from src.validation_utilities import training


def k_fold_cross_validation(data_set, output_data_set, hyper_parameters_set, k, randomized_search, filename,
                            is_classification, dt):
    temp_data = np.array([[inp, out] for inp, out in zip(data_set, output_data_set)], dtype=dt)
    temp_data = preprocessing.shuffle_data(temp_data)
    data_set = temp_data[:, 0]
    output_data_set = temp_data[:, 1]

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

    validation_utilities.dump_on_json(best_performance, best_hp, filename, is_classification)
    best_model.plot_learning_rate()
    if is_classification:
        print("Best accuracy on validation set: " + str(best_performance) + " | List of hyperparameters used: " + str(
            best_hp))
    else:
        print("Best error on validation set: " + str(best_performance) + " | List of hyperparameters used: " + str(
            best_hp))

    model = network.Network(best_hp[0], best_hp[1], best_hp[2], best_hp[3], is_classification,
                            best_hp[6], best_hp[4])
    max_epoch = best_model.epoch
    model.train(data_set, output_data_set, best_hp[5], model.stop, max_epoch)

    stop = timer()

    print('model selection in seconds: ' + str(np.ceil(stop - start)))

    return model


def cross_validation_inner(data_set, output_data_set, parameters, k):
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

    best_performance = 0
    performance_sum = 0
    for result in pool.map(training, parameters_and_data):
        performance = result[0]
        performance_sum += performance
        if performance > best_performance:
            best_performance = performance
            best_model = result[2]

    performance_mean = performance_sum / k

    return performance_mean, best_model
