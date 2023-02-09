import numpy as np
import validation_utilities
from src.holdout import training


def k_fold_cross_validation(data_set, output_data_set, hyper_parameters_set, k, randomized_search, filename,
                            is_classification):
    hps = validation_utilities.get_hyper_parameters(hyper_parameters_set, randomized_search, is_classification)

    # grid search over k
    best_performance = 0
    best_hp = []
    for hp in hps:
        performance, network = cross_validation_inner(data_set, output_data_set, hp, k)
        if performance > best_performance:
            best_performance = performance
            best_hp = hp
            best_network = network

    validation_utilities.dump_on_json(performance, best_hp, filename, is_classification)
    best_network.plot_learning_rate()
    if is_classification:
        print("Best accuracy: " + str(best_performance) + " | List of hyperparameters used: " + str(best_hp))
    else:
        print("Best error: " + str(best_performance) + " | List of hyperparameters used: " + str(best_hp))

    net = network.Network(best_hp[0], best_hp[1], best_hp[2], best_hp[3], is_classification, best_hp[6], best_hp[4])
    net.train(net.stop(), data_set, output_data_set, best_hp[5])

    return net


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

        performance, arguments, network = training(hyperparameters)

        performance_sum += performance

    performance = performance_sum / k

    return performance, network
