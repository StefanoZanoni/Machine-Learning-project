import random
from multiprocessing.pool import ThreadPool
from typing import Type

import numpy as np
import network as nt
import multiprocessing
import json
import os

from src import preprocessing


def threaded_training(arguments):
    structure = arguments[0]
    activation_functions = arguments[1]
    error_function = arguments[2]
    hyper_parameters = arguments[3]
    gradient_descent_technique = arguments[4]
    mini_batch_size = arguments[5]
    regularization_technique = arguments[6]
    training_set = arguments[7]
    output_training_set = arguments[8]
    validation_set = arguments[9]
    output_validation_set = arguments[10]
    validation_set_len = arguments[11]

    network = nt.Network(structure, activation_functions, error_function, hyper_parameters, regularization_technique,
                         gradient_descent_technique)
    network.train(network.stop, training_set, output_training_set, mini_batch_size)
    # network.plot_learning_rate()

    correct_prevision = 0
    for x, y in zip(validation_set, output_validation_set):
        predicted_output = network.forward(x)
        if predicted_output == y:
            correct_prevision += 1

    accuracy = correct_prevision * 100 / validation_set_len
    # print(f'Accuracy: {accuracy:.2f}%\n')

    return accuracy, arguments, network


def randomized_grid_search(structures, activation_functions_list, error_functions, hyper_parameters_list,
                           gradient_descent_techniques, mini_batch_sizes, regularization_techniques):
    hp = []
    for i in range(20):
        structure = structures[random.randint(0, len(structures) - 1)]
        activation_functions = activation_functions_list[random.randint(0, len(activation_functions_list) - 1)]
        error_function = error_functions[random.randint(0, len(error_functions) - 1)]
        hyper_parameters = hyper_parameters_list[random.randint(0, len(hyper_parameters_list) - 1)]
        gradient_descent_technique = gradient_descent_techniques[
            random.randint(0, len(gradient_descent_techniques) - 1)]
        mini_batch_size = mini_batch_sizes[random.randint(0, len(mini_batch_sizes) - 1)]
        regularization_technique = regularization_techniques[random.randint(0, len(regularization_techniques) - 1)]

        hp.append([structure, activation_functions, error_function, hyper_parameters, gradient_descent_technique,
                   mini_batch_size, regularization_technique])
    return hp


def exhaustive_grid_search(structures, activation_functions_list, error_functions, hyper_parameters_list,
                           gradient_descent_techniques, mini_batch_sizes, regularization_techniques):
    hp = []
    for structure in structures:
        for activation_functions in activation_functions_list:
            for error_function in error_functions:
                for hyper_parameters in hyper_parameters_list:
                    for gradient_descent_technique in gradient_descent_techniques:
                        for mini_batch_size in mini_batch_sizes:
                            for regularization_technique in regularization_techniques:
                                hp.append([structure, activation_functions, error_function, hyper_parameters,
                                           gradient_descent_technique, mini_batch_size, regularization_technique])
    return hp


def holdout_validation(data_set, output_data_set, hyper_parameters_set, split_percentage, randomized_search, filename):
    validation_set_len = int(np.ceil((100 - split_percentage) * data_set.shape[0] / 100))
    dt = np.dtype(np.ndarray, Type[int])
    temp_data = np.array([(inp, out) for inp, out in zip(data_set, output_data_set)], dtype=dt)
    temp_data = preprocessing.shuffle_data(temp_data)
    validation_set = temp_data[:validation_set_len, 0]
    training_set = temp_data[validation_set_len:, 0]
    output_validation_set = temp_data[:validation_set_len, 1]
    output_training_set = temp_data[validation_set_len:, 1]

    hp = get_hyper_parameters(hyper_parameters_set, randomized_search)
    for h in hp:
        h = h + [training_set, output_training_set, validation_set, output_validation_set, validation_set_len]

    best_model, mini_batch_size = search_best_model(hp, filename)
    best_model.train(best_model.stop, data_set, output_data_set, mini_batch_size)


def get_hyper_parameters(hyper_parameters_set, randomized_search):
    # [(structures, [[s1], [s2]]), (af, [[lr, sg], [r, sg]]), (ef, []), (hp, [(lr, []), (), ()]), (gdt, [""]), (batch, [])]
    structures = hyper_parameters_set[0][1]
    activation_functions_list = hyper_parameters_set[1][1]
    error_functions = hyper_parameters_set[2][1]
    hyper_parameters_list = hyper_parameters_set[3][1]
    gradient_descent_techniques = hyper_parameters_set[4][1]
    mini_batch_sizes = hyper_parameters_set[5][1]
    regularization_techniques = hyper_parameters_set[6][1]

    if randomized_search:
        hp = randomized_grid_search(structures, activation_functions_list, error_functions, hyper_parameters_list,
                                    gradient_descent_techniques, mini_batch_sizes, regularization_techniques)
    else:
        hp = exhaustive_grid_search(structures, activation_functions_list, error_functions, hyper_parameters_list,
                                    gradient_descent_techniques, mini_batch_sizes, regularization_techniques)
    return hp


def search_best_model(parameters, filename):
    core_count = multiprocessing.cpu_count()
    pool = ThreadPool(processes=core_count)
    max_accuracy_achieved = 0
    min_accuracy_achieved = 101
    best_hyper_parameters_found_max = []
    best_hyper_parameters_found_min = []
    for result in pool.map(threaded_training, parameters):
        if result[0] > max_accuracy_achieved:
            max_accuracy_achieved = result[0]
            best_hyper_parameters_found_max = result[1]
            network_max = result[2]
        if result[0] < min_accuracy_achieved:
            min_accuracy_achieved = result[0]
            best_hyper_parameters_found_min = result[1]
            network_min = result[2]

    if (100 - min_accuracy_achieved) > max_accuracy_achieved:
        max_accuracy_achieved = (100 - min_accuracy_achieved)
        best_hyper_parameters_found_max = best_hyper_parameters_found_min
        network_max = network_min

    network_max.plot_learning_rate('red')

    print("Best accuracy: " + str(max_accuracy_achieved) + " | List of hyperparameters used: " + str(
        best_hyper_parameters_found_max[:7]))
    dump_on_json(max_accuracy_achieved, best_hyper_parameters_found_max, filename)

    # best_hyper_parameters_found_max[:7]
    nn_model = nt.Network(best_hyper_parameters_found_max[0], best_hyper_parameters_found_max[1],
                          best_hyper_parameters_found_max[2], best_hyper_parameters_found_max[3],
                          best_hyper_parameters_found_max[6],
                          best_hyper_parameters_found_max[4])

    return nn_model, best_hyper_parameters_found_max[5]


def dump_on_json(accuracy, hyper_parameters, filename):
    activation_functions = []
    for functions in hyper_parameters[1]:
        activation_functions.append((
            (str(functions[0])).split(" ")[1],
            (str(functions[1])).split(" ")[1]
        ))

    error_functions = (
        (str(hyper_parameters[2][0])).split(" ")[1],
        (str(hyper_parameters[2][1])).split(" ")[1]
    )

    model = {
        "accuracy": accuracy,
        "structure": hyper_parameters[0],
        "activation_function": activation_functions,
        "error_function": error_functions,
        "hyper_parameters": hyper_parameters[3],
        "gradient_descend_technique": hyper_parameters[4],
        "mini_batch_size": hyper_parameters[5],
        "regularization_technique": hyper_parameters[6]
    }

    models = []

    # if not os.path.exists(filename):
    #     file = open(filename, "x")

    file_exists = os.path.exists(filename)
    is_file_empty = file_exists and os.stat(filename).st_size == 0

    if file_exists and not is_file_empty:
        with open(filename, "r") as open_file:
            read_models = json.load(open_file)
            for single_model in read_models:
                models.append(single_model)
            open_file.close()

        with open(filename, "w") as open_file:
            models.append(model)
            json_object = json.dumps(models, indent=4)
            open_file.write(json_object)
            open_file.close()
    else:
        with open(filename, "w+") as open_file:
            models.append(model)
            json_object = json.dumps(models, indent=4)
            open_file.write(json_object)
            open_file.close()


def k_fold_cross_validation(data_set, output_data_set, hyper_parameters_set, k, randomized_search, filename):
    hps = get_hyper_parameters(hyper_parameters_set, randomized_search)

    # grid search over k
    best_accuracy = 0
    best_hp = []
    for hp in hps:
        accuracy = cross_validation_inner(data_set, output_data_set, hp, k)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_hp = hp

    network = nt.Network(best_hp[0], best_hp[1], best_hp[2], best_hp[3], best_hp[6], best_hp[4])
    network.train(network.stop, data_set, output_data_set, best_hp[5])

    return network


def cross_validation_inner(data_set, output_data_set, parameters, k):
    data_set_len = data_set.shape[0]
    proportions = int(np.ceil(data_set_len / k))
    accuracy_sum = 0
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

        accuracy, arguments, _ = threaded_training(hyperparameters)
        accuracy_sum += accuracy
    accuracy = accuracy_sum / k

    return accuracy
