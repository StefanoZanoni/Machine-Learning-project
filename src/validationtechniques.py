import random
from multiprocessing.pool import ThreadPool
from timeit import default_timer as timer

import numpy as np
import network as nt
import multiprocessing


def threaded_training(list):
    structure = list[0]
    activation_functions = list[1]
    error_function = list[2]
    hyper_parameters = list[3]
    gradient_descent_technique = list[4]
    mini_batch_size = list[5]
    training_set = list[6]
    output_training_set = list[7]
    validation_set = list[8]
    output_validation_set = list[9]
    validation_set_len = list[10]

    network = nt.Network(structure, activation_functions, error_function, hyper_parameters, gradient_descent_technique)
    # start = timer()
    network.train(network.stop, training_set, output_training_set, mini_batch_size)
    # print("Training time in seconds:", np.ceil(timer() - start))
    # network.plot_learning_rate()

    correct_prevision = 0
    for x, y in zip(validation_set, output_validation_set):
        predicted_output = network.forward(x, True)
        if predicted_output == y:
            correct_prevision += 1

    accuracy = correct_prevision * 100 / validation_set_len
    # print(f'Accuracy: {accuracy:.2f}%\n')

    return accuracy, list


def holdout_validation(data_set, output_data_set, hyper_parameters_set, split_percentage):
    validation_set_len = int(np.ceil((100 - split_percentage) * data_set.shape[0] / 100))
    validation_set = data_set[:validation_set_len]
    training_set = data_set[validation_set_len:]
    output_validation_set = output_data_set[:validation_set_len]
    output_training_set = output_data_set[validation_set_len:]

    # [(structures, [[s1], [s2]]), (af, [[lr, sg], [r, sg]]), (ef, []), (hp, [(lr, []), (), ()]), (gdt, [""]), (batch, [])]
    structures = hyper_parameters_set[0][1]
    activation_functions_list = hyper_parameters_set[1][1]
    error_functions = hyper_parameters_set[2][1]
    hyper_parameters_list = hyper_parameters_set[3][1]
    gradient_descent_techniques = hyper_parameters_set[4][1]
    mini_batch_sizes = hyper_parameters_set[5][1]

    core_count = multiprocessing.cpu_count()

    if randomized_search:
        hp = randomized_grid_search(structures, activation_functions_list, error_functions, hyper_parameters_list,
                                    gradient_descent_techniques, mini_batch_sizes, output_training_set, training_set,
                                    validation_set,
                                    output_validation_set, validation_set_len)
    else:
        hp = exhaustive_grid_search(structures, activation_functions_list, error_functions, hyper_parameters_list,
                                    gradient_descent_techniques, mini_batch_sizes, output_training_set, training_set,
                                    validation_set,
                                    output_validation_set, validation_set_len)

    pool = ThreadPool(processes=core_count)
    max_accuracy_achieved = 0
    min_accuracy_achieved = 101
    best_hyper_parameters_found_max = []
    best_hyper_parameters_found_min = []
    for result in pool.map(threaded_training, hp):
       if result[0] > max_accuracy_achieved:
           max_accuracy_achieved = result[0]
           best_hyper_parameters_found_max = result[1]
       if result[0] < min_accuracy_achieved:
           min_accuracy_achieved = result[0]
           best_hyper_parameters_found_min = result[1]

    if (100 - min_accuracy_achieved) > max_accuracy_achieved:
       max_accuracy_achieved = (100 - min_accuracy_achieved)
       best_hyper_parameters_found_max = best_hyper_parameters_found_min

    print("Best accuracy: " + str(max_accuracy_achieved) + " | List of hyperparameters used: " + str(
        best_hyper_parameters_found_max[:6]))
    dump_on_json(max_accuracy_achieved, best_hyper_parameters_found_max, filename)


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
        "mini_batch_size": hyper_parameters[5]
    }

    models = []

    # if not os.path.exists(filename):
    #     file = open(filename, "x")
        
    
    with open(filename, "r") as openfile:
        # Reading from json file
        read_models = json.load(openfile)
        for single_model in read_models:
            models.append(single_model)

    models.append(model)
    json_object = json.dumps(models, indent=4)

    # Writing to sample.json
    with open(filename, "w") as outfile:
        outfile.write(json_object)


def k_fold_cross_validation(data_set, output_data_set, hyper_parameters_set):
    # grid search
    pass


def cross_validation_inner(data_set, output_data_set, hyper_parameters_set, k):
    data_set_len = data_set.shape[0]
    proportions = int(np.ceil(data_set_len / k))
    for validation_start in range(0, data_set_len, proportions):
        validation_end = validation_start + proportions
        validation_set = data_set[validation_start:validation_end]
        training_set1 = data_set[:validation_start]
        training_set2 = data_set[validation_end:]
        training_set = np.concatenate(training_set1, training_set2)

    pass
