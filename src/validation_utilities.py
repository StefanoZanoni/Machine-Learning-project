import json
import os
from random import random


# return a list of sets of hyperparameters to try
def get_hyper_parameters(hyper_parameters_set, randomized_search, is_classification):
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
                                    gradient_descent_techniques, mini_batch_sizes, regularization_techniques, is_classification)
    else:
        hp = exhaustive_grid_search(structures, activation_functions_list, error_functions, hyper_parameters_list,
                                    gradient_descent_techniques, mini_batch_sizes, regularization_techniques, is_classification)
    return hp


def randomized_grid_search(structures, activation_functions_list, error_functions, hyper_parameters_list,
                           gradient_descent_techniques, mini_batch_sizes, regularization_techniques, is_classification):
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
                   mini_batch_size, regularization_technique, is_classification])
    return hp


def exhaustive_grid_search(structures, activation_functions_list, error_functions, hyper_parameters_list,
                           gradient_descent_techniques, mini_batch_sizes, regularization_techniques, is_classification):
    hp = []
    for structure in structures:
        for activation_functions in activation_functions_list:
            for error_function in error_functions:
                for hyper_parameters in hyper_parameters_list:
                    for gradient_descent_technique in gradient_descent_techniques:
                        for mini_batch_size in mini_batch_sizes:
                            for regularization_technique in regularization_techniques:
                                hp.append([structure, activation_functions, error_function, hyper_parameters,
                                           gradient_descent_technique, mini_batch_size, regularization_technique, is_classification])
    return hp


def dump_on_json(performance, hyper_parameters, filename, is_classification):
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

    if is_classification:
        if performance < 90:
            return
        model = {
            "accuracy": performance,
            "structure": hyper_parameters[0],
            "activation_function": activation_functions,
            "error_function": error_functions,
            "hyper_parameters": hyper_parameters[3],
            "gradient_descent_technique": hyper_parameters[4],
            "mini_batch_size": hyper_parameters[5],
            "regularization_technique": hyper_parameters[6]
        }
    else:
        if performance > 1.5:
            return
        model = {
            "error": performance,
            "structure": hyper_parameters[0],
            "activation_function": activation_functions,
            "error_function": error_functions,
            "hyper_parameters": hyper_parameters[3],
            "gradient_descent_technique": hyper_parameters[4],
            "mini_batch_size": hyper_parameters[5],
            "regularization_technique": hyper_parameters[6]
        }

    models = []

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
