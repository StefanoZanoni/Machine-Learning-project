from src import activation_functions
from src import error_functions
from src import preprocessing
from src import holdout
from src import utilities


# MONK3
if __name__ == '__main__':
    # Reads the dataset
    data = utilities.read_monk_data_set(3)
    training_input3 = data[0]
    training_output3 = data[1]
    testing_input3 = data[2]
    testing_output3 = data[3]

    # Defines the list of activation functions to try
    # Each element of the list is a list that contains the activation functions for the model.
    # Each element of the internal list contains tuples that are assigned to each layer of the model.
    # A single tuple is of the form (function, gradient of that function)
    activation_functions3 = [
                             [(activation_functions.relu, activation_functions.relu_gradient),
                              (activation_functions.sigmoid, activation_functions.sigmoid_gradient)]
                             ]

    # Defines the tuple for the error function. The form of the tuple is
    # (error function, gradient of that error function)
    error_function3 = (error_functions.bce, error_functions.bce_gradient)

    # Defines the list of hyperparameters to try
    # Each element of the list is a list that contains a tuple (or more than one if we are using activation functions
    # that requires an additional parameters) with the name of the parameter and the value to try
    hyper_parameters3 = [[('learning_rate', 0.2)]]

    # Define a list of regularization to try. Each element of the list is a tuple that contains the name of the
    # regularization technique and then its value.
    # If there's no need to try a regularization technique, it is sufficient to add the tuple ("None", 0)
    regularization_techniques3 = [("None", 0)]

    # Encoding of the inputs
    training_input3 = preprocessing.one_hot_encoding(training_input3)
    testing_input3 = preprocessing.one_hot_encoding(testing_input3)

    # Model selection through holdout technique
    # The percentage of data to keep for the training is 70%
    # The False let us do an exhaustive search through all possible combinations of hyperparameters
    # The True let us define that it's a classification problem
    optimal_model = holdout.holdout_selection(training_input3, training_output3, [("structures", [[17, 1, 1]]),
                                                                                  (
                                                                                      "activation_functions",
                                                                                      activation_functions3),
                                                                                  (
                                                                                  "error_functions", [error_function3]),
                                                                                  ("hyper_parameters",
                                                                                   hyper_parameters3),
                                                                                  (
                                                                                      "gradient_descend_techniques",
                                                                                      ["NesterovM"]),
                                                                                  ("mini_batch_sizes",
                                                                                   [2]),
                                                                                  ("regularization_techniques",
                                                                                   regularization_techniques3)],
                                              70, False, "../Monk3_models.json", True)

    # With the optimal model found in the holdout selection, we compute the performance on the testing data
    performance = optimal_model.compute_performance(testing_input3, testing_output3)[0]
    print('performance on the test set: ' + str(performance))
