import sys
import numpy as np
from random import randint
from typing import Type
from inspect import signature
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

from src import activation_functions as af


class Network:

    def __init__(self, structure, activation_functions, error_function, hyper_parameters, is_classification,
                 regularization=("None", 0), optimizer="None", patience=10, delta=0):

        # Neural network structure in the form [in, L1, L2, ... , Ln, out]
        # where in and out are respectively the input and the output layer and
        # L1, ..., Ln are the hidden layers. Each layer can have a different number of neurons
        self.structure = structure

        # a list of tuples in the form [(f_1, f'_1), (f_2, f'_2), ..., (f_n, f'_n)]
        # where the k-th tuple represent the activation function and its derivative for the k-th layer
        self.activation_functions = activation_functions

        # the total number of layers
        self.num_layers = len(structure)

        # the metric used to compute the error of the neural network
        self.error_function = error_function

        # a list of tuples in the form [(hp_1, value_1), (hp_2, value_2), ..., (hp_n, value_n)]
        # where each tuple specify the name of one hyperparameter and its value
        self.hyper_parameters = hyper_parameters

        # the optimizer type for the gradient descent algorithm
        self.gradient_descent = optimizer

        # regularization technique
        self.regularization = regularization

        # a flag used to identy the problem type
        self.is_classification = is_classification

        # the list of biases matrices of each layer
        self.B = [np.zeros((l, 1)) for l in structure[1:]]

        # the list of weights matrices of each layer
        self.W = self.__weights_initialization()

        # the best list of biases found at the end of model selection
        self.best_B = self.B

        # the best list of weights found at the end of model selection
        self.best_W = self.W

        # the list of all lists of biases found over the epochs during the training
        self.Bs = []

        # the list of all lists of weights found over the epochs during the training
        self.Ws = []

        # the list of the derivative of the error function w.r.t the bias of each layer
        self.DE_B = [np.zeros(b.shape) for b in self.B]

        # the list of the derivative of the error function w.r.t the weights of each layer
        self.DE_W = [np.zeros(W.shape) for W in self.W]

        # the list of all training errors on the training data set
        self.training_errors = []

        # the list of all validation errors on the validation data set
        self.validation_errors = []

        # the list of the average errors over the epochs
        self.training_errors_means = []
        self.validation_errors_means = []

        if is_classification:
            self.best_validation_errors_means = (0, 0)
        else:
            self.best_validation_errors_means = sys.float_info.max

        self.epoch = 0
        self.patience = patience
        self.delta = delta

        # a flag used in case of a binary classification problem to take the "inverse" of the accuracy
        self.take_opposite = False

    # This function initializes the weights of the network
    def __weights_initialization(self):
        np.random.seed(0)
        weights_list = []

        for l, next_l, fun in zip(self.structure[:-1], self.structure[1:], self.activation_functions):

            # He weights uniform initialization for ReLu and its variants
            if fun[0].__code__.co_code == af.relu.__code__.co_code or \
                    fun[0].__code__.co_code == af.leaky_relu.__code__.co_code or \
                    fun[0].__code__.co_code == af.elu.__code__.co_code or \
                    fun[0].__code__.co_code == af.selu.__code__.co_code:

                std = np.sqrt(6.0 / l)
                weights = np.random.rand(l, next_l)
                scaled_weights = weights * std
                weights_list.append(scaled_weights)

            # Xavier/Glorot weights uniform initialization for Sigmoid and Tanh
            elif fun[0].__code__.co_code == af.sigmoid.__code__.co_code or \
                    fun[0].__code__.co_code == af.tanh.__code__.co_code:

                lower, upper = -(np.sqrt(6.0) / np.sqrt(l + next_l)), (
                        np.sqrt(6.0) / np.sqrt(l + next_l))
                weights = np.random.rand(l, next_l)
                scaled_weights = lower + weights * (upper - lower)
                weights_list.append(scaled_weights)

            else:
                weights = np.random.uniform(-0.7, 0.7001, (l, next_l))
                weights_list.append(weights)

        return weights_list

    # This function implements the forward propagation
    def forward(self, x):
        x = np.array([x]).T
        output = 0

        # Computes the output of each layer which it will be the input for the next one
        for bias, weights, j, i in zip(self.B, self.W, range(len(self.B)), range(len(self.structure))):
            _, output = self.__forward_step(bias, weights, x, output, j, i)

        # compute the predicted belonging class in case of a classification problem
        if self.is_classification:
            rounded_output = np.copy(output)

            # multi-class classification
            # return the most probable class
            if len(output) > 1:
                index = output.argmax(axis=0)
                index = index.item()
                mask = [1 if index == i else 0 for i in range(output.shape[0])]
                return np.array(mask), output

            # binary classification
            else:
                if self.take_opposite:
                    # Case where the model produced an accuracy less of 50%.
                    # So it's convenient to take as a prediction the opposite of the predicted output
                    rounded_output[output > 0.5] = 0
                    rounded_output[output < 0.5] = 1
                else:
                    # The Model produced an accuracy greater than 50%
                    rounded_output[output > 0.5] = 1
                    rounded_output[output < 0.5] = 0

                # In the case of prediction exactly 0.5,
                # the output is randomly chosen between 0 and 1
                rounded_output[output == 0.5] = randint(0, 1)
                return int(rounded_output), output

        # return simply the output in case of regression problem
        return output

    # This function executes a single step of the forward method.
    # It takes:
    #   - The vector of biases
    #   - The matrix weights
    #   - The input data x
    #   - The output of the previous iteration
    #   - The index j to access the activation function of the j-th layer
    #   - The index i to check if it's a computation of the first layer or not
    def __forward_step(self, bias, weights, x, output, j, i):
        f = self.activation_functions[j][0]

        # get the activation function signature to check whether some hyperparameters are needed
        sig = signature(f)
        params = sig.parameters

        # computation of the net
        net = weights.T @ output + bias if i != 0 else weights.T @ x + bias

        # Computation of the output
        if len(params) == 2:
            # in case of leaky ReLu or ELU
            alpha = self.hyper_parameters[1][1]
            output = f(net, alpha)
        else:
            output = f(net)

        return net, output

    # This function executes the backpropagation
    def __backpropagation(self, x, y, *args):

        # error derivative w.r.t the bias for pattern p
        pDE_B = [np.zeros(b.shape) for b in self.B]

        # error derivative w.r.t the weights for pattern p
        pDE_W = [np.zeros(w.shape) for w in self.W]

        # forward pass (same as before)
        output = 0
        NETs = []
        OUTPUTs = []
        for bias, weights, j, i in zip(self.B, self.W, range(len(self.B)), range(len(self.structure))):
            net, output = self.__forward_step(bias, weights, x, output, j, i)

            # store all the nets and outputs for the backward step
            NETs.append(net)
            OUTPUTs.append(output)

        last = self.num_layers - 2
        # 'e' is the error function and 'de' is its derivative
        e, de = self.error_function

        # storing the error for the current pattern at the end of the forward pass
        if self.is_classification:
            # Assign a value 0 or 1 to the predicted output
            if np.equal(output, 0.5):
                predicted_output = randint(0, 1)
            else:
                predicted_output = 1 if np.greater(output, 0.5) else 0

            prediction_error = 0 if np.equal(y, predicted_output) else 1
            error = e(y, OUTPUTs[last])
            self.training_errors.append((prediction_error, error))
        else:
            error = e(y, OUTPUTs[last])
            self.training_errors.append(error)

        # backward pass
        for layer in range(last, -1, -1):

            # get the derivative of the current layer activation function
            f1 = self.activation_functions[layer][1]

            # get the derivative function signature to check whether some hyperparameters are needed
            sig = signature(f1)
            params = sig.parameters

            if layer == last:
                # We compute the delta as the product between the error derivative
                # and the activation function derivative
                delta = np.multiply(de(y, OUTPUTs[layer]), f1(NETs[layer]))
            else:
                # We compute the delta as the product between the activation function derivative and the dot product
                # between the weights of the previous level (since we are backpropagating,
                # the previous layer is the layer + 1) and the previous delta
                if len(params) == 1:
                    if self.gradient_descent == "NesterovM":
                        # in the case of Nesterov Momentum, we compute the weights as the
                        # current weights minus a constant gamma
                        # (this is computed in __get_weights)
                        # times the momentum of the previous layer.
                        nesterov_vw = args[0]
                        delta = np.multiply(f1(NETs[layer]),
                                            (self.__get_weights(self.W[layer + 1], nesterov_vw[layer + 1]) @ delta))
                    else:
                        delta = np.multiply(f1(NETs[layer]), self.W[layer + 1] @ delta)
                else:
                    # in case of leaky ReLu or ELU
                    alpha = self.hyper_parameters[1][1]

                    if self.gradient_descent == "NesterovM":
                        nesterov_vw = args[0]
                        delta = np.multiply(f1(NETs[layer], alpha),
                                            (self.__get_weights(self.W[layer + 1], nesterov_vw[layer + 1]) @ delta))
                    else:
                        delta = np.multiply(f1(NETs[layer], alpha), self.W[layer + 1] @ delta)

            # the partial derivative of the error with respect to the bias is simply delta itself
            pDE_B[layer] = delta
            # the partial derivative of the error with respect to the weights is the dot product between the
            # output of the next layer (since we are backpropagating, the next layer is layer-1) and delta
            # if the current layer is the hidden one, or the dot product between
            # the input and delta if the current layer is the first one.
            pDE_W[layer] = OUTPUTs[layer - 1] @ delta.T if layer != 0 else x @ delta.T

        return pDE_B, pDE_W

    @staticmethod
    def __get_weights(w, v):
        gamma = 0.9
        w_look_ahead = w - gamma * v[0]
        return w_look_ahead

    # This function implements the gradient descent technique.
    # It takes:
    #   - The mini batch of data to train on
    #   - The training set length is the length of the whole training set
    #
    #   - The args parameter can take different meanings based on the gradient descent technique:
    #       - args[0] and args[1] are the velocities for weights and biases in the case of Nesterov Momentum
    #       - args[0] and args[1] are the velocities for weights and biases in the case of Ada Gradient
    #       - args[0] and args[1] are the velocities for weights and biases in the case of RMS Prop
    def __gradient_descent(self, mini_batch, training_set_len, *args):

        # error derivative w.r.t the bias
        self.DE_B = [np.zeros(b.shape) for b in self.B]

        # error derivative w.r.t the weights
        self.DE_W = [np.zeros(W.shape) for W in self.W]

        # takes Nesterov velocity for weights and biases
        if self.gradient_descent == "NesterovM":
            nesterov_vw = args[0]
            nesterov_vb = args[1]

        d = 1
        # In the case that the mini batch size is not the same of the training set, we set 'd' to the size of the
        # mini batch.
        # This 'd' will be used to divide the 'eta' parameter
        if len(mini_batch) != training_set_len:
            d = len(mini_batch)

        # Retrieves the regularization technique with its hyperparameter
        regularization, lambda_hp = self.regularization

        for x, y in mini_batch:
            if self.gradient_descent == "NesterovM":
                pDE_B, pDE_W = self.__backpropagation(np.asmatrix(x).T, y, nesterov_vw, nesterov_vb)
            else:
                pDE_B, pDE_W = self.__backpropagation(np.asmatrix(x).T, y)

            # update of the derivatives with the contribution of the p-th pattern
            self.DE_B = [DE_b + np.array(pDE_b) for DE_b, pDE_b in zip(self.DE_B, pDE_B)]
            self.DE_W = [DE_w + np.array(pDE_w) for DE_w, pDE_w in zip(self.DE_W, pDE_W)]

            # updates the velocities
            match self.gradient_descent:
                case "NesterovM":
                    nesterov_vw, nesterov_vb = self.__nesterovm_velocities_update(nesterov_vw, nesterov_vb, d)
                case "AdaGrad":
                    w_cache, b_cache = self.__ada_velocity_update(args[0], args[1])
                case "RMSprop":
                    w_cache, b_cache = self.__rmsprop_velocity_update(args[0], args[1])

        match self.gradient_descent:
            case "None":
                self.__standard_gradient_descent(regularization, lambda_hp, d)
            case "NesterovM":
                self.__nesterov_momentum(regularization, lambda_hp, d, nesterov_vw, nesterov_vb)
            case "AdaGrad":
                self.__ada_grad(w_cache, b_cache, regularization, lambda_hp, d)
            case "RMSprop":
                self.__rms_prop(w_cache, b_cache, regularization, lambda_hp, d)

    def __nesterovm_velocities_update(self, nesterov_vw, nesterov_vb, d):
        eta = self.hyper_parameters[0][1]
        # constant value, typically 0.9, 0.99, 0.999
        gamma = 0.9

        # the weight/bias velocity is updated with the following rule:
        # gamma times the previous velocity,
        # plus eta hyperparameter divided by the length of the mini batch
        # times the derivative of the error w.r.t the weights/biases
        temp_vw = [gamma * v + (eta / d) * DE_w for v, DE_w in zip(nesterov_vw, self.DE_W)]
        temp_vb = [gamma * v + (eta / d) * DE_b for v, DE_b in zip(nesterov_vb, self.DE_B)]

        for i in range(len(nesterov_vw)):
            nesterov_vw[i] = temp_vw[i]

        for i in range(len(nesterov_vb)):
            nesterov_vb[i] = temp_vb[i]

        return nesterov_vw, nesterov_vb

    def __ada_velocity_update(self, w_cache, b_cache):
        # the weight/bias velocity is updated with the following rule:
        # sum the previous values of the weight/bias cache with the square of
        # the derivative of the error w.r.t. weights/biases
        tempw_cache = list(map(np.add, w_cache, list(map(np.square, self.DE_W))))
        tempb_cache = list(map(np.add, b_cache, list(map(np.square, self.DE_B))))

        for i in range(len(w_cache)):
            tempw_cache[i] = np.array(tempw_cache[i])
            w_cache[i] = tempw_cache[i]

        for i in range(len(b_cache)):
            tempb_cache[i] = np.array(tempb_cache[i])
            b_cache[i] = tempb_cache[i]

        return w_cache, b_cache

    def __rmsprop_velocity_update(self, w_cache, b_cache):
        # constant value, typically 0.9, 0.99, 0.999
        decay_rate = 0.9

        # The following rule updates the weight/bias velocity:
        # sum of the first and second term.
        # Where:
        #   - first term is decay rate times the dimension of the velocity times the previous velocity
        #   - second term is (1 - decay rate) times the dimension of the velocity times the derivative of the error
        #     w.r.t the weight/bias squared
        w_first_term = list(map(np.multiply, [decay_rate] * len(w_cache), w_cache))
        w_second_term = list(
            map(np.multiply, [1 - decay_rate] * len(w_cache), list(map(np.square, self.DE_W))))
        tempw_cache = list(map(sum, w_first_term, w_second_term))

        b_first_term = list(map(np.multiply, [decay_rate] * len(b_cache), b_cache))
        b_second_term = list(
            map(np.multiply, [1 - decay_rate] * len(b_cache), list(map(np.square, self.DE_B))))
        tempb_cache = list(map(sum, b_first_term, b_second_term))

        for i in range(len(w_cache)):
            tempw_cache[i] = np.array(tempw_cache[i])
            w_cache[i] = tempw_cache[i]

        for i in range(len(b_cache)):
            tempb_cache[i] = np.array(tempb_cache[i])
            b_cache[i] = tempb_cache[i]

        return w_cache, b_cache

    # This function implements the standard gradient descent
    def __standard_gradient_descent(self, regularization, lambda_hp, d):
        eta = self.hyper_parameters[0][1]

        if regularization == "None":
            self.W = [W - ((eta / d) * DE_w) for W, DE_w in zip(self.W, self.DE_W)]
        if regularization == "L1":
            self.W = [W - ((eta / d) * (DE_w + lambda_hp * np.sign(W))) for W, DE_w in
                      zip(self.W, self.DE_W)]
        if regularization == "L2":
            self.W = [W - ((eta / d) * (DE_w + lambda_hp * W)) for W, DE_w in
                      zip(self.W, self.DE_W)]

        self.B = [b - ((eta / d) * DE_b) for b, DE_b in zip(self.B, self.DE_B)]

    # This function implements Nesterov Momentum
    def __nesterov_momentum(self, regularization, lambda_hp, d, nesterov_vw, nesterov_vb):
        eta = self.hyper_parameters[0][1]

        match regularization:
            # Nesterov Momentum weight/bias update rule, when no regularization occurs, is:
            # previous weight/bias minus the previous velocity
            case "None":
                self.W = [W - v for v, W in zip(nesterov_vw, self.W)]

            # L1 regularization follows the following rule:
            # add a penalty term equal to: lambda hyperparameter times the sign of the previous weights
            case "L1":
                self.W = [W - v + ((eta / d) * lambda_hp * np.sign(W)) for v, W in zip(nesterov_vw, self.W)]

            # L2 regularization follows the following rule:
            # add a penalty term equal to: lambda hyperparameter times the previous weights
            case "L2":
                self.W = [W - v + ((eta / d) * lambda_hp * W) for v, W in zip(nesterov_vw, self.W)]

        self.B = [B - v for v, B in zip(nesterov_vb, self.B)]

    def __ada_grad(self, w_cache, b_cache, regularization, lambda_hp, d):
        eta = self.hyper_parameters[0][1]
        eps = 1e-7

        if regularization == "None":
            self.W = [W - ((eta / d) * np.divide(DE_w, np.sqrt(w) + eps)) for W, DE_w, w in
                      zip(self.W, self.DE_W, w_cache)]
        if regularization == "L1":
            self.W = [W - ((eta / d) * (np.divide(DE_w, np.sqrt(w) + eps) + lambda_hp * np.sign(W)))
                      for
                      W, DE_w, w in
                      zip(self.W, self.DE_W, w_cache)]
        if regularization == "L2":
            self.W = [W - ((eta / d) * (np.divide(DE_w, np.sqrt(w) + eps) + lambda_hp * W))
                      for
                      W, DE_w, w in
                      zip(self.W, self.DE_W, w_cache)]

        self.B = [B - ((eta / d) * np.divide(DE_b, (np.sqrt(b) + eps))) for B, DE_b, b in
                  zip(self.B, self.DE_B, b_cache)]

    # This function implements RMS Prop
    def __rms_prop(self, w_cache, b_cache, regularization, lambda_hp, d):
        eta = self.hyper_parameters[0][1]
        # is a constant, typically 10^-7
        eps = 1e-7

        match regularization:
            # weights/bias update rule is: previous weights/bias minus (eta hyperparameter divided
            # by the mini batch size times the derivative of the error w.r.t the weights divided by (the square root
            # of the previous weights + epsilon))
            case "None":
                self.W = [W - (eta / d) * np.divide(DE_w, (np.sqrt(w) + eps)) for W, DE_w, w in
                          zip(self.W, self.DE_W, w_cache)]

            # L1 regularization follows the following rule:
            # add a penalty term equal to: lambda hyperparameter times the sign of the previous weights
            case "L1":
                self.W = [W - ((eta / d) * (np.divide(DE_w, np.sqrt(w) + eps) + lambda_hp * np.sign(W)))
                          for
                          W, DE_w, w in
                          zip(self.W, self.DE_W, w_cache)]

            # L2 regularization follows the following rule:
            # add a penalty term equal to: lambda hyperparameter times the previous weights
            case "L2":
                self.W = [W - ((eta / d) * (np.divide(DE_w, np.sqrt(w) + eps) + lambda_hp * W))
                          for
                          W, DE_w, w in
                          zip(self.W, self.DE_W, w_cache)]

        self.B = [B - (np.multiply(eta / d, DE_b) / (np.sqrt(b) + eps)) for B, DE_b, b in
                  zip(self.B, self.DE_B, b_cache)]

    def train(self, training_input, training_output, mini_batch_size, end, *args):

        # dividing training data into mini batches
        training_input = np.array(training_input)
        training_output = np.array(training_output)
        n = len(training_output)
        mini_batches = []
        mini_batch = []
        dt = np.dtype(np.ndarray, Type[int])

        if mini_batch_size == n:
            mini_batch = np.array([(inp, output) for inp, output in zip(training_input, training_output)], dtype=dt)
            mini_batches = [mini_batch]
        else:
            for i in range(1, n):
                mini_batch.append((training_input[i], training_output[i]))
                if i % mini_batch_size == 0 and i >= mini_batch_size:
                    temp = mini_batch.copy()
                    temp = np.array(temp, dtype=dt)
                    mini_batches.append(temp)
                    mini_batch.clear()

        mini_batches = np.array(mini_batches, dtype=np.ndarray)

        # start training
        while not end(args):
            self.epoch += 1
            self.training_errors = []

            for mini_batch in mini_batches:
                if self.gradient_descent == "AdaGrad" or self.gradient_descent == "RMSprop":
                    w_cache = [np.zeros_like(DE_w) for DE_w in self.DE_W]
                    b_cache = [np.zeros_like(DE_b) for DE_b in self.DE_B]
                    self.__gradient_descent(mini_batch, n, w_cache, b_cache)
                elif self.gradient_descent == "NesterovM":
                    nesterov_vw = [np.zeros_like(W) for W in self.W]
                    nesterov_vb = [np.zeros_like(B) for B in self.B]
                    self.__gradient_descent(mini_batch, n, nesterov_vw, nesterov_vb)
                else:
                    self.__gradient_descent(mini_batch, n)

            if self.is_classification:
                self.training_errors_means.append(
                    100 - (np.sum(self.training_errors) / len(self.training_errors) * 100))
            else:
                self.training_errors_means.append(np.mean(self.training_errors))

            self.Ws.append(self.W)
            self.Bs.append(self.B)

        if end.__code__.co_code == self.early_stopping.__code__.co_code:
            self.W = self.best_W
            self.B = self.best_B

    # training stop function with a maximum epoch limit
    def stop(self, args):
        max_epoch = args[0]
        if self.epoch >= max_epoch:
            return True
        return False

    # early stopping function for training termination
    def early_stopping(self, args):
        input_validation_set = args[0]
        output_validation_set = args[1]

        # check if at least one epoch is run
        if self.epoch >= 1:
            # compute the model performance on the validation set
            performance = self.compute_performance(input_validation_set, output_validation_set)
            self.validation_errors_means.append(performance)

        if self.epoch >= 2:
            if self.is_classification:
                # check if the accuracy is diminished on the validation set
                error_increasing = performance[0] - self.delta <= self.best_validation_errors_means[0]
            else:
                # check if the error is increased on the validation set
                error_increasing = performance + self.delta >= self.best_validation_errors_means

            if not error_increasing:
                # save the best weights and biases
                self.best_W = self.Ws[self.epoch - 1]
                self.best_B = self.Bs[self.epoch - 1]
                self.best_validation_errors_means = self.validation_errors_means[self.epoch - 1]
            else:
                # diminish the patience
                self.patience -= 1

        return self.patience == 0

    # This function is used to compute the model performance on a given dataset
    def compute_performance(self, input_data, output_data):

        if self.is_classification:
            # compute the accuracy
            correct_prevision = 0
            errors = []
            for x, y in zip(input_data, output_data):
                rounded = self.forward(x)[0]
                predicted = self.forward(x)[1]
                error = self.error_function[0](y, predicted)
                errors.append(error)
                if np.array_equal(rounded, y):
                    correct_prevision += 1

            accuracy = correct_prevision * 100 / len(output_data)

            return accuracy, np.mean(errors)
        else:
            # compute the error
            errors = []
            for x, y in zip(input_data, output_data):
                yx = self.forward(x)
                error = self.error_function[0](y, yx)
                errors.append(error)

            return np.mean(errors)

    def plot_learning_rate(self):
        if self.is_classification:
            training_error_a = [first for (first, second) in self.training_errors_means]
            training_error_l = [second for (first, second) in self.training_errors_means]

            validation_error_a = [first for (first, second) in self.validation_errors_means]
            validation_error_l = [second for (first, second) in self.validation_errors_means]

            # plot the error or accuracy on the training and validation set
            figure(figsize=(10, 6))
            plt.plot(range(1, self.epoch + 1), training_error_a, color='red', label='Training accuracy curve')
            plt.plot(range(1, self.epoch + 1), validation_error_a, color='green',
                     linestyle='dashed', label='Validation accuracy curve')
            plt.xlabel('Epochs', fontsize='20')
            plt.ylabel('Accuracy', fontsize='20')
            plt.legend(fontsize='20')
            plt.show()
        else:
            training_error_l = self.training_errors_means
            validation_error_l = self.validation_errors_means

        # plot the error or accuracy on the training and validation set
        figure(figsize=(10, 6))
        plt.plot(range(1, self.epoch + 1), training_error_l, color='red', label='Training loss curve')
        plt.plot(range(1, self.epoch + 1), validation_error_l, color='green',
                 linestyle='dashed', label='Validation loss curve')
        plt.xlabel('Epochs', fontsize='20')
        plt.ylabel('Loss', fontsize='20')
        plt.legend(fontsize='20')
        plt.show()
