from random import randint
from typing import Type

import numpy as np
from inspect import signature
import preprocessing
import activation_functions as af
import error_functions

from matplotlib import pyplot as plt


class Network:
    def __init__(self, structure, activation_functions, error_function, hyper_parameters, regularization=("None", 0),
                 optimizer="None", is_classification=True, eps=1e-6):
        self.structure = structure
        self.activation_functions = activation_functions
        self.num_layers = len(structure)
        self.error_function = error_function
        self.hyper_parameters = hyper_parameters
        self.gradient_descent = optimizer
        self.eps = eps
        self.regularization = regularization
        self.is_classification = is_classification

        self.B = [np.zeros((l, 1)) for l in structure[1:]]
        self.W = self.__weights_initialization()

        self.DE_B = [np.zeros(b.shape) for b in self.B]
        self.DE_W = [np.zeros(W.shape) for W in self.W]
        self.pred_W = [np.zeros((l, next_l)) for l, next_l in zip(structure[:-1], structure[1:])]
        self.errors = []
        self.errors_means = []
        self.epochs = 0

    def __weights_initialization(self):
        weights_list = []

        for l, next_l, s, fun in zip(self.structure[:-1], self.structure[1:], self.structure,
                                     self.activation_functions):

            # He weights initialization for ReLu
            if fun[0].__code__.co_code == af.relu.__code__.co_code or fun[
                0].__code__.co_code == af.leaky_relu.__code__.co_code:
                std = np.sqrt(2.0 / s)
                weights = np.random.randn(l, next_l)
                scaled_weights = weights * std
                weights_list.append(scaled_weights)

            # Xavier/Glorot weight initialization for Sigmoid or Tanh
            elif fun[0].__code__.co_code == af.sigmoid.__code__.co_code or fun[
                0].__code__.co_code == af.tanh.__code__.co_code:
                lower, upper = -(1.0 / np.sqrt(s)), (1.0 / np.sqrt(s))
                weights = np.random.randn(l, next_l)
                scaled_weights = lower + weights * (upper - lower)
                weights_list.append(scaled_weights)
            else:
                weights = np.random.randn(l, next_l)
                weights_list.append(weights)

        return weights_list

    def forward(self, x):
        x = np.array([x]).T

        for b, W, j, i in zip(self.B, self.W, range(len(self.B)), range(len(self.structure))):
            f = self.activation_functions[j][0]
            sig = signature(f)
            params = sig.parameters
            net = W.T @ output + b if i != 0 else W.T @ x + b
            if len(params) == 2:
                alpha = self.hyper_parameters[1][1]
                output = f(net, alpha)
            else:
                output = f(net)

# 1-0 [0.98 0.2]



        if self.is_classification:
            if len(output) > 1:
                # multiclass
                index = output.argmax(axis=1)
                mask = [1 if index == i else 0 for i in range(output[0])]
                return np.array(mask)
                # pass [0, 0, 1, 0] != [0, 0, 0, 1]
            elif len(output) == 1:
                output[output > 0.5] = 1
                output[output < 0.5] = 0
                output[output == 0.5] = randint(0, 1)
                return int(output)

        return output

    def __backpropagation(self, x, y):
        pDE_B = [np.zeros(b.shape) for b in self.B]
        pDE_W = [np.zeros(w.shape) for w in self.W]

        # forward
        NETs = []
        OUTPUTs = []
        for b, W, j in zip(self.B, self.W, range(len(self.B))):
            f = self.activation_functions[j][0]
            sig = signature(f)
            params = sig.parameters
            net = W.T @ output + b if NETs else W.T @ x + b
            if len(params) == 2:
                alpha = self.hyper_parameters[1][1]
                output = f(net, alpha)
            else:
                output = f(net)
            NETs.append(net)
            OUTPUTs.append(output)

        # storing the error for the current pattern
        last = self.num_layers - 2
        e, de = self.error_function
        sig2 = signature(e)
        params2 = sig2.parameters
        if len(params2) == 2:
            error = e(y, OUTPUTs[last])
        else:
            beta = self.hyper_parameters[2][1]
            e(y, OUTPUTs[last], beta)
        if np.shape(error) == (1, 1):
            self.errors.append(error.item())
        else:
            self.errors.append(error)

        # backward
        for layer in range(last, -1, -1):

            f1 = self.activation_functions[layer][1]
            sig1 = signature(f1)
            params1 = sig1.parameters

            if layer == last:
                if len(params2) == 2:
                    delta = np.multiply(de(y, OUTPUTs[layer]), f1(NETs[layer]))
                else:
                    beta = self.hyper_parameters[2][1]
                    delta = np.multiply(de(y, OUTPUTs[layer], beta), f1(NETs[layer]))
            else:
                if len(params1) == 1:
                    delta = np.multiply(f1(NETs[layer]), (self.W[layer + 1] @ delta))
                else:
                    alpha = self.hyper_parameters[1][1]
                    delta = np.multiply(f1(NETs[layer], alpha), (self.W[layer + 1] @ delta))

            pDE_B[layer] = delta
            pDE_W[layer] = OUTPUTs[layer - 1] @ delta.T if layer != 0 else x @ delta.T

        return pDE_B, pDE_W

    def __gradient_descent(self, mini_batch, training_set_len):
        self.DE_B = [np.zeros(b.shape) for b in self.B]
        self.DE_W = [np.zeros(W.shape) for W in self.W]
        w_cache = [np.zeros_like(DE_w) for DE_w in self.DE_W]
        b_cache = [np.zeros_like(DE_b) for DE_b in self.DE_B]

        for x, y in mini_batch:
            pDE_B, pDE_W = self.__backpropagation(np.asmatrix(x).T, y)
            self.DE_B = [DE_b + pDE_b for DE_b, pDE_b in zip(self.DE_B, pDE_B)]
            self.DE_W = [DE_w + pDE_w for DE_w, pDE_w in zip(self.DE_W, pDE_W)]

        d = 1
        if len(mini_batch) != training_set_len:
            d = len(mini_batch)
        regularization, lambda_hp = self.regularization

        if self.gradient_descent == "None":
            self.__standard_gradient_descent(regularization, lambda_hp, d)
        elif self.gradient_descent == "NesterovM":
            self.__nesterov_momentum(regularization, lambda_hp, d)
        elif self.gradient_descent == "AdaGrad":
            self.__ada_grad(w_cache, b_cache, regularization, lambda_hp, d)
        elif self.gradient_descent == "RMSProp":
            self.__rms_prop(0.9, w_cache, b_cache, regularization, lambda_hp, d)

    def __standard_gradient_descent(self, regularization, lambda_hp, d):
        eta = self.hyper_parameters[0][1]
        if regularization == "None":
            self.W = [W - ((eta / d) * DE_w) for W, DE_w in zip(self.W, self.DE_W)]
        if regularization == "L1":
            self.W = [W - ((eta / d) * lambda_hp * np.sign(W)) - ((eta / d) * DE_w) for W, DE_w in
                      zip(self.W, self.DE_W)]
        if regularization == "L2":
            self.W = [W - (2 * (eta / d) * lambda_hp * W) - ((eta / d) * DE_w) for W, DE_w in
                      zip(self.W, self.DE_W)]

        self.B = [b - ((eta / d) * DE_b) for b, DE_b in zip(self.B, self.DE_B)]

    def __nesterov_momentum(self, regularization, lambda_hp, d):
        eta = self.hyper_parameters[0][1]
        # if regularization == "None":
        # self.W = [W - ((eta / d) * DE_w) + v]
        pass

    def __ada_grad(self, w_cache, b_cache, regularization, lambda_hp, d):
        eta = self.hyper_parameters[0][1]
        w_cache = list(map(np.add, w_cache, list(map(np.square, self.DE_W))))
        b_cache = list(map(np.add, b_cache, list(map(np.square, self.DE_B))))
        if regularization == "None":
            self.W = [W - (np.multiply(eta / d, DE_w) / (np.sqrt(w) + self.eps)) for W, DE_w, w in
                      zip(self.W, self.DE_W, w_cache)]
        if regularization == "L1":
            self.W = [W - ((eta / d) * lambda_hp * np.sign(W)) - (np.multiply(eta / d, DE_w) / (np.sqrt(w) + self.eps))
                      for
                      W, DE_w, w in
                      zip(self.W, self.DE_W, w_cache)]
        if regularization == "L2":
            self.W = [W - (2 * (eta / d) * lambda_hp * W) - (np.multiply(eta / d, DE_w) / (np.sqrt(w) + self.eps)) for
                      W, DE_w, w in
                      zip(self.W, self.DE_W, w_cache)]

        # self.W = [W - np.multiply(eta, DE_w) / (np.sqrt(w) + self.eps) for W, DE_w, w in
        #           zip(self.W, self.DE_W, w_cache)]

        self.B = [B - np.multiply(eta / d, DE_b) / (np.sqrt(b) + self.eps) for B, DE_b, b in
                  zip(self.B, self.DE_B, b_cache)]

    def __rms_prop(self, decay_rate, w_cache, b_cache, regularization, lambda_hp, d):
        eta = self.hyper_parameters[0][1]
        w_first_term = list(map(np.multiply, [decay_rate for i in range(len(w_cache))], w_cache))
        w_second_term = list(
            map(np.multiply, [1 - decay_rate for i in range(len(w_cache))], list(map(np.square, self.DE_W))))
        w_cache = list(map(sum, w_first_term, w_second_term))

        b_first_term = list(map(np.multiply, [decay_rate for i in range(len(b_cache))], b_cache))
        b_second_term = list(
            map(np.multiply, [1 - decay_rate for i in range(len(b_cache))], list(map(np.square, self.DE_B))))
        b_cache = list(map(sum, b_first_term, b_second_term))

        if regularization == "None":
            self.W = [W - np.multiply(eta / d, DE_w) / (np.sqrt(w) + self.eps) for W, DE_w, w in
                      zip(self.W, self.DE_W, w_cache)]
        if regularization == "L1":
            self.W = [W - ((eta / d) * lambda_hp * np.sign(W)) - (np.multiply(eta / d, DE_w) / (np.sqrt(w) + self.eps))
                      for
                      W, DE_w, w in
                      zip(self.W, self.DE_W, w_cache)]
        if regularization == "L2":
            self.W = [W - (2 * (eta / d) * lambda_hp * W) - (np.multiply(eta / d, DE_w) / (np.sqrt(w) + self.eps)) for
                      W, DE_w, w
                      in
                      zip(self.W, self.DE_W, w_cache)]

        # self.W = [W - np.multiply(eta, DE_w) / (np.sqrt(w) + self.eps) for W, DE_w, w in
        #           zip(self.W, self.DE_W, w_cache)]

        self.B = [B - (np.multiply(eta / d, DE_b) / (np.sqrt(b) + self.eps)) for B, DE_b, b in
                  zip(self.B, self.DE_B, b_cache)]

    # end: boolean function
    # training_input: array like
    # training_output: array like
    # mini_batch_size: int
    def train(self, end, training_input, training_output, mini_batch_size):
        # dividing training data into mini batch
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
        while not end():
            self.errors = []
            self.epochs += 1
            # print("Epochs: " + str(self.epochs))

            for mini_batch in mini_batches:
                # mini_batch = preprocessing.shuffle_data(mini_batch)
                self.__gradient_descent(mini_batch, n)
                # print("DE_W: " + str(self.DE_W))

            self.errors_means.append(np.sum(self.errors) / len(self.errors))
            # print("Sum of all error at epoch " + str(self.epochs) + ": " + str(np.sum(self.errors)))
            # print("Error mean at epoch " + str(self.epochs) + ": " + str(np.sum(self.errors) / len(self.errors)))

    def test_set_accuracy(self, test_data_input, test_data_output):

        correct_prevision = 0

        for x, y in zip(test_data_input, test_data_output):
            predicted_output = self.forward(x)
            if predicted_output == y:
                correct_prevision += 1

        if self.is_classification:

            accuracy = correct_prevision * 100 / len(test_data_output)
            print("Accuracy: ", accuracy)

            return accuracy

        else:
            mean_squared_error = error_functions.mse(test_data_output, predicted_output)
            root_mean_squared_error = np.sqrt(mean_squared_error)
            mean_absolute_error = error_functions.mae(test_data_output, predicted_output)
            print("MSE: {}, RMSE: {}, MAE: {}".format(str(mean_squared_error), str(root_mean_squared_error),
                                                      str(mean_absolute_error)))

            return mean_squared_error, root_mean_squared_error, mean_absolute_error

    def stop(self):
        return self.epochs > 700
        # return np.sum([np.linalg.norm(np.abs(m1 - m2)) for m1, m2 in zip(self.W, self.pred_W)]) / len(self.W) < 0.1

    def plot_learning_rate(self, color='red'):
        plt.plot(range(1, self.epochs + 1), self.errors_means, color=color)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Learning curve')
        plt.show()
