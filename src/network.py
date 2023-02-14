import sys
from random import randint
from typing import Type

import numpy as np
from inspect import signature
import activation_functions as af
import error_functions

from matplotlib import pyplot as plt


class Network:

    def __init__(self, structure, activation_functions, error_function, hyper_parameters, is_classification,
                 regularization=("None", 0), optimizer="None", eps=1e-7):
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
        self.best_B = self.B
        self.best_W = self.W
        self.Ws = []
        self.Bs = []

        self.DE_B = [np.zeros(b.shape) for b in self.B]
        self.DE_W = [np.zeros(W.shape) for W in self.W]
        self.errors = []
        self.validation_errors = []
        self.errors_means = []
        self.validation_errors_means = []
        self.best_validation_errors_means = sys.float_info.max
        self.epoch = 0

        self.take_opposite = False

    def __weights_initialization(self):
        np.random.seed(0)
        weights_list = []

        for l, next_l, s, fun in zip(self.structure[:-1], self.structure[1:], range(len(self.structure)),
                                     self.activation_functions):

            # He weights initialization for ReLu
            if fun[0].__code__.co_code == af.relu.__code__.co_code or \
                    fun[0].__code__.co_code == af.parametric_relu.__code__.co_code:
                std = np.sqrt(2.0 / self.structure[s])
                weights = np.random.rand(l, next_l)
                scaled_weights = weights * std
                weights_list.append(scaled_weights)

            # normalized Xavier/Glorot weights initialization for Sigmoid or Tanh
            elif fun[0].__code__.co_code == af.sigmoid.__code__.co_code or \
                    fun[0].__code__.co_code == af.tanh.__code__.co_code:
                lower, upper = -(np.sqrt(6.0) / np.sqrt(self.structure[s] + self.structure[s + 1])), (
                        np.sqrt(6.0) / np.sqrt(self.structure[s] + self.structure[s + 1]))
                weights = np.random.rand(l, next_l)
                scaled_weights = lower + weights * (upper - lower)
                weights_list.append(scaled_weights)
            else:
                weights = np.random.uniform(-0.7, 0.7001, (l, next_l))
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

        if self.is_classification:
            if len(output) > 1:
                index = output.argmax(axis=0)
                index = index.item()
                mask = [1 if index == i else 0 for i in range(output.shape[0])]
                return np.array(mask)
            elif len(output) == 1:
                if self.take_opposite:
                    output[output > 0.5] = 0
                    output[output < 0.5] = 1
                else:
                    output[output > 0.5] = 1
                    output[output < 0.5] = 0
                output[output == 0.5] = randint(0, 1)
                return int(output)

        return output

    def __backpropagation(self, x, y, *args):
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
            error = e(y, OUTPUTs[last], beta)
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
                    if self.gradient_descent == "NesterovM":
                        nesterov_vw = args[0]
                        delta = np.multiply(f1(NETs[layer]),
                                            (self.__get_weights(self.W[layer + 1], nesterov_vw[layer + 1]) @ delta))
                    else:
                        delta = np.multiply(f1(NETs[layer]), self.W[layer + 1] @ delta)
                else:
                    alpha = self.hyper_parameters[1][1]
                    if self.gradient_descent == "NesterovM":
                        nesterov_vw = args[0]
                        delta = np.multiply(f1(NETs[layer], alpha),
                                            (self.__get_weights(self.W[layer + 1], nesterov_vw[layer + 1]) @ delta))
                    else:
                        delta = np.multiply(f1(NETs[layer], alpha), self.W[layer + 1] @ delta)

            pDE_B[layer] = delta
            pDE_W[layer] = OUTPUTs[layer - 1] @ delta.T if layer != 0 else x @ delta.T

        return pDE_B, pDE_W

    @staticmethod
    def __get_weights(w, v):
        gamma = 0.9
        w_look_ahead = w - gamma * v[0]
        return w_look_ahead

    def __gradient_descent(self, mini_batch, training_set_len, *args):
        self.DE_B = [np.zeros(b.shape) for b in self.B]
        self.DE_W = [np.zeros(W.shape) for W in self.W]

        if self.gradient_descent == "NesterovM":
            nesterov_vw = args[0]
            nesterov_vb = args[1]

        d = 1
        if len(mini_batch) != training_set_len:
            d = len(mini_batch)
        regularization, lambda_hp = self.regularization

        for x, y in mini_batch:
            if self.gradient_descent == "NesterovM":
                pDE_B, pDE_W = self.__backpropagation(np.asmatrix(x).T, y, nesterov_vw, nesterov_vb)
            else:
                pDE_B, pDE_W = self.__backpropagation(np.asmatrix(x).T, y)
            self.DE_B = [DE_b + np.array(pDE_b) for DE_b, pDE_b in zip(self.DE_B, pDE_B)]
            self.DE_W = [DE_w + np.array(pDE_w) for DE_w, pDE_w in zip(self.DE_W, pDE_W)]

        if self.gradient_descent == "None":
            self.__standard_gradient_descent(regularization, lambda_hp, d)
        elif self.gradient_descent == "NesterovM":
            self.__nesterov_momentum(regularization, lambda_hp, d, nesterov_vw, nesterov_vb)
        elif self.gradient_descent == "AdaGrad":
            w_cache = args[0]
            b_cache = args[1]
            self.__ada_grad(w_cache, b_cache, regularization, lambda_hp, d)
        elif self.gradient_descent == "RMSprop":
            w_cache = args[0]
            b_cache = args[1]
            self.__rms_prop(0.9, w_cache, b_cache, regularization, lambda_hp, d)

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

    def __nesterov_momentum(self, regularization, lambda_hp, d, nesterov_vw, nesterov_vb):
        eta = self.hyper_parameters[0][1]
        gamma = 0.9

        temp_vw = [gamma * v + (eta / d) * DE_w for v, DE_w in zip(nesterov_vw, self.DE_W)]
        temp_vb = [gamma * v + (eta / d) * DE_b for v, DE_b in zip(nesterov_vb, self.DE_B)]
        for i in range(len(nesterov_vw)):
            nesterov_vw[i] = temp_vw[i]
        for i in range(len(nesterov_vb)):
            nesterov_vb[i] = temp_vb[i]

        if regularization == "None":
            self.W = [W - v for v, W in zip(nesterov_vw, self.W)]
        elif regularization == "L1":
            self.W = [W - v - ((eta / d) * lambda_hp * np.sign(W)) for v, W in zip(nesterov_vw, self.W)]
        elif regularization == "L2":
            self.W = [W - v - ((eta / d) * lambda_hp * W) for v, W in zip(nesterov_vw, self.W)]

        self.B = [B - v for v, B in zip(nesterov_vb, self.B)]

    def __ada_grad(self, w_cache, b_cache, regularization, lambda_hp, d):
        eta = self.hyper_parameters[0][1]

        tempw_cache = list(map(np.add, w_cache, list(map(np.square, self.DE_W))))
        tempb_cache = list(map(np.add, b_cache, list(map(np.square, self.DE_B))))

        for i in range(len(w_cache)):
            tempw_cache[i] = np.array(tempw_cache[i])
            w_cache[i] = tempw_cache[i]
        for i in range(len(b_cache)):
            tempb_cache[i] = np.array(tempb_cache[i])
            b_cache[i] = tempb_cache[i]

        if regularization == "None":
            self.W = [W - ((eta / d) * np.divide(DE_w, np.sqrt(w) + self.eps)) for W, DE_w, w in
                      zip(self.W, self.DE_W, w_cache)]
        if regularization == "L1":
            self.W = [W - ((eta / d) * (np.divide(DE_w, np.sqrt(w) + self.eps) + lambda_hp * np.sign(W)))
                      for
                      W, DE_w, w in
                      zip(self.W, self.DE_W, w_cache)]
        if regularization == "L2":
            self.W = [W - ((eta / d) * (np.divide(DE_w, np.sqrt(w) + self.eps) + lambda_hp * W))
                      for
                      W, DE_w, w in
                      zip(self.W, self.DE_W, w_cache)]

        self.B = [B - ((eta / d) * np.divide(DE_b, (np.sqrt(b) + self.eps))) for B, DE_b, b in
                  zip(self.B, self.DE_B, b_cache)]

    def __rms_prop(self, decay_rate, w_cache, b_cache, regularization, lambda_hp, d):
        eta = self.hyper_parameters[0][1]

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

        if regularization == "None":
            self.W = [W - (eta / d) * np.divide(DE_w, (np.sqrt(w) + self.eps)) for W, DE_w, w in
                      zip(self.W, self.DE_W, w_cache)]
        if regularization == "L1":
            self.W = [W - ((eta / d) * (np.divide(DE_w, np.sqrt(w) + self.eps) + lambda_hp * np.sign(W)))
                      for
                      W, DE_w, w in
                      zip(self.W, self.DE_W, w_cache)]
        if regularization == "L2":
            self.W = [W - ((eta / d) * (np.divide(DE_w, np.sqrt(w) + self.eps) + lambda_hp * W))
                      for
                      W, DE_w, w in
                      zip(self.W, self.DE_W, w_cache)]

        self.B = [B - (np.multiply(eta / d, DE_b) / (np.sqrt(b) + self.eps)) for B, DE_b, b in
                  zip(self.B, self.DE_B, b_cache)]

    # end: boolean function
    # training_input: array like
    # training_output: array like
    # mini_batch_size: int
    def train(self, training_input, training_output, mini_batch_size, end, *args):

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
        while not end(args):
            self.epoch += 1
            self.errors = []

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

            self.errors_means.append(np.mean(self.errors))
            self.Ws.append(self.W)
            self.Bs.append(self.B)

        self.W = self.best_W
        self.B = self.best_B

    def stop(self, args):
        patience_starting_point = args[0]
        patience = args[1]
        max_epoch = args[2]

        if self.epoch > 1:
            if self.errors_means[self.epoch - 1] - self.errors_means[self.epoch - 2] >= 0:
                patience[0] -= 1
                if patience[0] == patience_starting_point - 1:
                    self.best_W = self.Ws[self.epoch - 2]
                    self.best_B = self.Bs[self.epoch - 2]
            else:
                patience[0] = patience_starting_point

        if self.epoch > max_epoch:
            self.best_W = self.W
            self.best_B = self.B
            return True

        return patience[0] == 0

    def early_stopping(self, args):
        input_validation_set = args[0]
        output_validation_set = args[1]
        patience_starting_point = args[2]
        patience = args[3]

        if self.epoch >= 1:
            if self.is_classification:
                for pattern_in, pattern_out in zip(input_validation_set, output_validation_set):
                    predicted_output = self.forward(pattern_in)
                    error = np.abs(pattern_out - predicted_output)
                    self.validation_errors.append(error)
                accuracy = np.mean(self.validation_errors)
                self.validation_errors_means.append(accuracy) if accuracy >= 50 \
                    else self.validation_errors_means.append(100 - accuracy)
            else:
                for pattern_in, pattern_out in zip(input_validation_set, output_validation_set):
                    predicted_output = self.forward(pattern_in)
                    error = self.error_function[0](pattern_out, predicted_output)
                    self.validation_errors.append(error)
                self.validation_errors_means.append(np.mean(self.validation_errors))

        if self.epoch > 1:
            if self.is_classification:
                if self.validation_errors_means[self.epoch - 1] - self.validation_errors_means[self.epoch - 2] <= 0:
                    patience[0] -= 1
                    if patience[0] == patience_starting_point - 1:
                        self.best_W = self.Ws[self.epoch - 2]
                        self.best_B = self.Bs[self.epoch - 2]
                        self.best_validation_errors_means = self.validation_errors_means[self.epoch - 2]
                else:
                    patience[0] = patience_starting_point
            else:
                if self.validation_errors_means[self.epoch - 1] - self.validation_errors_means[self.epoch - 2] >= 0:
                    patience[0] -= 1
                    if patience[0] == patience_starting_point - 1:
                        self.best_W = self.Ws[self.epoch - 2]
                        self.best_B = self.Bs[self.epoch - 2]
                        self.best_validation_errors_means = self.validation_errors_means[self.epoch - 2]
                else:
                    patience[0] = patience_starting_point

        return patience[0] == 0

    def compute_performance(self, input_data, output_data):

        if self.is_classification:
            correct_prevision = 0

            for x, y in zip(input_data, output_data):
                predicted_output = self.forward(x)

                if np.array_equal(predicted_output, y):
                    correct_prevision += 1

            accuracy = correct_prevision * 100 / len(output_data)

            return accuracy
        else:
            predicted_outputs = []
            for x, y in zip(input_data, output_data):
                predicted_outputs.append(self.forward(x))
            predicted_outputs = np.array(predicted_outputs)

            errors = []
            if self.error_function[0].__code__.co_code == error_functions.mse.__code__.co_code:
                for predicted_output, output in zip(predicted_outputs, output_data):
                    errors.append(error_functions.mse(output, predicted_output))
                error = np.mean(errors)
            elif self.error_function[0].__code__.co_code == error_functions.mae.__code__.co_code:
                for predicted_output, output in zip(predicted_outputs, output_data):
                    errors.append(error_functions.mae(output, predicted_output))
                error = np.mean(errors)
            elif self.error_function[0].__code__.co_code == error_functions.rmse.__code__.co_code:
                for predicted_output, output in zip(predicted_outputs, output_data):
                    errors.append(error_functions.rmse(output, predicted_output))
                error = np.mean(errors)
            elif self.error_function[0].__code__.co_code == error_functions.mee.__code__.co_code:
                for predicted_output, output in zip(predicted_outputs, output_data):
                    errors.append(error_functions.mee(output, predicted_output))
                error = np.mean(errors)

            return error

    def plot_learning_rate(self, color='red'):
        plt.plot(range(1, self.epoch + 1), self.errors_means, color=color)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Learning curve')
        plt.show()
