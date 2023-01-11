from random import randint

import numpy as np
from inspect import signature
import preprocessing
import activationfunctions as af

from matplotlib import pyplot as plt


class Network:
    def __init__(self, structure, activation_functions, error_function, hyper_parameters, optimizer="SGD", eps=1e-6):
        self.structure = structure
        self.activation_functions = activation_functions
        self.num_layers = len(structure)
        self.error_function = error_function
        self.hyper_parameters = hyper_parameters
        self.gradient_descent = optimizer
        self.eps = eps

        self.B = [np.zeros((l, 1)) for l in structure[1:]]
        # self.W = [np.random.randn(l, next_l) for l, next_l in zip(structure[:-1], structure[1:])]
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
            if fun == af.relu or fun == af.leaky_relu:
                std = np.sqrt(2.0 / s)
                weights = np.random.randn(l, next_l)
                scaled_weights = weights * std
                weights_list.append(scaled_weights)

            # Xavier/Glorot weight initialization for Sigmoid or Tanh
            elif fun == af.sigmoid or fun == af.tanh:
                lower, upper = -(1.0 / np.sqrt(s)), (1.0 / np.sqrt(s))
                weights = np.random.randn(l, next_l)
                scaled_weights = lower + weights * (upper - lower)
                weights_list.append(weights)
            else:
                weights = np.random.randn(l, next_l)
                weights_list.append(weights)

        return weights_list

    def forward(self, x, is_classification):
        x = np.array([x]).T
        alpha = self.hyper_parameters[0][1]
        output = 0

        for b, W, j, i in zip(self.B, self.W, range(len(self.B)), range(len(self.structure))):
            f = self.activation_functions[j][0]
            sig = signature(f)
            params = sig.parameters
            net = np.array(W.T @ output + b if i != 0 else W.T @ x + b)
            output = f(net, alpha) if len(params) == 2 else f(net)

        if is_classification:
            output[output > 0.5] = 1
            output[output < 0.5] = 0
            output[output == 0.5] = randint(0, 1)

        return int(output)

    def __backpropagation(self, x, y):
        pDE_B = [np.zeros(b.shape) for b in self.B]
        pDE_W = [np.zeros(w.shape) for w in self.W]

        alpha = self.hyper_parameters[0][1]
        beta = self.hyper_parameters[2][1]

        # forward
        NETs = []
        OUTPUTs = []
        for b, W, j in zip(self.B, self.W, range(len(self.B))):
            f = self.activation_functions[j][0]
            sig = signature(f)
            params = sig.parameters
            net = np.array(W.T @ output + b if NETs else W.T @ x + b)
            output = f(net, alpha) if len(params) == 2 else f(net)
            NETs.append(net)
            OUTPUTs.append(output)

        # storing the error for the current pattern
        last = self.num_layers - 2
        e, de = self.error_function
        sig2 = signature(de)
        params2 = sig2.parameters
        error = e(y, OUTPUTs[last]) if len(params2) == 2 else e(y, OUTPUTs[last], beta)
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
                    delta = np.multiply(de(y, OUTPUTs[layer], beta), f1(NETs[layer]))
            else:
                if len(params1) == 1:
                    delta = np.multiply(f1(NETs[layer]), (self.W[layer + 1] @ delta))
                else:
                    delta = np.multiply(f1(NETs[layer], alpha), (self.W[layer + 1] @ delta))

            pDE_B[layer] = delta
            pDE_W[layer] = OUTPUTs[layer - 1] @ delta.T if layer != 0 else x @ delta.T

        return pDE_B, pDE_W

    def __gradient_descent(self, mini_batch, w_cache, b_cache):

        for x, y in mini_batch:
            pDE_B, pDE_W = self.__backpropagation(np.asmatrix(x).T, y)
            self.DE_B = [DE_b + pDE_b for DE_b, pDE_b in zip(self.DE_B, pDE_B)]
            self.DE_W = [DE_w + pDE_w for DE_w, pDE_w in zip(self.DE_W, pDE_W)]

        eta = self.hyper_parameters[1][1]
        d = len(mini_batch)
        self.pred_W = self.W

        if self.gradient_descent == "SGD":
            self.W = [W - eta / d * DE_w for W, DE_w in zip(self.W, self.DE_W)]
            self.B = [b - eta / d * DE_b for b, DE_b in zip(self.B, self.DE_B)]
        elif self.gradient_descent == "AdaGrad":
            self.__ada_grad(w_cache, b_cache)
        elif self.gradient_descent == "RMSProp":
            self.__rms_prop(0.9, w_cache, b_cache)

    def __ada_grad(self, w_cache, b_cache):
        w_cache = list(map(np.add, w_cache, list(map(np.square, self.DE_W))))
        b_cache = list(map(np.add, b_cache, list(map(np.square, self.DE_B))))
        eta = self.hyper_parameters[1][1]
        self.W = [W - np.multiply(eta, DE_w) / (np.sqrt(w) + self.eps) for W, DE_w, w in
                  zip(self.W, self.DE_W, w_cache)]
        self.B = [B - np.multiply(eta, DE_b) / (np.sqrt(b) + self.eps) for B, DE_b, b in
                  zip(self.B, self.DE_B, b_cache)]

    def __rms_prop(self, decay_rate, w_cache, b_cache):
        w_first_term = list(map(np.multiply, [decay_rate for i in range(len(w_cache))], w_cache))
        w_second_term = list(
            map(np.multiply, [1 - decay_rate for i in range(len(w_cache))], list(map(np.square, self.DE_W))))
        w_cache = list(map(sum, w_first_term, w_second_term))

        b_first_term = list(map(np.multiply, [decay_rate for i in range(len(b_cache))], b_cache))
        b_second_term = list(
            map(np.multiply, [1 - decay_rate for i in range(len(b_cache))], list(map(np.square, self.DE_B))))
        b_cache = list(map(sum, b_first_term, b_second_term))

        eta = self.hyper_parameters[1][1]
        self.W = [W - np.multiply(eta, DE_w) / (np.sqrt(w) + self.eps) for W, DE_w, w in
                  zip(self.W, self.DE_W, w_cache)]
        self.B = [B - np.multiply(eta, DE_b) / (np.sqrt(b) + self.eps) for B, DE_b, b in
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

        if mini_batch_size == n:
            mini_batch = [(training_input[i], training_output[i]) for i in range(1, n)]
            mini_batches = [mini_batch]
        else:
            for i in range(1, n):
                mini_batch.append((training_input[i], training_output[i]))
                if i % mini_batch_size == 0 and i >= mini_batch_size:
                    temp = mini_batch.copy()
                    mini_batches.append(temp)
                    mini_batch.clear()

        mini_batches = np.array(mini_batches, dtype=np.ndarray)

        w_cache = [np.zeros_like(DE_w) for DE_w in self.DE_W]
        b_cache = [np.zeros_like(DE_b) for DE_b in self.DE_B]

        # start training
        while not end():
            self.epochs += 1

            if mini_batch_size == 1:
                mini_batches = preprocessing.shuffle_data(mini_batches)

            for mini_batch in mini_batches:
                self.__gradient_descent(mini_batch, w_cache, b_cache)
            self.errors_means.append(np.sum(self.errors) / len(self.errors))

    def stop(self):
        return self.epochs > 10
        # return np.sum([np.linalg.norm(np.abs(m1 - m2)) for m1, m2 in zip(self.W, self.pred_W)]) / len(self.W) < 0.001

    def plot_learning_rate(self):
        plt.plot(range(1, self.epochs + 1), self.errors_means, color='red')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('learning rate')
        plt.show()
