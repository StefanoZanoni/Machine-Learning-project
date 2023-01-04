import numpy as np
from inspect import signature

from matplotlib import pyplot as plt


class Network:
    def __init__(self, structure, activation_functions, error_function, hyper_parameters):
        self.structure = structure
        self.activation_functions = activation_functions
        self.num_layers = len(structure)
        self.error_function = error_function
        self.hyper_parameters = hyper_parameters
        self.B = [np.random.randn(l, 1) for l in structure[1:]]
        self.W = [np.random.randn(l, next_l) for l, next_l in zip(structure[:-1], structure[1:])]
        self.pred_W = [np.zeros((l, next_l)) for l, next_l in zip(structure[:-1], structure[1:])]
        self.errors = []
        self.errors_means = []
        self.epochs = 0

    def backpropagation(self, x, y):
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

    def gradient_descent(self, mini_batch):
        DE_B = [np.zeros(b.shape) for b in self.B]
        DE_W = [np.zeros(W.shape) for W in self.W]

        for x, y in mini_batch:
            pDE_B, pDE_W = self.backpropagation(np.asmatrix(x).T, y)
            DE_B = [DE_b + pDE_b for DE_b, pDE_b in zip(DE_B, pDE_B)]
            DE_W = [DE_w + pDE_w for DE_w, pDE_w in zip(DE_W, pDE_W)]

        eta = self.hyper_parameters[1][1]
        d = len(mini_batch)
        self.pred_W = self.W
        self.W = [W - eta / d * DE_w for W, DE_w in zip(self.W, DE_W)]
        self.B = [b - eta / d * DE_b for b, DE_b in zip(self.B, DE_B)]

    # end: boolean function
    # training_input: array like
    # training_output: array like
    # mini_batch_size: int
    # eta: float
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

        # start training
        while not end():
            self.epochs += 1
            for mini_batch in mini_batches:
                self.gradient_descent(mini_batch)
            self.errors_means.append(np.sum(self.errors) / len(self.errors))

    def stop(self):
        # return self.epochs > 1000
        return np.sum([np.linalg.norm(np.abs(m1 - m2)) for m1, m2 in zip(self.W, self.pred_W)]) / len(self.W) < 0.001

    def plot_learning_rate(self, problem_number):
        plt.plot(range(1, self.epochs + 1), self.errors_means)
        plt.xlabel('epochs')
        plt.ylabel('error')
        plt.title('Monk' + str(problem_number) + ' learning rate')
        plt.show()
