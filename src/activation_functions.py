import numpy as np


def binary_step(x):
    y = np.ones_like(x)
    y[x < 0] = 0
    return y


def binary_step_gradient(x):
    return np.zeros_like(x)


def linear(x):
    return x


def linear_gradient(x):
    return np.ones_like(x)


def tanh(x):
    return np.divide(np.subtract(np.exp(x), np.exp(-x)), np.add(np.exp(x), np.exp(-x)))


def tanh_gradient(x):
    return np.square(tanh(x))


def elu(x, alpha):
    y = np.ravel(np.copy(x))
    y[np.ravel(x) < 0] = alpha * (np.exp(np.ravel(x)[np.ravel(x) < 0]) - 1)
    return np.array([y]).T


def elu_gradient(x, alpha):
    dx = np.ones_like(np.ravel(x))
    dx[np.ravel(x) < 0] = alpha * np.exp(np.ravel(x)[np.ravel(x) < 0])
    return np.array([dx]).T


# dimensions problems
def softmax(x):
    y = np.copy(x)
    y -= np.max(y)
    return (np.exp(y).T / np.sum(np.exp(y), axis=0)).T


def softmax_gradient(x):
    sm = softmax(x)
    s = np.reshape(sm, (-1, 1))
    return np.diagflat(s) - np.dot(s, s.T)


def swish(x):
    return np.multiply(x, sigmoid(x))


def swish_gradient(x):
    return np.add(sigmoid(x), np.multiply(x, sigmoid_gradient(x)))


def gelu(x):
    return np.multiply(0.5 * x, (np.ones_like(x) + tanh(np.sqrt(2 / np.pi) * np.add(x, 0.044715 * np.power(x, 3)))))


def gelu_gradient(x):
    return 0.5 * (1 + tanh(np.sqrt(2 / np.pi) * np.add(x, 0.044715 * np.power(x, 3)))) + 0.5 * np.multiply(x, tanh(
        np.sqrt(2 / np.pi) *
        (np.add(x, 0.044715 * np.power(
            x, 3))))) * np.sqrt(
        2 / np.pi) * (1 + 3 * 0.044715 * np.square(x))


# dimensions problems
def selu(x):
    lambd_a = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    y = np.copy(np.ravel(x))
    y[np.ravel(x) > 0] = lambd_a * np.ravel(x)
    y[np.ravel(x) <= 0] = alpha * lambd_a * (np.exp(np.ravel(x)) - 1)
    return y


def selu_gradient(x):
    lambd_a = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    y = np.copy(np.ravel(x))
    y[np.ravel(x) > 0] = lambd_a
    y[np.ravel(x) <= 0] = lambd_a * alpha * np.exp(np.ravel(x))
    return y


def relu(x):
    return np.maximum(0, x)


def relu_gradient(x):
    return (x > 0) * 1


def leaky_relu(x, alpha):
    return np.maximum(np.multiply(alpha, x), x)


def leaky_relu_gradient(x, alpha):
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx


def softplus(x):
    return np.log(np.add(np.ones_like(x), np.exp(x)))


def softplus_gradient(x):
    return np.multiply(np.divide(np.ones_like(x), np.add(np.ones_like(x), np.exp(x))), np.exp(x))


def sigmoid(x):
    x = np.clip(x, -700, 700)
    result = np.divide(np.ones_like(x), np.add(np.ones_like(x), np.exp(-x)))
    result = np.minimum(result, 0.9999)
    result = np.maximum(result, 1e-20)
    return result


def sigmoid_gradient(x):
    result = np.multiply(sigmoid(x), (np.ones_like(x) - sigmoid(x)))
    return result
