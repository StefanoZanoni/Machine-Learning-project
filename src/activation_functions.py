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
    return np.tanh(x)


def tanh_gradient(x):
    return 1 - np.square(np.tanh(x))


def elu(x, alpha):
    y = np.ravel(np.copy(x))
    y[np.ravel(x) < 0] = alpha * (np.exp(np.ravel(x)[np.ravel(x) < 0]) - 1)
    return np.array([y]).T


def elu_gradient(x, alpha):
    dx = np.ones_like(np.ravel(x))
    dx[np.ravel(x) < 0] = alpha * np.exp(np.ravel(x)[np.ravel(x) < 0])
    return np.array([dx]).T


def softmax(x):
    z = np.subtract(x, np.max(x))
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    return np.divide(numerator, denominator)


# dimensions problems
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


def selu(x):
    lambd_a = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    return lambd_a * (np.maximum(0, x) + np.minimum(0, alpha * (np.exp(x) - 1)))


def selu_gradient(x):
    lambda_hp = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    y = np.copy(np.ravel(x))
    y[np.ravel(x) > 0] = lambda_hp
    y[np.ravel(x) <= 0] = lambda_hp * alpha * np.exp(np.ravel(x[np.ravel(x) < 0]))
    return np.array([y]).T


def relu(x):
    return np.maximum(0, x)


def relu_gradient(x):
    y = np.ones_like(x)
    y[x < 0] = 0
    return y


def leaky_relu(x, alpha):
    return np.maximum(np.multiply(alpha, x), x)


def leaky_relu_gradient(x, alpha):
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def softplus_gradient(x):
    return sigmoid(x)


def sigmoid(x):
    result = np.exp(-np.logaddexp(0., -x))
    result = np.minimum(result, 0.9999)
    result = np.maximum(result, 1e-20)
    return result


def sigmoid_gradient(x):
    return np.multiply(sigmoid(x), (np.ones_like(x) - sigmoid(x)))
