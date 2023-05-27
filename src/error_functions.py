import numpy as np

# all the derivatives were computed w.r.t the output (yx)
# x1, x2 are vectors, each operation has to be considered as an element wise one


# mean Euclidean error
def mee(y, yx):
    y = np.reshape(y, np.shape(yx))
    return np.linalg.norm(np.subtract(y, yx))


# mean Euclidean error gradient
def mee_gradient(y, yx):
    y = np.reshape(y, np.shape(yx))
    gradient = np.zeros_like(y)
    denominator = mee(y, yx)
    for i in range(len(gradient)):
        gradient[i] = (yx[i] - y[i]) / denominator
    return gradient


# root mean squared error
def rmse(y, yx):
    return np.sqrt(mse(y, yx))


# root mean squared error gradient
def rmse_gradient(y, yx):
    return (1 / 2 * np.sqrt(mse(y, yx))) * mse_gradient(y, yx)


# mean squared error
def mse(y, yx):
    if np.shape(yx) != (1, 1):
        y = np.reshape(y, np.shape(yx))
        return np.square(np.subtract(y, yx)).mean()
    else:
        return (y - yx) ** 2


# mean squared error gradient
def mse_gradient(y, yx):
    if np.shape(yx) != (1, 1):
        y = np.reshape(y, np.shape(yx))
        return 2 * (np.subtract(yx, y))
    else:
        return 2 * (yx - y)


# mean absolute error
def mae(y, yx):
    if np.shape(yx) != (1, 1):
        y = np.reshape(y, np.shape(yx))
        return np.abs(np.subtract(yx, y)).mean()
    else:
        np.abs(yx - y)


# mean absolute error gradient
def mae_gradient(y, yx):
    if np.shape(yx) != (1, 1):
        y = np.reshape(y, np.shape(yx))
        return np.divide(np.subtract(yx, y), np.abs(np.subtract(yx, y)))
    else:
        return (yx - y) / np.abs(yx - y)


# binary cross entropy
def bce(y, yx):
    y = np.reshape(y, np.shape(yx))
    return -(y * np.log(yx) + (1 - y) * np.log(1 - yx)).mean()


# binary cross entropy gradient
def bce_gradient(y, yx):
    y = np.reshape(y, np.shape(yx))
    return np.divide((yx - y), np.multiply(yx, (1 - yx)))


# categorical cross entropy
def cce(y, yx):
    if np.shape(yx) != (1, 1):
        y = np.reshape(y, np.shape(yx))
        return np.multiply(-y, np.log(yx)).mean()
    else:
        return -y * np.log(yx)


# categorical cross entropy gradient
def cce_gradient(y, yx):
    if np.shape(yx) != (1, 1):
        y = np.reshape(y, np.shape(yx))
        return np.divide(-y, yx)
    else:
        return -y / yx
