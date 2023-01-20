import numpy as np


# all the derivatives were computed with respect to the output (yx)

def rmse(y, yx):
    return np.sqrt(mse(y, yx))


def rmse_derivative(y, yx):
    return (1 / 2) * np.sqrt(mse(y, yx)) * mse_derivative(y, yx)


# mean squared error
def mse(y, yx):
    if np.shape(yx) != (1, 1):
        return np.square(np.subtract(yx, y)).mean()
    else:
        return (yx - y) ** 2


# mean squared error derivative
def mse_derivative(y, yx):
    if np.shape(yx) != (1, 1):
        return 2 * ((np.subtract(yx, y)).mean())
    else:
        return 2 * (yx - y)


# mean absolute error
def mae(y, yx):
    if np.shape(yx) != (1, 1):
        return np.abs(np.subtract(yx, y)).mean()
    else:
        np.abs(yx - y)


# mean absolute error derivative
def mae_derivative(y, yx):
    if np.shape(yx) != (1, 1):
        return np.multiply(1 / len(yx), np.divide(np.subtract(yx, y), np.abs(np.subtract(yx, y)))).mean()
    else:
        return (yx - y) / np.abs(yx - y)


# huber loss
# problems with dimensions
def huber_loss(y, yx, delta):
    if np.shape(yx) != (1, 1):
        hl = np.zeros_like(yx)
        hl[np.abs(np.subtract(yx, y)) <= delta] = mse(y, yx)
        hl[np.abs(np.subtract(yx, y)) > delta] = (np.multiply(delta,
                                                              (np.subtract(np.abs(np.subtract(yx, y)),
                                                                           0.5 * delta)))) / len(yx)
        return hl.mean()
    else:
        if np.abs(yx - y) <= delta:
            return mse(y, yx)
        else:
            return delta * (np.abs(yx - y) - 0.5 * delta)


# huber loss derivative
def huber_loss_derivative(y, yx, delta):
    if np.shape(yx) != (1, 1):
        dhl = np.zeros_like(yx)
        dhl[np.abs(np.subtract(yx, y)) <= delta] = mse_derivative(y, yx)
        dhl[np.abs(np.subtract(yx, y)) > delta] = 1 / len(yx) * (np.multiply(delta, np.divide(np.subtract(yx, y),
                                                                                              np.abs(
                                                                                                  np.subtract(yx, y)))))
        return dhl.mean()
    else:
        if np.abs(yx - y) <= delta:
            return mse_derivative(yx, y)
        else:
            return delta * (yx - y) / np.abs(yx - y)


# binary cross entropy
# problems with divide by zero
def bce(y, yx):
    return -(y * np.log(yx) + (1 - y) * np.log(1 - yx)).mean()


# binary cross entropy derivative
def bce_derivative(y, yx):
    return np.divide((yx - y), np.multiply(yx, (1 - yx))).mean()


# categorical cross entropy
def cce(y, yx):
    if np.shape(yx) != (1, 1):
        return np.multiply(-y, np.log(yx)).mean()
    else:
        return -y * np.log(yx)


# categorical cross entropy derivative
def cce_derivative(y, yx):
    if np.shape(yx) != (1, 1):
        return np.divide(-y, yx).mean()
    else:
        return -y / yx
