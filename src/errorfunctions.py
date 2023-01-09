import numpy as np


# all the derivatives were computed with respect to the output (yx)


# mean squared error
def mse(y, yx):
    if np.shape(yx) != (1, 1):
        return np.sum(0.5 * np.square(np.subtract(yx, y))) / len(yx)
    else:
        return 0.5 * (yx - y) ** 2


# mean squared error derivative
def mse_derivative(y, yx):
    if np.shape(yx) != (1, 1):
        return np.sum(np.subtract(yx, y)) / len(yx)
    else:
        return yx - y


# mean absolute error
def mae(y, yx):
    if np.shape(yx) != (1, 1):
        return np.sum(np.abs(np.subtract(yx, y))) / len(yx)
    else:
        np.abs(yx - y)


# mean absolute error derivative
def mae_derivative(y, yx):
    if np.shape(yx) != (1, 1):
        return np.sum(np.multiply(1 / len(yx), np.divide(np.subtract(yx, y), np.abs(np.subtract(yx, y))))) / len(yx)
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
        return np.sum(hl) / len(yx)
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
        dhl[np.abs(np.subtract(yx, y)) > delta] = 1 / len(yx) * \
                                                  (np.multiply(delta, np.divide(np.subtract(yx, y),
                                                                                np.abs(np.subtract(yx, y)))))
        return np.sum(dhl) / len(yx)
    else:
        if np.abs(yx - y) <= delta:
            return mse_derivative(yx, y)
        else:
            return delta * (yx - y) / np.abs(yx - y)


# binary cross entropy
# problems with divide by zero
def bce(y, yx):
    # if np.shape(yx) != (1, 1):
    #     yx[yx == 0] = 0
    #     yx[yx == 1] = 0
    #     bool0_yx = yx != 0
    #     bool1_yx = yx != 1
    #     bool_yx = np.logical_or(bool0_yx, bool1_yx)
    #     return np.sum(-(np.add(np.multiply(y, np.log(yx), where=bool_yx),
    #                            np.multiply(np.subtract(np.ones_like(yx), y, where=bool_yx),
    #                                        np.log(np.subtract(np.ones_like(yx), yx), where=bool_yx), where=bool_yx),
    #                            where=bool_yx))) \
    #         / len(yx)
    # else:
    #     if yx == 0 or yx == 1:
    #         return 0
    return -(y * np.log(yx) + (1 - y) * np.log(1 - yx)).mean()


# binary cross entropy derivative
def bce_derivative(y, yx):
    # if np.shape(yx) != (1, 1):
    #     yx[yx == 1] = 0
    #     bool_yx = yx != 1
    #     return np.sum(np.divide(np.subtract(yx, y, where=bool_yx),
    #                             np.multiply(yx, np.subtract(np.ones_like(yx), yx, where=bool_yx), where=bool_yx),
    #                             where=bool_yx)) / len(yx)
    # else:
    #     if yx == 1:
    #         return 0
    return (yx - y) / (yx * (1 - yx)).mean()


# categorical cross entropy
def cce(y, yx):
    if np.shape(yx) != (1, 1):
        return np.sum(np.multiply(-y, np.log(yx))) / len(yx)
    else:
        return -y * np.log(yx)


# categorical cross entropy derivative
def cce_derivative(y, yx):
    if np.shape(yx) != (1, 1):
        return np.sum(np.divide(-y, yx))
    else:
        return -y / yx
