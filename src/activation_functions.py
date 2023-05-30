import numpy as np


# linear activation function
# useful in the output layer of an NN for a regression problem since it leaves the input unchanged.
# f(x) = x
#
# pros: 1) very simple.
#
# cons: 1) all layers of the neural network will collapse into one if a linear activation function is used,
#          so, essentially, a linear activation function turns the neural network into just one layer.
def linear(x):
    return x


# linear activation function gradient
# df(x) = 1
#
# pros: 1) very simple.
#
# cons: 1) it’s not possible to use backpropagation as the derivative of the function is a constant
#          and has no relation to the input x.
def linear_gradient(x):
    return np.ones_like(x)


# tanh activation function
# usually used in hidden layers of a neural network
# f(x) = (e^x - e^-x) / (e^x + e^-x)
#
# pros: 1) the output of the tanh activation function is zero centered;
#          hence we can easily map the output values as strongly negative, neutral, or strongly positive.
#       2) usually used in hidden layers of a neural network as its values lie between -1 to 1;
#          therefore, the mean for the hidden layer comes out to be 0 or very close to it.
#          It helps in centering the data and makes learning for the next layer much easier.
#       3) the output values are bound between -1 to 1. This means the activations will not be blown up.
#
# Cons: 1) computationally expansive
def tanh(x):
    return np.tanh(x)


# Tanh activation function gradient
# df(x) = 1 - tanh^2(x)
#
# pros: -
#
# cons: 1) computationally expensive.
#       2) it has the problem of vanishing gradient.
#       3) it saturates gradients. At both positive and negative ends, the value of the gradient saturates at 0.
#          That means for those values, the gradient will be 0 or close to 0,
#          which simply means no learning in backpropagation.
def tanh_gradient(x):
    return 1 - np.square(np.tanh(x))


# Swish activation function
# f(x) = x * sig(x)
#
# pros: 1) Swish is unbounded above. This means that for very large values,
#          the outputs do not saturate to the maximum value.
#       2) it is bounded below. This means as the input tends to negative infinity,
#          the output tends to some constant. With this power,
#          Swish forgets the very large negative values which are nothing but the deactivations.
#          This feature of Swish introduces regularization in the model.
#       3) it is non-monotonic. Because of this feature, it is possible for the output to still fall
#          even if the input increases. This in return increases the information storage capacity of the model
#          and of course the discriminative capacity.
#       4) it is a smooth function. This allows the optimizer to go through fewer oscillations,
#          which helps in faster convergence, effective optimization and generalization.
#
# cons: 1) computationally expensive.
def swish(x):
    return np.multiply(x, sigmoid(x))


# swish activation function gradient
# df(x) = sig(x) + (x * dsig(x))
#
# pros: -
#
# cons: 1) computationally expensive.
def swish_gradient(x):
    return np.add(sigmoid(x), np.multiply(x, sigmoid_gradient(x)))


# ReLu activation function
# ReLU should be used in the hidden layers.
# f(x) = max(0, x)
#
# pros: 1) computationally effective.
#       2) it is unbounded on the positive side, hence removing the problem of gradient saturation.
#       3) it provides sparsity to the network, which as a result lessens the space and time complexity.
#
# cons: 1) it is not differentiable at 0.
def relu(x):
    return np.maximum(0, x)


# ReLu activation function gradient
# df(x) = { 1 for x > 0
#         { 0 for x < 0
#
# pros: 1) it doesn't suffer from the vanishing gradient problem.
#
# cons: 1) it suffers from the dying ReLU problem.
def relu_gradient(x):
    y = np.ones_like(x)
    y[x < 0] = 0
    return y


# Leaky ReLu activation function
# leaky ReLU should be used in the hidden layers.
# f(x) = max(ax, x)
#
# pros: 1) it tries to remove the dying ReLU problem. Instead of making the negative input 0,
#          which was the case of ReLU, it makes the input value very small but proportional to the input.
#          Because of this, the gradient doesn't saturate to 0.
#
# Cons: 1) the value of α is always constant and is a hyperparameter.
def leaky_relu(x, alpha):
    return np.maximum(alpha * x, x)


# leaky ReLu activation function gradient
# df(x) = { 1 for x > 0
#         { a for x < 0
#
# pros: 1) computationally effective.
#
# cons: -
def leaky_relu_gradient(x, alpha):
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx


# ELU activation function
# f(x) = { x for x >= 0
#        { a(e^x - 1) for x < 0
#
# pros: 1) unlike ReLU, it is derivable at 0.
#       2) it bends smoothly at 0 whereas ReLU's bent was really sharp.
#          This smoothness plays a beneficial role in optimization and generalization.
#       3) the negative values of ELU pushes the mean value towards zero.
#       4) ELU ensures a noise-robust deactivation state.
#          With ELU, in the negative region, the curve is not a straight line because of the exponential term.
#          Because of this, the negative values saturate to some level, and as a result,
#          the model is not impacted more by the noise. So there will be a taste of the negative
#          inputs, but they would not be allowed to create any unbalance in the model.
#       5) because of the above characteristic of ELU, the risk of overfitting is also reduced.
#
# Cons: 1) it is computationally expensive.
#       2) a is a hyperparameter.
#       3) computationally expensive.
def elu(x, alpha):
    rx = np.ravel(x)
    y = np.copy(rx)
    y[rx < 0] = alpha * (np.exp(rx[rx < 0]) - 1)
    return np.array([y]).T


# ELU activation function gradient
# df(x) = { 1 for x >= 0
#         { a * e^x  for x < 0
#
# pros: -
#
# cons: 1) computationally expensive.
def elu_gradient(x, alpha):
    rx = np.ravel(x)
    dx = np.ones_like(rx)
    dx[rx < 0] = alpha * np.exp(rx[rx < 0])
    return np.array([dx]).T


# SELU activation function
# f(x) = lambda * { x for x >= 0
#                 { a(e^x - 1) for x < 0
#
# pros: 1) Like ReLU, SELU does not have a vanishing gradient problem and hence, is used in deep neural networks.
#       2) Compared to ReLUs, SELUs cannot die.
#       3) SELUs learn faster and better than other activation functions without needing further procession.
#
# cons: 1) SELU is a relatively new activation function, so it is not yet used widely in practice.
#       2) computationally expensive.
def selu(x):
    lambd_a = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    return lambd_a * (np.maximum(0, x) + np.minimum(0, alpha * (np.exp(x) - 1)))


# SELU activation function gradient
# df(x) = lambda * { 1 for x >= 0
#                  { a * e^x for x < 0
#
# pros: -
#
# cons: 1) computationally expensive.
def selu_gradient(x):
    lambd_a = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    rx = np.ravel(x)
    y = np.copy(rx)
    y[rx > 0] = lambd_a
    y[rx <= 0] = lambd_a * alpha * np.exp(rx[rx <= 0])
    return np.array([y]).T


# sigmoid activation function
# it is commonly used for models where we have to predict the probability as an output.
# f(x) = 1 / (1 + e^-x)
#
# pros: 1) the output values are bound between 0 and 1. This means the activations will not be blown up.
#
# cons: 1) computationally expensive.
#       2) the outputs are not zero centered.
def sigmoid(x):
    result = np.exp(-np.logaddexp(0., -x))
    result = np.minimum(result, 0.9999)
    result = np.maximum(result, 1e-20)
    return result


# sigmoid activation function gradient
# df(x) = sig(x) * (1 - sig(x))
#
# pros: -
#
# cons: 1) computationally expensive.
#       2) it has the problem of vanishing gradient.
#       3) it saturates gradients.
def sigmoid_gradient(x):
    return np.multiply(sigmoid(x), (np.ones_like(x) - sigmoid(x)))
