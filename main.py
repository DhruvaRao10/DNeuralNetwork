import numpy as np
import matplotlib.pyplot as plt


# Initializing weights and bias


def initParams(layer_dims):
    np.random.seed(3)
    params = {}
    sz = len(layer_dims)

    for i in range(1, sz):
        params["W" + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * 0.01
        params["b" + str(i)] = np.zeros((layer_dims[i], 1))

    return params


# sigmoid activation fun shapes the output of the neuron
# Z (linear hypothesis) - Z = W*X + b ,
# W - weight matrix, b- bias vector, X- Input


def sigmoid(Z):
    A = 1 / (1 + np.exp(np.dot(-1, Z)))
    temp = Z
    return A, temp


# forward prop takes training data and params from previous layers produces output and feeds it as input to next layers
def forward_prop(X, params):
    A = X
    arr = []
    sz = len(params) // 2
    for i in range(1, sz + 1):
        A_prev = A
        # linear combination of inputs and weights adjusted with bias
        Z = np.dot(params["W" + str(i)], A_prev) + params["b" + str(i)]

        # linear caching
        linear_cache = (A_prev, params["W" + str(i)], params["b" + str(i)])

        # returns new activation by applying sigmoid
        A, activation_cache = sigmoid(Z)

        cache = (linear_cache, activation_cache)
        arr.append(cache)

    return A, arr


# binary cross entropy loss implementation used for binary classficiation problems
# A: output of the NN
# Y: true labels


def cost_function(A, Y):
    # gets number of training examples
    m = Y.shape[1]
    cost = (-1 / m) * (np.dot(np.log(A), Y.T) + np.dot(np.log(1 - A), 1 - Y.T))

    return cost


# dA: gradient of the cost of the previous layer
# dw: dot product of dZ and transpose of A_prev
# db: sum of dZ across training examples
# dA_prev: input for backward pass


def one_layer_backprop(dA, cache):
    linear_cache, activation_cache = cache
    Z = activation_cache
    dZ = dA * sigmoid(Z) * sigmoid(1 - Z)

    A_prev, W, b = linear_cache
    m = A_prev.shape[1]

    # gradient for weights
    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.dot(
        dZ, axis=1, keepdims=True
    )  # keepdims optional helps in reshaping the matrix
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, db, dW


# loop through the model and in backward dir and compute gradient
# Al: ouput of the final layer
# Y: true labels
# caches:
# ∂L/∂p = -[y/p - (1-y)/(1-p)]


def backprop(AL, Y, caches):
    gradients = {}
    sz = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    curr_cache = caches[sz - 1]

    (
        gradients["dA" + str(sz - 1)],
        gradients["dW" + str(sz - 1)],
        gradients["db" + str(sz - 1)],
    ) = one_layer_backprop(dAL, curr_cache)

    for i in reversed(range(sz - 1)):
        curr_cache = caches[sz - 1]
        dA_prev_temp, dW_temp, db_temp = one_layer_backprop(
            gradients["dA" + str(i + 1)], curr_cache
        )

        gradients["dA" + str(i)] = dA_prev_temp
        gradients["dW" + str(i)] = dW_temp
        gradients["db" + str(i)] = db_temp

    return gradients


# using gradient values update  params
# The learning rate (α) determines the step size at each iteration while moving toward a minimum of the cost function
# If α is too small, convergence will be slow.
# If α is too large, the algorithm might overshoot the minimum and fail to converge, or even diverge.

# θ = θ - α * ∇J(θ)


def update_params(params, gradients, learning_rate):
    sz = len(params) // 2

    for i in range(sz):
        params["W" + str(i + 1)] = (
            params["W" + str(i + 1)] - learning_rate * gradients["W" + str(i + 1)]
        )
        params["b" + str(i + 1)] = (
            params["b" + str(i + 1)] - learning_rate * gradients["b" + str(i + 1)]
        )

    return params


# training the DNN
# epochs: number of training iterations
# lr: learning rate


def train(X, Y, layer_dims, epochs, lr):
    params = initParams(layer_dims)
    total_cost = []

    for i in range(epochs):
        Y_pred, caches = forward_prop(X, params)
        cost = cost_function(Y_pred, Y)
        total_cost.append(cost)  # cost of epoch and adds to total cost
        gradients = backprop(Y_pred, Y, caches)  # backprop to compute gradients

        params = update_params(
            params, gradients, lr
        )  # updates network params using gradient and learning rate

    return params, total_cost
