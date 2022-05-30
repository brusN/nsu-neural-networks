import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Layer:
    def __init__(self, n_input, n_neurons):
        self.weights = np.random.rand(n_input, n_neurons)
        self.bias = np.random.rand(n_neurons)
        self.last_activation = None
        self.error = None
        self.delta = None

    def activate(self, x):
        res = np.dot(x, self.weights) + self.bias
        self.last_activation = sigmoid(res)
        return self.last_activation

    def print_weights(self):
        print(self.weights)
