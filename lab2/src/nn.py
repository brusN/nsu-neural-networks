import numpy as np
from layer import sigmoid


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def feed_forward(self, x):
        for layer in self.layers:
            x = layer.activate(x)
        return x

    def predict(self, x):
        return self.feed_forward(x)

    def backpropagation(self, x, y, learning_rate):
        output = self.feed_forward(x)
        self.layers[-1].error = y - output
        self.layers[-1].delta = self.layers[-1].error * sigmoid_derivative(output)
        for i in reversed(range(len(self.layers) - 1)):
            layer = self.layers[i]
            next_layer = self.layers[i + 1]
            layer.error = np.dot(next_layer.weights, next_layer.delta)
            layer.delta = layer.error * sigmoid_derivative(layer.last_activation).T

        self.layers[0].weights += self.layers[0].delta.T * x * learning_rate
        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            input_to_use = np.atleast_2d(self.layers[i - 1].last_activation)
            layer.weights += learning_rate * input_to_use.T * layer.delta.T

    def train(self, x, y, learning_rate, max_epochs):
        mses = []
        for i in range(max_epochs):
            for j in range(len(x)):
                self.backpropagation(x[j], y[j], learning_rate)
            mses.append(self.mse_per_epoch(i, x, y))
        return mses

    def mse_per_epoch(self, epoch, x, y):
        mses = []
        for i in range(len(x)):
            mses.append((y[i] - self.feed_forward(x[i]))**2)
        mse = float(sum(mses) / len(mses))
        print('Epoch: №%s | MSE: %f' % (epoch, mse))
        return mse

    def count_r_squared(self, x, y):
        result = 0.0
        total = np.sum(np.square(y - np.mean(y)))
        for i in range(len(x)):
            result += (y[i] - self.feed_forward(x[i]))**2
        print('R^2= %f' % (1 - result / total))
