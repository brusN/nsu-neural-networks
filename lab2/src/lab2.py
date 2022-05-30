import matplotlib.pyplot as plt
import numpy as np
from layer import Layer
from nn import NeuralNetwork


def main():
    # Generating sample
    n = 1000
    x_sample = np.random.uniform(-1, 1, size=n)
    y_sample = [np.cos(x) for x in x_sample]

    # ~ Optimal, received as a result of verification
    neurons_per_layer = 40

    # Creating NN
    nn = NeuralNetwork()
    nn.add_layer(Layer(1, neurons_per_layer))
    nn.add_layer(Layer(neurons_per_layer, neurons_per_layer))
    nn.add_layer(Layer(neurons_per_layer, 1))

    # Training sample is 70% of whole sample
    learning_rate = 0.8
    max_epochs = 100
    mses = nn.train(x_sample[:700], y_sample, learning_rate, max_epochs)

    # MSE per epoch
    plt.scatter(np.arange(0, 100), mses)
    plt.show()

    # Approximation graphic
    y_approximation = []
    for x in x_sample[700:]:
        y_approximation.append(nn.predict(x).flatten())
    plt.scatter(x_sample[700:], y_approximation)
    plt.scatter(x_sample[700:], y_sample[700:])
    plt.show()

    # R^2
    nn.count_r_squared(x_sample[700:], y_sample[700:])


if __name__ == '__main__':
    main()
