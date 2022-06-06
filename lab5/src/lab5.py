from keras.datasets import mnist

from src.gan import GAN


def main():
    (x_train, y_train), (_, _) = mnist.load_data()
    print("y_train shape", y_train.shape)
    print("x_train shape", x_train.shape)
    numbers = [1]
    model = GAN(numbers, learning_rate=1e-3, decay_rate=1e-4, epochs=50)
    discriminator_losses, generator_losses = model.train(x_train, y_train)


if __name__ == '__main__':
    main()
