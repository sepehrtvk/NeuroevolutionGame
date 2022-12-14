import math
import numpy as np


class NeuralNetwork:

    def __init__(self, layer_sizes):
        """
        Neural Network initialization.
        Given layer_sizes as an input, you have to design a Fully Connected Neural Network architecture here.
        :param layer_sizes: A list containing neuron numbers in each layers. For example [3, 10, 2] means that there are
        3 neurons in the input layer, 10 neurons in the hidden layer, and 2 neurons in the output layer.
        """
        # TODO (Implement FCNNs architecture here)
        self.layer1 = layer_sizes[0]     # number of neurons of layer 1
        self.layer2 = layer_sizes[1]     # number of neurons of layer 2
        self.layer3 = layer_sizes[2]     # number of neurons of layer 3

        self.W1 = np.random.normal(size=(self.layer2, self.layer1))
        self.W2 = np.random.normal(size=(self.layer3, self.layer2))

        # Initialize b = 0, for each layer.
        self.b1 = np.zeros((self.layer2, 1))
        self.b2 = np.zeros((self.layer3, 1))


    def activation(self, x):
        """
        The activation function of our neural network, e.g., Sigmoid, ReLU.
        :param x: Vector of a layer in our network.
        :return: Vector after applying activation function.
        """
        # TODO (Implement activation function here)
        return 1 / (np.exp(-x) + 1)  # sigmoid

    def forward(self, x):
        """
        Receives input vector as a parameter and calculates the output vector based on weights and biases.
        :param x: Input vector which is a numpy array.
        :return: Output vector
        """
        # TODO (Implement forward function here)
        x1 = self.activation(self.W1 @ x + self.b1)
        y = self.activation(self.W2 @ x1 + self.b2)
        return y
