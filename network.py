import numpy as np


def sigmoid(arr):
    """
    Applies sigmoid function to all elements in a numpy array
    f(s) = 1/(1+exp(-s))
    @param arr: Input array
    @return: Output array
    """
    return np.array(list(map(lambda s: 1/(1+np.exp(-s)), arr)))


def sigmoid_derivative(arr):
    """
    Applies derivative of sigmoid function to all elements in a numpy array with applied sigmoid
    f'(u) = u*(1-u)
    @param arr: Input array (outputs from sigmoid function)
    @return: Output array
    """
    return np.array(list(map(lambda u: u*(1-u), arr)))


class Network:
    def __init__(self, neurons, input_size):
        """
        Initializes the neural network with given parameters
        @param neurons: Tuple containing the number of neurons in consecutive layers
        @param input_size: Size of input vectors
        """
        self.neurons = neurons
        self.input_size = input_size
        self.W = []   #network weights
        self.B = []   #network biases

    def fit(self, iterations, input_data, input_labels):
        """
        Trains network with given training data and corresponding labels
        @param iterations: Number of total iterations the networks
        @param input_data: List of training data vectors with shape (input_size, 1)
        @param input_labels: List of training labels with shape (neurons[-1], 1)
        """
        self.W = [np.random.rand(self.neurons[0], self.input_size) - 0.5,
                  np.random.rand(self.neurons[1], self.neurons[0]) - 0.5,
                  np.random.rand(self.neurons[2], self.neurons[1]) - 0.5]
        self.B = [np.random.rand(self.neurons[0], 1),
                  np.random.rand(self.neurons[1], 1),
                  np.random.rand(self.neurons[2], 1)]
        S = [np.zeros((self.neurons[0], 1)),
             np.zeros((self.neurons[1], 1)),
             np.zeros((self.neurons[2], 1))]  #outputs of layers
        U = [np.zeros((self.neurons[0], 1)),
             np.zeros((self.neurons[1], 1)),
             np.zeros((self.neurons[2], 1))]  #outputs of layers with applied sigmoid
        D = [np.zeros((self.neurons[0], 1)),
             np.zeros((self.neurons[1], 1)),
             np.zeros((self.neurons[2], 1))]  #backpropagated error

        ro = 1
        F = [np.zeros((self.neurons[0], 1)),
             np.zeros((self.neurons[1], 1)),
             np.zeros((self.neurons[2], 1))]  #derivatives of sigmoid function

        for i in range(iterations):
            E = np.array(input_data[i % len(input_data)]).reshape(35, 1)
            C = np.array(input_labels[i % len(input_data)]).reshape(17, 1)
            S[0] = self.W[0] @ E + self.B[0]
            U[0] = sigmoid(S[0])
            S[1] = self.W[1] @ U[0] + self.B[1]
            U[1] = sigmoid(S[1])
            S[2] = self.W[2] @ U[1] + self.B[2]
            U[2] = sigmoid(S[2])

            for j in range(len(F)):
                F[j] = sigmoid_derivative(U[j])

            D[2] = (C - U[2]) * F[2]
            D[1] = (self.W[2].transpose() @ D[2]) * F[1]
            D[0] = (self.W[1].transpose() @ D[1]) * F[0]

            self.W[2] = self.W[2] + ro * np.multiply(np.tile(D[2], (1, self.W[2].shape[1])),
                                                     np.tile(U[1].transpose(), (self.W[2].shape[0], 1)))
            self.B[2] = self.B[2] + ro * D[2]
            self.W[1] = self.W[1] + ro * np.multiply(np.tile(D[1], (1, self.W[1].shape[1])),
                                                     np.tile(U[0].transpose(), (self.W[1].shape[0], 1)))
            self.B[1] = self.B[1] + ro * D[1]
            self.W[0] = self.W[0] + ro * np.multiply(np.tile(D[0], (1, self.W[0].shape[1])),
                                                     np.tile(E.transpose(), (self.W[0].shape[0], 1)))
            self.B[0] = self.B[0] + ro * D[0]

    def evaluate(self, input_vector) -> int:
        """
        Evaluates given input vector using the weights obtained during the training process
        @param input_vector: Input vector with shape (input_size, 1)
        @return: Index of a class assigned by the network
        """
        input_vector = np.array(input_vector).reshape(35, 1)
        U = sigmoid(self.W[0] @ input_vector + self.B[0])
        U = sigmoid(self.W[1] @ U + self.B[1])
        U = sigmoid(self.W[2] @ U + self.B[2])

        index = int(U.argmax(axis=0))
        print(str(int(U.argmax(axis=0))) + ": " + str(float(U[index]) * 100))
        print()
        return index
