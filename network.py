import numpy as np
import random
from functions import Functions


class Network:
    def __init__(self, neurons, input_size) -> None:
        """
        Initializes the neural network with given parameters
        @param neurons: Tuple containing the number of neurons in consecutive layers
        @param input_size: Size of input vectors
        """
        # print("{ Initializing network with layers of sizes " + str(neurons) + " }")
        self.neurons = neurons
        self.layers = len(neurons)
        self.input_size = input_size
        self.weights = [np.random.rand(self.neurons[0], self.input_size) - 0.5]
        self.biases = [np.random.rand(self.neurons[0], 1)]
        for i in range(self.layers - 1):
            self.weights.append(np.random.rand(self.neurons[i + 1], self.neurons[i]) - 0.5)
            self.biases.append(np.random.rand(self.neurons[i + 1], 1))

    def fit(self, iterations, input_data, input_labels, ro, alpha) -> None:
        """
        Trains network with given training data and corresponding labels
        @param iterations: Number of total iterations the networks
        @param input_data: List of training data vectors with shape (input_size, 1)
        @param input_labels: List of training labels with shape (neurons[-1], 1)
        @param ro:
        @param alpha:
        """
        print("{ Training network for " + str(iterations) + " iterations and " + str(len(input_data)) + " samples }")
        activations = []
        errors = []
        derivatives = []
        for i in range(self.layers):
            activations.append(np.zeros((self.neurons[i], 1)))
            errors.append(np.zeros((self.neurons[i], 1)))
            derivatives.append(np.zeros((self.neurons[i], 1)))
        diff_weights = [np.zeros((self.neurons[0], self.input_size))]
        diff_biases = [np.zeros((self.neurons[0], 1))]
        for i in range(self.layers - 1):
            diff_weights.append(np.zeros((self.neurons[i + 1], self.neurons[i])))
            diff_biases.append(np.zeros((self.neurons[i + 1], 1)))

        samples = list(range(len(input_data)))

        for iteration in range(iterations):
            # if iteration % 500 == 0:
            #       self.progress = ((iteration / iterations)+ (500/ iterations))*100%
            #     print(iteration / iterations)
            if iteration % len(input_data) == 0:
                random.shuffle(samples)
            sample = samples[iteration % len(input_data)]
            input_vector = np.array(input_data[sample]).reshape(self.input_size, 1)
            input_label = np.array(input_labels[sample]).reshape(self.neurons[-1], 1)

            activations[0] = Functions.sigmoid(self.weights[0] @ input_vector + self.biases[0])
            for i in range(self.layers - 1):
                activations[i + 1] = Functions.sigmoid(self.weights[i + 1] @ activations[i] + self.biases[i + 1])

            for i in range(len(derivatives)):
                derivatives[i] = Functions.sigmoid_derivative(activations[i])
            errors[self.layers - 1] = (input_label - activations[self.layers - 1]) * derivatives[self.layers - 1]
            for i in range(self.layers - 2, 0, -1):
                errors[i] = (self.weights[i + 1].transpose() @ errors[i + 1]) * derivatives[i]

            diff_weights[0] = ro * np.multiply(
                np.tile(errors[0], (1, self.weights[0].shape[1])),
                np.tile(input_vector.transpose(), (self.weights[0].shape[0], 1))) + alpha * diff_weights[0]
            self.weights[0] += diff_weights[0]
            diff_biases[0] = ro * errors[0] + alpha * diff_biases[0]
            self.biases[0] += diff_biases[0]
            for i in range(1, self.layers):
                diff_weights[i] = ro * np.multiply(
                    np.tile(errors[i], (1, self.weights[i].shape[1])),
                    np.tile(activations[i - 1].transpose(), (self.weights[i].shape[0], 1))) + alpha * diff_weights[i]
                self.weights[i] += diff_weights[i]
                diff_biases[i] = ro * errors[i] + alpha * diff_biases[i]
                self.biases[i] += diff_biases[i]

    def evaluate(self, input_vector) -> int:
        """
        Evaluates given input vector using the weights obtained during the training process
        @param input_vector: Input vector with shape (input_size, 1)
        @return: Index of a class assigned by the network
        """
        input_vector = np.array(input_vector).reshape(self.input_size, 1)
        activations = Functions.sigmoid(self.weights[0] @ input_vector + self.biases[0])
        for i in range(1, self.layers):
            activations = Functions.sigmoid(self.weights[i] @ activations + self.biases[i])

        index = int(activations.argmax(axis=0))
        print(str(int(activations.argmax(axis=0))) + ": " + str(float(activations[index]) * 100))
        print()
        return index

    def save(self, file_name) -> None:
        """
        Saves network parameters to a csv file.
        @param file_name: The file will be created in ./data/<file_name>.csv
        """
        # print("{ Saving weights to file " + file_name + ".csv }")
        with open(file_name, 'ab') as f:
            for layer in range(self.layers):
                np.savetxt(f, self.weights[layer], delimiter=',')
                np.savetxt(f, self.biases[layer], delimiter=',')

    def load(self, file_name) -> None:
        """
        Loads network parameters from a csv file.
        @param file_name: The file will be loaded from ./data/<file_name>.csv
        """
        # print("{ Loading weights from file " + file_name + ".csv }")
        skip_header = 0
        for layer in range(self.layers):
            self.weights[layer] = np.genfromtxt(file_name, delimiter=",",
                                                skip_header=skip_header, max_rows=self.weights[layer].shape[0])
            skip_header += self.neurons[layer]
            self.biases[layer] = np.genfromtxt(file_name, delimiter=",", skip_header=skip_header,
                                               max_rows=self.biases[layer].shape[0]).reshape((self.neurons[layer], 1))
            skip_header += self.neurons[layer]
