import numpy as np


class Functions:
    @staticmethod
    def sigmoid(arr):
        """
        Applies sigmoid function to all elements in a numpy array
        f(s) = 1/(1+exp(-s))
        @param arr: Input array
        @return: Output array
        """
        return np.array(list(map(lambda s: 1 / (1 + np.exp(-s)), arr)))

    @staticmethod
    def sigmoid_derivative(arr):
        """
        Applies derivative of sigmoid function to all elements in a numpy array with applied sigmoid
        f'(u) = u*(1-u)
        @param arr: Input array (outputs from sigmoid function)
        @return: Output array
        """
        return np.array(list(map(lambda u: u * (1 - u), arr)))
