# the model, a 1 layer extreme learning machine

import numpy as np


class ELM:

    def __init__(
        self, input_size, hidden_size, output_size=3, seed=0, init="fan_in"
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        np.random.seed(seed)

        if init == "fan_in":
            self.input_weights = np.random.randn(
                input_size, hidden_size
            ) / np.sqrt(input_size)
            self.bias = np.random.randn(hidden_size) / np.sqrt(input_size)
            self.output_weights = np.random.randn(
                hidden_size, output_size
            ) / np.sqrt(hidden_size)

        elif init == "std":
            self.input_weights = np.random.randn(input_size, hidden_size)
            self.bias = np.random.randn(hidden_size)
            self.output_weights = np.random.randn(hidden_size, output_size)

        else:
            raise ValueError("Invalid initialization type.")

    def tanh(self, x):
        return np.tanh(x)

    def predict(self, X):
        return (
            self.sigmoid(X @ self.input_weights + self.bias)
            @ self.output_weights
        )
