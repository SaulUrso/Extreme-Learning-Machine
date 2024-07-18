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
            self.b_in = np.random.randn(hidden_size) / np.sqrt(input_size)
            self.output_weights = np.random.randn(
                hidden_size, output_size
            ) / np.sqrt(hidden_size)
            self.b_out = np.random.randn(output_size) / np.sqrt(hidden_size)

        elif init == "std":
            self.input_weights = np.random.randn(input_size, hidden_size)
            self.b_in = np.random.randn(hidden_size)
            self.output_weights = np.random.randn(hidden_size, output_size)
            self.b_out = np.random.randn(output_size)

        else:
            raise ValueError("Invalid initialization type.")

    def tanh(self, x):
        return np.tanh(x)

    def predict(self, X):
        return (
            self.tanh(X @ self.input_weights + self.b_in) @ self.output_weights
            + self.b_out
        )


def compute_loss(y_true, y_pred, alpha=0):
    """
    Compute the loss between the true labels and predicted labels.

    Parameters:
    - y_true: numpy array, true labels
    - y_pred: numpy array, predicted labels
    - alpha: float, regularization parameter (default: 0)

    Returns:
    - loss: float, computed loss value
    """
    return (
        np.linalg.norm(y_true - y_pred, "fro") ** 2
        + alpha * np.linalg.norm(y_pred, "fro") ** 2
    )
