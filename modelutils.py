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

        elif init == "std":
            self.input_weights = np.random.randn(input_size, hidden_size)
            self.b_in = np.random.randn(hidden_size)
            self.output_weights = np.random.randn(hidden_size, output_size)

        else:
            raise ValueError("Invalid initialization type.")

    def tanh(self, x):
        return np.tanh(x)

    def predict(self, x):
        A = self.tanh(x.dot(self.input_weights) + self.b_in)
        return A.dot(self.output_weights)

    def compute_gradient(self, X, Y, alpha=0, W_out=None):
        A = self.tanh(X @ self.input_weights + self.b_in)
        BtB = A.T @ A + alpha * np.eye(self.hidden_size)
        BtY = A.T @ Y
        if W_out is not None:
            grad = BtB @ W_out - BtY
        else:
            grad = BtB @ self.output_weights - BtY
        return grad


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
    ) / y_true.shape[0]
