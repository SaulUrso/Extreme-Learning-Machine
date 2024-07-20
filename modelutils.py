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

    def hidden_activations(self, x):
        return self.tanh(x.dot(self.input_weights) + self.b_in)

    def compute_gradient(
        self, X=None, Y=None, alpha=0, W_out=None, BtB=None, BtY=None
    ):
        """
        Compute the gradient of the model's output with respect to the weights.

        Args:
            X (ndarray, optional): Input data matrix. Defaults to None.
            Y (ndarray, optional): Target output matrix. Defaults to None.
            alpha (float, optional): Regularization parameter. Defaults to 0.
            W_out (ndarray, optional): Output weight matrix. Defaults to None.
            BtB (ndarray, optional): Precomputed matrix A^T @ A + alpha * I. Defaults to None.
            BtY (ndarray, optional): Precomputed matrix A^T @ Y. Defaults to None.

        Returns:
            ndarray: The computed gradient.

        Raises:
            ValueError: If X and Y are not provided.

        Note:
            This method computes the gradient of the model's output with respect to the weights.
            It can be used for accelerated gradient computation by providing precomputed matrix W_out.
            If W_out is provided, the gradient is computed with respect to W_out, otherwise it is computed with respect to self.output_weights.
        """  # noqa: E501

        if BtB is None or BtY is None:
            if X is None or Y is None:
                raise ValueError("X and Y must be provided.")
            A = self.tanh(X @ self.input_weights + self.b_in)

        if BtB is None:
            BtB = A.T @ A + alpha * np.eye(self.hidden_size)

        if BtY is None:
            BtY = A.T @ Y

        # compute gradients w.r.t different weights (useful for accelerated gradient)
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
