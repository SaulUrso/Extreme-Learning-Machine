# the model, a 1 layer extreme learning machine

import numpy as np
from backfwd import solve_system


class ELM:

    def __init__(self, input_size, hidden_size, output_size=3, seed=0, init="fan-in"):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        np.random.seed(seed)

        if init == "fan-in":
            self.input_weights = np.random.randn(input_size, hidden_size) / np.sqrt(
                input_size
            )
            self.b_in = np.random.randn(hidden_size) / np.sqrt(input_size)
            self.output_weights = np.random.randn(hidden_size, output_size) / np.sqrt(
                hidden_size
            )

        elif init == "std":
            self.input_weights = np.random.randn(input_size, hidden_size)
            self.b_in = np.random.randn(hidden_size)
            self.output_weights = np.random.randn(hidden_size, output_size)
        else:
            raise ValueError("Invalid initialization type.")
        assert self.input_weights.shape == (
            input_size,
            hidden_size,
        ), "Input weights matrix has incorrect dimensions."
        assert self.b_in.shape == (
            hidden_size,
        ), "Bias vector has incorrect dimensions."
        assert self.output_weights.shape == (
            hidden_size,
            output_size,
        ), "Output weights matrix has incorrect dimensions."

    def tanh(self, x):
        return np.tanh(x)

    def predict(self, x=None, A=None):
        if A is None:
            if x is None:
                raise ValueError("x must be provided.")
            A = self.tanh(x.dot(self.input_weights) + self.b_in)
        return A.dot(self.output_weights)

    def hidden_activations(self, x):
        return self.tanh(x.dot(self.input_weights) + self.b_in)

    def compute_gradient(self, X=None, Y=None, alpha=0, W_out=None, BtB=None, BtY=None):
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

    def compute_wout_system(self, X, Y, alpha=0):

        A = self.hidden_activations(X)
        M = np.matmul(A.T, A) + alpha * np.eye(self.hidden_size)
        B = np.matmul(A.T, Y)
        self.output_weights = solve_system(M, B)

    def condition_number_m(self, X, alpha=0):
        A = self.hidden_activations(X)
        M = np.matmul(A.T, A) + alpha * np.eye(self.hidden_size)
        condition_number = np.linalg.cond(M, 2)
        return condition_number

    def compute_wout_system_np(self, X, Y, alpha=0):
        A = self.hidden_activations(X)
        M = np.matmul(A.T, A) + alpha * np.eye(self.hidden_size)

        B = np.matmul(A.T, Y)
        self.output_weights = np.linalg.solve(M, B)

    def compute_wout_system_qr(self, X, Y, alpha=0):
        """
        Compute the output weights using QR decomposition with L2 regularization.

        Args:
            X (ndarray): Input data (N x d)
            Y (ndarray): Target output (N x M)
            alpha (float): L2 regularization parameter

        Updates:
            self.output_weights (hidden_size x output_size)
        """
        A = self.hidden_activations(X)  # Compute hidden activations H
        Q, R = np.linalg.qr(A)  # QR decomposition of H

        # Compute (R^T R + alpha * I) β = R^T Q^T Y
        RtY = R.T @ Q.T @ Y
        RtR = R.T @ R + alpha * np.eye(self.hidden_size)

        # Solve for output weights
        self.output_weights = np.linalg.solve(RtR, RtY)

    def computewoutsystem(self, X, Y, alpha=0):
        A = self.hidden_activations(X)
        M = np.matmul(A.T, A) + alpha * np.eye(self.hidden_size)
        B = np.matmul(A.T, Y)

        self.output_weights = solve_system(M, B)


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


def compute_variance(y_true, y_pred):
    """
    Compute the variance between the true labels and predicted labels.

    Parameters:
    - y_true: numpy array, true labels
    - y_pred: numpy array, predicted labels

    Returns:
    - variance: float, computed variance value
    """
    return np.var(y_true - y_pred)
