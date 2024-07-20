# implementation of Nesterov accelerated gradient descent

import numpy as np
import modelutils as mu


def nag(
    model: mu.ELM,
    X,
    Y,
    X_val,
    Y_val,
    lr=np.float64(0.01),
    alpha=np.float64(0),
    beta=np.float64(0),
    max_epochs=1000,
    eps=np.float64(1e-6),
    verbose=False,
    check_float64=False,
):
    """
    Perform Nesterov accelerated gradient descent.

    Parameters:
    - model: ELM object, the model to be trained
    - X: numpy matrix, features
    - Y: numpy matrix, targets
    - lr: float, learning rate (default: 0.01)
    - alpha: float, regularization parameter (default: 0)
    - beta: float, momentum parameter (default: 0)
    - max_epochs: int, maximum number of epochs (default: 1000)
    - eps: float, convergence threshold (default: 1e-6)
    - verbose: bool, whether to print training information (default: False)

    Returns:
    - model: ELM object, the trained model
    - loss_history: list, history of loss values
    """

    if check_float64:
        if X.dtype != np.float64 or Y.dtype != np.float64:
            raise ValueError("X and Y must be of type np.float")

    if check_float64:
        if X_val.dtype != np.float64 or Y_val.dtype != np.float64:
            raise ValueError("X_val and Y_val must be of type np.float")

    if check_float64:
        if (
            model.input_weights.dtype != np.float64
            or model.b_in.dtype != np.float64
            or model.output_weights.dtype != np.float64
        ):
            raise ValueError("Model weights must be of type np.float")

    # check if parameters are of the correct type
    if check_float64:
        if (
            not lr.dtype == np.float64
            or not alpha.dtype == np.float64
            or not beta.dtype == np.float64
            or not isinstance(max_epochs, int)
            or not eps.dtype == np.float64
        ):
            raise ValueError(
                "lr, alpha, beta, max_epochs and eps must be of type np.float64 and int"
            )

    if verbose:
        print("Training model using Nesterov accelerated gradient descent...")

    loss_train_history = []
    loss_val_history = []

    # initialize the velocity
    v = np.zeros_like(model.output_weights)

    # compute BtB and BtY, which are fixes throughout the training
    A = model.hidden_activations(X)

    BtB = A.T @ A + alpha * np.eye(model.hidden_size)
    BtY = A.T @ Y

    for epoch in range(max_epochs):

        true_grad = model.compute_gradient(BtB=BtB, BtY=BtY)

        if np.linalg.norm(true_grad, "fro") < eps:
            if verbose:
                print(f"Converged at epoch for true grad {epoch + 1}")
            break

        # happens when gradient explodes
        if np.isnan(true_grad).any():
            print("Warning: NaN gradient encountered")
            break

        update_grad = model.compute_gradient(
            W_out=model.output_weights + beta * v,
            BtB=BtB,
            BtY=BtY,
        )

        # stop when gradient small enough
        # if np.linalg.norm(grad, "fro") < eps:
        #    if verbose:
        #        print(f"Converged at epoch {epoch + 1}")
        #    break

        # compute velocity
        v = beta * v - lr * update_grad

        # update the weights
        model.output_weights += v

        # compute the loss
        loss_train = mu.compute_loss(Y, model.predict(X), alpha)
        loss_val = mu.compute_loss(Y_val, model.predict(X_val))

        if check_float64:
            if (
                not loss_train.dtype == np.float64
                or not loss_val.dtype == np.float64
            ):
                raise ValueError("Losses must be of type np.float64")

        if np.isnan(loss_train) or np.isnan(loss_val):
            print("Warning: NaN loss encountered")
            break

        loss_train_history.append(loss_train)
        loss_val_history.append(loss_val)

        if verbose:
            print(
                f"Epoch {epoch + 1}: \t train loss = {loss_train:.8f}, \t val loss = {loss_val:.8f}, \tgrad norm = {np.linalg.norm(true_grad, 'fro'):.8f}"  # noqa
            )

    return model, loss_train_history, loss_val_history
