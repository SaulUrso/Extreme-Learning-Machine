# implementation of Nesterov accelerated gradient descent

import numpy as np
import modelutils as mu


def nag(
    model: mu.ELM,
    X,
    Y,
    X_val,
    Y_val,
    lr="auto",
    alpha=np.float64(0),
    beta="schedule",
    max_epochs=1000,
    eps=np.float64(1e-6),
    verbose=False,
    check_float64=False,
    fast_mode=False,
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
            not alpha.dtype == np.float64
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

    # compute hidden activation for validation set
    A_val = model.hidden_activations(X_val)

    # check if there is a problem with the model
    has_problem = False

    if beta == "schedule":
        sched = 1
        true_beta = 0
    else:
        true_beta = beta

    for epoch in range(max_epochs):

        # compute the true gradient
        true_grad = model.compute_gradient(BtB=BtB, BtY=BtY)

        # check for convergence
        if np.linalg.norm(true_grad, "fro") < eps:
            if verbose:
                print(f"Converged at epoch for true grad {epoch + 1}")
            break

        # happens when gradient explodes
        if np.isnan(true_grad).any():
            has_problem = True
            print("Warning: NaN gradient encountered")
            break

        # compute the update gradient
        update_grad = model.compute_gradient(
            W_out=model.output_weights + true_beta * v,
            BtB=BtB,
            BtY=BtY,
        )

        if lr == "auto":  # exact line search
            stepsize = np.linalg.norm(update_grad, "fro") ** 2 / (
                np.trace(update_grad.T @ BtB @ update_grad) + 1e-8
            )
        elif lr == "col":  # exact line search on all columns
            # obtain 2-norm squared for each column
            col_norms = np.einsum("ij,ji->i", update_grad.T, update_grad)
            col_BtB_norms = np.diag(update_grad.T @ BtB @ update_grad)
            # col_BtB_norms = np.einsum(
            #    "ij,ji->i", update_grad.T, BtB @ update_grad
            # )
            stepsize = col_norms / (col_BtB_norms + 1e-8)
        else:
            stepsize = lr

        if beta == "schedule":
            prec_sched = sched
            sched = (1 + np.sqrt(1 + 4 * sched**2)) / 2
            true_beta = (prec_sched - 1) / sched

        # compute momentum
        v = true_beta * v - stepsize * update_grad

        # update the weights
        model.output_weights += v

        if fast_mode:
            continue

        # compute the loss
        loss_train = mu.compute_loss(Y, model.predict(A=A), alpha)
        loss_val = mu.compute_loss(Y_val, model.predict(A=A_val), alpha)

        if check_float64:
            if (
                not loss_train.dtype == np.float64
                or not loss_val.dtype == np.float64
            ):
                raise ValueError("Losses must be of type np.float64")

        # happens when loss explodes
        if np.isnan(loss_train) or np.isnan(loss_val):
            has_problem = True
            print("Warning: NaN loss encountered")
            break

        # save the loss history
        loss_train_history.append(loss_train)
        loss_val_history.append(loss_val)

        if verbose:
            print(
                f"Epoch {epoch + 1}: \t train loss = {loss_train:.8f}, \t val loss = {loss_val:.8f}, \tgrad norm = {np.linalg.norm(true_grad, 'fro'):.8f}"  # noqa
            )

    if fast_mode:
        loss_train = mu.compute_loss(Y, model.predict(A=A), alpha)
        loss_val = mu.compute_loss(Y_val, model.predict(A=A_val), alpha)
        loss_train_history.append(loss_train)
        loss_val_history.append(loss_val)

    return model, loss_train_history, loss_val_history, epoch + 1, has_problem
