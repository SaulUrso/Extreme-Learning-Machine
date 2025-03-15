# implementation of Nesterov accelerated gradient descent

import numpy as np
import modelutils as mu
import time


def nag(
    model: mu.ELM,
    X,
    Y,
    lr="auto",  # the stepsize
    alpha=np.float64(0),
    beta="schedule",
    max_epochs=1000,
    eps=np.float64(1e-6),  # stopping criterion for gradient norm
    prec_error=0,  # never used, adds small number to denominator for zero division
    exact_solution=None,
    verbose=False,  # used for debugging
    check_float64=False,  # used for debugging
    fast_mode=False,  # used when computing execution time
):
    """
    Perform Nesterov accelerated gradient descent with exact line search for quadratic functions.

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
            raise ValueError("X and Y must be of type np.float64")

    if check_float64:
        if (
            model.input_weights.dtype != np.float64
            or model.b_in.dtype != np.float64
            or model.output_weights.dtype != np.float64
        ):
            raise ValueError("Model weights must be of type np.float64")

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

    loss_train_history = []
    sol_dist_history = []
    grad_history = []

    # initialize the velocity
    v = np.zeros_like(model.output_weights)

    # compute BtB and BtY, which are fixed throughout the training
    A = model.hidden_activations(X)

    AtA = A.T @ A  # + alpha * np.eye(model.hidden_size)

    # guarantee posdef in case of numerical errors (happens for large hidden sizes)
    # NOTE: investigate what gradient descent does, however alpha propably guarantees always posdef
    eig_min = np.min(np.linalg.eigvalsh(AtA))
    old_tau = max(0, -eig_min)
    BtB = AtA + (old_tau + alpha) * np.eye(model.hidden_size)

    # check BtB is float64
    if BtB.dtype != np.float64:
        raise ValueError("BtB must be of type np.float64")

    BtY = A.T @ Y

    # check if there is a problem with the model
    has_problem = False
    eigenvalues = None

    if beta == "schedule":
        sched = 1
        true_beta = 0
    elif beta == "optimal":
        eigenvalues = np.linalg.eigvalsh(BtB)
        L = np.max(eigenvalues)
        tau = np.min(eigenvalues)

        true_beta = np.float64(
            (np.sqrt(L) - np.sqrt(tau)) / (np.sqrt(L) + np.sqrt(tau))
        )
        if verbose:
            print(true_beta)
    else:
        true_beta = beta

    if lr == "optimal":
        if eigenvalues is None:
            eigenvalues = np.linalg.eigvalsh(BtB)
        L = np.max(eigenvalues)
        opt_lr = np.float64(1 / L)
        if verbose:
            print("Optimal beta: ", true_beta, " Optimal lr: ", opt_lr)
        lr = opt_lr

    if verbose:
        print("Training model using Nesterov accelerated gradient descent...")

    start_time = time.process_time()

    for epoch in range(max_epochs):

        # compute the true gradient
        true_grad = model.compute_gradient(BtB=BtB, BtY=BtY)
        true_grad_norm = np.linalg.norm(true_grad, "fro")
        grad_history.append(true_grad_norm)

        # check for convergence
        if true_grad_norm < eps:
            if verbose:
                print(f"Converged at epoch {epoch + 1}")
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

        if not update_grad.dtype == np.float64:
            raise ValueError("update_grad must be of type np.float64")

        if lr == "auto":  # exact line search
            stepsize = np.linalg.norm(update_grad, "fro") ** 2 / (
                np.trace(update_grad.T @ BtB @ update_grad) + prec_error
            )
        elif lr == "col":  # exact line search on all columns
            col_norms = np.einsum("ij,ji->i", update_grad.T, update_grad)
            col_BtB_norms = np.diag(update_grad.T @ BtB @ update_grad)
            stepsize = col_norms / (col_BtB_norms + prec_error)
        else:
            stepsize = lr

        if beta == "schedule":
            prec_sched = sched
            sched = (1 + np.sqrt(1 + 4 * (sched**2))) / 2
            true_beta = (prec_sched - 1) / sched

        if verbose:
            print("computing momentum")

        # compute momentum
        v = true_beta * v - stepsize * update_grad

        if verbose:
            print(type(v))

        # update the weights
        model.output_weights += v

        if fast_mode:  # skipping computation of loss for faster execution
            continue

        # compute distance from exact solution if not none
        if exact_solution is not None:
            sol_dist = np.linalg.norm(model.output_weights - exact_solution, "fro")
            sol_dist_history.append(sol_dist)

        # compute the loss
        loss_train = mu.compute_loss(Y, model.predict(A=A), alpha)

        if check_float64:
            if not loss_train.dtype == np.float64:
                raise ValueError("Loss must be of type np.float64")

        # happens when loss explodes
        if np.isnan(loss_train):
            has_problem = True
            print("Warning: NaN loss encountered")
            break

        # save the loss history
        loss_train_history.append(loss_train)

        if verbose:
            print(
                f"Epoch {epoch + 1}: \t train loss = {loss_train:.8f}, \tgrad norm = {np.linalg.norm(true_grad, 'fro'):.8f}"
            )

    end_time = time.process_time()

    if fast_mode:  # need to compute the final values since they were never  computed
        loss_train = mu.compute_loss(Y, model.predict(A=A), alpha)
        loss_train_history.append(loss_train)
        if exact_solution is not None:
            sol_dist = np.linalg.norm(model.output_weights - exact_solution, "fro")
            sol_dist_history.append(sol_dist)

    return (
        model,
        loss_train_history,
        sol_dist_history,
        grad_history,
        epoch + 1,
        end_time - start_time,
        has_problem,
    )
