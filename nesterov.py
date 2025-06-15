# implementation of Nesterov accelerated gradient descent

import time

import numpy as np

import modelutils as mu


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
    precision=np.float64,  # added precision parameter
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
    - precision: numpy dtype, precision for computations (default: np.float64)

    Returns:
    - model: ELM object, the trained model
    - loss_history: list, history of loss values
    """

    # Cast X, Y, alpha, eps, etc. to the desired precision
    X = X.astype(precision, copy=False)
    Y = Y.astype(precision, copy=False)
    alpha = precision(alpha)
    eps = precision(eps)
    prec_error = precision(prec_error)
    # If exact_solution is provided, cast it as well
    if exact_solution is not None:
        exact_solution = exact_solution.astype(precision, copy=False)

    if check_float64:
        if X.dtype != precision or Y.dtype != precision:
            raise ValueError(f"X and Y must be of type {precision}")

    if check_float64:
        if (
            model.input_weights.dtype != precision
            or model.b_in.dtype != precision
            or model.output_weights.dtype != precision
        ):
            model.input_weights = model.input_weights.astype(precision, copy=False)
            model.b_in = model.b_in.astype(precision, copy=False)
            model.output_weights = model.output_weights.astype(precision, copy=False)
            if (
                model.input_weights.dtype != precision
                or model.b_in.dtype != precision
                or model.output_weights.dtype != precision
            ):
                raise ValueError(f"Model weights must be of type {precision}")

    # check if parameters are of the correct type
    if check_float64:
        if (
            not alpha.dtype == precision
            or not isinstance(max_epochs, int)
            or not eps.dtype == precision
        ):
            raise ValueError(
                f"lr, alpha, beta, max_epochs and eps must be of type {precision} and int"
            )

    loss_train_history = []
    sol_dist_history = []
    grad_history = []

    # For stationary gradient check using moving window means
    # stationary_patience = 1000  # Number of epochs in each window
    # stationary_tol = 1e-3  # Relative tolerance for mean gradient change
    # grad_window = []

    # initialize the velocity
    v = np.zeros_like(model.output_weights, dtype=precision)

    # compute BtB and BtY, which are fixed throughout the training
    A = model.hidden_activations(X).astype(precision, copy=False)

    AtA = A.T @ A  # + alpha * np.eye(model.hidden_size)
    AtA = AtA.astype(precision, copy=False)

    # guarantee posdef in case of numerical errors (happens for large hidden sizes)
    # NOTE: investigate what gradient descent does, however alpha propably guarantees always posdef
    eig_min = np.min(np.linalg.eigvalsh(AtA))
    old_tau = max(precision(0), -eig_min)
    BtB = AtA + (alpha) * np.eye(model.hidden_size, dtype=precision)  # + old_tau
    print(
        old_tau
    )  # sometimes is negative, but a factor of -13, meaning that alpha should make it positive

    # check BtB is correct precision
    if BtB.dtype != precision:
        raise ValueError(f"BtB must be of type {precision}")

    BtY = A.T @ Y
    BtY = BtY.astype(precision, copy=False)

    # check if there is a problem with the model
    has_problem = False
    eigenvalues = None

    if beta == "schedule":
        sched = precision(1)
        true_beta = precision(0)
    elif beta == "optimal":
        eigenvalues = np.linalg.eigvalsh(BtB)
        L = np.max(eigenvalues)
        tau = np.min(eigenvalues)

        true_beta = precision((np.sqrt(L) - np.sqrt(tau)) / (np.sqrt(L) + np.sqrt(tau)))

        print(f"Optimal Beta: {true_beta}")
    else:
        true_beta = (
            precision(beta) if isinstance(beta, (float, int, np.floating)) else beta
        )

    if lr == "optimal":
        if eigenvalues is None:
            eigenvalues = np.linalg.eigvalsh(BtB)
        L = np.max(eigenvalues)
        opt_lr = precision(1 / L)
        lr = opt_lr
        print(f"Optimal lr: {lr}")

    if verbose:
        print("Training model using Nesterov accelerated gradient descent...")

    start_time = time.process_time()

    for epoch in range(max_epochs):

        # compute the true gradient
        true_grad = model.compute_gradient(BtB=BtB, BtY=BtY)
        true_grad = true_grad.astype(precision, copy=False)
        true_grad_norm = np.linalg.norm(true_grad, "fro")
        grad_history.append(true_grad_norm)

        # check for convergence
        if true_grad_norm <= eps:
            if verbose:
                print(f"Converged at epoch {epoch + 1}")
            break

        # # Track gradient norms in a moving window for stationarity
        # grad_window.append(true_grad_norm)
        # if len(grad_window) > 2 * stationary_patience:
        #     grad_window.pop(0)
        # if len(grad_window) == 2 * stationary_patience:
        #     mean1 = np.mean(grad_window[:stationary_patience])
        #     mean2 = np.mean(grad_window[stationary_patience:])
        #     rel_change = abs(mean2 - mean1) / (abs(mean1) + 1e-14)
        #     if rel_change < stationary_tol:
        #         if verbose:
        #             print(
        #                 f"Stopped at epoch {epoch + 1} due to stationary gradient (mean grad change {rel_change:.2e} < tol {stationary_tol}) over {2*stationary_patience} epochs"
        #             )
        #         break

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
        update_grad = update_grad.astype(precision, copy=False)

        if not update_grad.dtype == precision:
            raise ValueError(f"update_grad must be of type {precision}")

        if lr == "auto":  # exact line search
            stepsize = np.linalg.norm(update_grad, "fro") ** 2 / (
                np.trace(update_grad.T @ BtB @ update_grad) + prec_error
            )
            stepsize = precision(stepsize)
        elif lr == "col":  # exact line search on all columns
            col_norms = np.einsum("ij,ji->i", update_grad.T, update_grad)
            col_BtB_norms = np.diag(update_grad.T @ BtB @ update_grad)
            stepsize = col_norms / (col_BtB_norms + prec_error)
            stepsize = stepsize.astype(precision, copy=False)
        else:
            stepsize = precision(lr)

        if beta == "schedule":
            prec_sched = sched
            sched = (
                precision(1) + np.sqrt(precision(1) + precision(4) * (sched**2))
            ) / precision(2)
            true_beta = (prec_sched - precision(1)) / sched

        if verbose:
            print("computing momentum")

        # compute momentum
        v = true_beta * v - stepsize * update_grad
        # v_history.append(np.median(np.abs(v)))

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
        if hasattr(loss_train, "astype"):
            loss_train = loss_train.astype(precision, copy=False)

        if check_float64:
            if not loss_train.dtype == precision:
                raise ValueError(f"Loss must be of type {precision}")

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
        if hasattr(loss_train, "astype"):
            loss_train = loss_train.astype(precision, copy=False)
        loss_train_history.append(loss_train)
        if exact_solution is not None:
            sol_dist = np.linalg.norm(model.output_weights - exact_solution, "fro")
            sol_dist_history.append(sol_dist)

    return (
        model,
        loss_train_history,
        sol_dist_history,
        grad_history,
        # v_history,
        epoch + 1,
        end_time - start_time,
        has_problem,
    )
