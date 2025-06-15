# Implementation of Nesterov Accelerated Gradient Descent for ELM models

import time

import numpy as np

import modelutils as mu


def nag(
    model: mu.ELM,
    X,
    Y,
    lr="auto", 
    alpha=np.float64(0),  
    beta="schedule", 
    max_epochs=1000, 
    eps=np.float64(1e-6),  
    prec_error=0,  
    exact_solution=None,  
    verbose=False,  # print debug/training info
    check_float64=False,  # check for correct dtype, it is called this way because initially we did not change the precision
    fast_mode=False,  # skip loss computation for speed
    precision=np.float64,  
):
    """
    Perform Nesterov Accelerated Gradient Descent (NAG) to train an ELM model.

    Parameters:
        model (mu.ELM): The ELM model to be trained.
        X (np.ndarray): Input features.
        Y (np.ndarray): Target outputs.
        lr (float or str): Learning rate, 'optimal' for optimal stepsize according to NAG,  or 'auto'/'col' for line search (default: 'auto').
        alpha (float): Regularization parameter (default: 0).
        beta (float or str): Momentum parameter, 'schedule', or 'optimal' (default: 'schedule').
        max_epochs (int): Maximum number of epochs (default: 1000).
        eps (float): Convergence threshold for gradient norm (default: 1e-6).
        prec_error (float): Small value to avoid division by zero (default: 0).
        exact_solution (np.ndarray, optional): Known solution for monitoring convergence (default: None).
        verbose (bool): Whether to print training progress (default: False).
        check_float64 (bool): Whether to check for correct dtype (default: False).
        fast_mode (bool): If True, skips loss computation for speed (default: False).
        precision (np.dtype): Numpy dtype for computation precision (default: np.float64).

    Returns:
        model (mu.ELM): The trained model.
        loss_train_history (list): Training loss at each epoch.
        sol_dist_history (list): Distance to exact solution at each epoch (if provided).
        grad_history (list): Gradient norm at each epoch.
        epoch (int): Number of epochs performed.
        elapsed_time (float): Total training time in seconds.
        has_problem (bool): True if NaN encountered in gradient or loss.
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

    # Check input dtypes if requested
    if check_float64:
        if X.dtype != precision or Y.dtype != precision:
            raise ValueError(f"X and Y must be of type {precision}")

    # Check model weights dtypes if requested
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

    # Check parameter types if requested
    if check_float64:
        if (
            not alpha.dtype == precision
            or not isinstance(max_epochs, int)
            or not eps.dtype == precision
        ):
            raise ValueError(
                f"lr, alpha, beta, max_epochs and eps must be of type {precision} and int"
            )

    # Initialize histories for tracking progress
    loss_train_history = []
    sol_dist_history = []
    grad_history = []

    # Initialize the velocity (momentum term)
    v = np.zeros_like(model.output_weights, dtype=precision)

    # Precompute hidden layer activations and related matrices
    A = model.hidden_activations(X).astype(precision, copy=False)
    AtA = A.T @ A
    AtA = AtA.astype(precision, copy=False)

    # Guarantee positive definiteness for numerical stability
    eig_min = np.min(np.linalg.eigvalsh(AtA))
    old_tau = max(precision(0), -eig_min)
    BtB = AtA + (alpha) * np.eye(model.hidden_size, dtype=precision)
    print(
        old_tau
    )  # sometimes is negative, but a factor of -13, meaning that alpha should make it positive

    # Check BtB dtype
    if BtB.dtype != precision:
        raise ValueError(f"BtB must be of type {precision}")

    BtY = A.T @ Y
    BtY = BtY.astype(precision, copy=False)

    # Track if any numerical problems occur
    has_problem = False
    eigenvalues = None

    # Determine momentum parameter beta
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

    # Determine learning rate
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

        # Compute the true gradient at current weights
        true_grad = model.compute_gradient(BtB=BtB, BtY=BtY)
        true_grad = true_grad.astype(precision, copy=False)
        true_grad_norm = np.linalg.norm(true_grad, "fro")
        grad_history.append(true_grad_norm)

        # Check for convergence
        if true_grad_norm <= eps:
            if verbose:
                print(f"Converged at epoch {epoch + 1}")
            break

        # Check for NaN in gradient (exploding/unstable)
        if np.isnan(true_grad).any():
            has_problem = True
            print("Warning: NaN gradient encountered")
            break

        # Compute the gradient at the lookahead position (Nesterov update)
        update_grad = model.compute_gradient(
            W_out=model.output_weights + true_beta * v,
            BtB=BtB,
            BtY=BtY,
        )
        update_grad = update_grad.astype(precision, copy=False)

        if not update_grad.dtype == precision:
            raise ValueError(f"update_grad must be of type {precision}")

        # Compute stepsize (learning rate)
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

        # Update momentum parameter if using schedule
        if beta == "schedule":
            prec_sched = sched
            sched = (
                precision(1) + np.sqrt(precision(1) + precision(4) * (sched**2))
            ) / precision(2)
            true_beta = (prec_sched - precision(1)) / sched

        if verbose:
            print("computing momentum")

        # Update velocity (momentum)
        v = true_beta * v - stepsize * update_grad

        if verbose:
            print(type(v))

        # Update model weights
        model.output_weights += v

        if fast_mode:  # skipping computation of loss for faster execution
            continue

        # Compute distance from exact solution if provided
        if exact_solution is not None:
            sol_dist = np.linalg.norm(model.output_weights - exact_solution, "fro")
            sol_dist_history.append(sol_dist)

        # Compute the loss
        loss_train = mu.compute_loss(Y, model.predict(A=A), alpha)
        if hasattr(loss_train, "astype"):
            loss_train = loss_train.astype(precision, copy=False)

        if check_float64:
            if not loss_train.dtype == precision:
                raise ValueError(f"Loss must be of type {precision}")

        # Check for NaN in loss (exploding/unstable)
        if np.isnan(loss_train):
            has_problem = True
            print("Warning: NaN loss encountered")
            break

        # Save loss history
        loss_train_history.append(loss_train)

        if verbose:
            print(
                f"Epoch {epoch + 1}: \t train loss = {loss_train:.8f}, \tgrad norm = {np.linalg.norm(true_grad, 'fro'):.8f}"
            )

    end_time = time.process_time()

    # If fast_mode, compute final loss and solution distance
    if fast_mode:
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
        epoch + 1,
        end_time - start_time,
        has_problem,
    )
