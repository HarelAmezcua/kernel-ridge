import numpy as np
import matplotlib.pyplot as plt


def kernel_ridge_regression(
    X, 
    y, 
    kernel='RBF', 
    param=1.0, 
    reg=0.1, 
    X_test=None,
    add_bias=False
):
    """
    Perform Kernel Ridge Regression on 1D inputs with multiple kernel options.

    Parameters
    ----------
    X : array-like of shape (n_samples,) or (n_samples, 1)
        Training input samples.
    y : array-like of shape (n_samples,)
        Target (desired) values.
    kernel : {'linear', 'rbf', 'poly', 'tanh'}, default='rbf'
        Which kernel to use:
            - 'linear': K(x, x') = x * x'
            - 'rbf':    K(x, x') = exp(-param * (x - x')^2)
            - 'poly':   K(x, x') = (1 + x*x')^param
            - 'tanh':   K(x, x') = tanh(param * x*x')
    param : float, default=1.0
        Kernel parameter:
            - For 'rbf', this is the "gamma"-like coefficient (NOT 1 / (2*sigma^2)!).
            -
    reg : float, default=0.1
        L2 regularization parameter (lambda). Larger values add more regularization.
    X_test : array-like of shape (n_test,) or (n_test, 1), optional parameter (lambda). Larger values add more regularization.
    X_test : array-like of shape (n_test,) or (n_test, 1), optional
        If provided, the function returns predictions for these new inputs.
    add_bias : bool, default=False
        If True, add a bias term to the linear kernel.

    Returns
    -------
    alpha : ndarray of shape (n_samples,)
        The fitted dual coefficients for the kernel ridge model
        (only returned if `X_test` is None).
    y_pred : ndarray of shape (n_test,)
        The predicted values at `X_test` (only returned if `X_test` is provided).

    Notes
    -----
    - This example focuses on the 1D feature case (n_samples, 1).
      If you have multi-dimensional data, you can easily adapt it by
      removing shape checks or adjusting the kernel computations.
    - The parameter `param` has different meanings for each kernel,
      so be sure to interpret/tune it accordingly.
    """
    # Convert inputs to numpy arrays with consistent shapes
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Ensure X is (n_samples, 1)
    if X.ndim == 1:
        X = X[:, None]
    # Ensure y is (n_samples,)
    if y.ndim != 1:
        raise ValueError("`y` must be 1-dimensional.")
    
    n_samples = X.shape[0]
    if y.shape[0] != n_samples:
        raise ValueError("`X` and `y` must have the same number of samples.")

    # -- Compute Kernel Matrix on training data --
    if kernel == 'LINEAR':
        # K(x, x') = x * x'
        K = X @ X.T
        if add_bias:
            K += 1
    elif kernel == 'RBF':
        # K(x, x') = exp(-param * ||x - x'||^2)
        # param is analogous to 'gamma' in many libraries
        X_sq = np.sum(X**2, axis=1)  # shape (n,)
        dist_matrix = X_sq[:, None] - 2 * X @ X.T + X_sq[None, :]
        K = np.exp(-param * dist_matrix)
    elif kernel == 'POLY':
        # K(x, x') = (1 + x*x')^param
        K = (1.0 + X @ X.T) ** param
    elif kernel == 'SIGMOID':
        # K(x, x') = tanh(param * x*x')
        K = np.tanh(param * (X @ X.T))
    else:
        raise ValueError(f"Unrecognized kernel: {kernel}")

    # -- Solve for alpha in (K + reg*I)*alpha = y --
    alpha = np.linalg.solve(K + reg * np.eye(n_samples), y)

    # If no test data, return the dual coefficients
    if X_test is None:
        return alpha

    # -- Compute predictions on test data --
    X_test = np.asarray(X_test, dtype=float)
    if X_test.ndim == 1:
        X_test = X_test[:, None]

    if kernel == 'LINEAR':
        # K_test(x_test, x_train) = x_test * x_train'
        K_test = X_test @ X.T
        if add_bias:
            K_test += 1
    elif kernel == 'RBF':
        # K_test(x_test, x_train) = exp(-param * ||x_test - x_train||^2)
        X_test_sq = np.sum(X_test**2, axis=1)  # shape (n_test,)
        dist_matrix_test = (
            X_test_sq[:, None]
            - 2 * X_test @ X.T
            + np.sum(X**2, axis=1)[None, :]
        )
        K_test = np.exp(-param * dist_matrix_test)
    elif kernel == 'POLY':
        K_test = (1.0 + X_test @ X.T) ** param
    elif kernel == 'SIGMOID':
        K_test = np.tanh(-param * (X_test @ X.T) + 1)

    y_pred = K_test @ alpha
    return y_pred


def plot_kernel_ridge(X, y, X_test, y_pred,nombre):
    plt.figure()
    plt.grid()
    plt.title(nombre)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.plot(X,y,'bo')
    plt.plot(X_test,y_pred,'ro')
    plt.legend(['entrenamiento','prediccion'])
    plt.show()