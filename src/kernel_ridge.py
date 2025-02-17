import numpy as np
import matplotlib.pyplot as plt

"""def kernel_ridge(X, y, kernel, gamma, X_test=None):
    \""" Kernel Ridge Regression\"""
    lambda_aux = 0.1
    n_samples, n_features = X.shape
    K = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = kernel(X[i], X[j], gamma)
    alpha = np.linalg.inv(K + lambda_aux * np.eye(n_samples)).dot(y)

    if X_test is not None:
        n_test_samples = X_test.shape[0]
        y_pred = np.zeros(n_test_samples)
        for i in range(n_test_samples):
            for j in range(n_samples):
                y_pred[i] += alpha[j] * kernel(X[j], X_test[i], gamma)
        return y_pred
    return alpha"""


def kernel_ridge_efficient(X, y, kernel_type='gaussian', gamma=1.0, X_test=None):
    """
    Efficient Kernel Ridge Regression using vectorized operations
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Training data
    y : array-like of shape (n_samples,)
        Target values
    kernel_type : str
        Type of kernel ('gaussian', 'polynomial', 'linear', 'logistic')
    gamma : float
        Kernel parameter (sigma for gaussian, p for polynomial)
    X_test : array-like of shape (n_test_samples, n_features), optional
        Test data
    """
    lambda_aux = 0.1
    n_samples = X.shape[0]
    
    # Compute pairwise distances or dot products efficiently
    if kernel_type == 'gaussian':
        # Compute squared Euclidean distances
        XX = np.sum(X**2, axis=1)[:, np.newaxis]
        YY = XX.T
        XY = np.dot(X, X.T)
        dist_matrix = XX + YY - 2 * XY
        K = np.exp(-dist_matrix / (2 * gamma**2))
    
    elif kernel_type in ['polynomial', 'linear', 'logistic']:
        K = np.dot(X, X.T)
        if kernel_type == 'polynomial':
            K = (1 + K) ** gamma
        elif kernel_type == 'logistic':
            K = np.tanh(K)
    
    # Solve for alpha
    alpha = np.linalg.solve(K + lambda_aux * np.eye(n_samples), y)
    
    if X_test is not None:
        # Compute kernel between test and training points
        if kernel_type == 'gaussian':
            test_XX = np.sum(X_test**2, axis=1)[:, np.newaxis]
            train_XX = np.sum(X**2, axis=1)[np.newaxis, :]
            test_XY = np.dot(X_test, X.T)
            test_dist_matrix = test_XX + train_XX - 2 * test_XY
            K_test = np.exp(-test_dist_matrix / (2 * gamma**2))
        else:
            K_test = np.dot(X_test, X.T)
            if kernel_type == 'polynomial':
                K_test = (1 + K_test) ** gamma
            elif kernel_type == 'logistic':
                K_test = np.tanh(K_test)
        
        return K_test.dot(alpha)
    
    return alpha


def plot_kernel_ridge(X, y, X_test, y_pred):
    plt.figure()
    plt.grid()
    plt.title('Regresion polinomial')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.plot(X,y,'bo')
    plt.plot(X_test,y_pred,'ro')
    plt.legend(['entrenamiento','prediccion'])
    plt.show()