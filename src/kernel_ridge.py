import numpy as np
import matplotlib.pyplot as plt

def kernel_ridge(X, y, kernel, gamma, X_test=None):
    """ Kernel Ridge Regression"""
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