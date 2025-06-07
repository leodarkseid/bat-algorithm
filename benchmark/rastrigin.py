import numpy as np

def rastrigin(X):
    A = 10.0
    n = X.size
    return A * n + np.sum(X**2 - A * np.cos(2 * np.pi * X))