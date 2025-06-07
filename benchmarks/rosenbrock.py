import numpy as np

def rosenbrock(X):
    xi = X[:-1]
    xnext = X[1:]
    return np.sum(100.0 * (xnext - xi**2)**2 + (xi - 1.0)**2)