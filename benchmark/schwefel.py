import numpy as np

def schwefel(X):
    n = X.size
    return 418.9829 * n - np.sum(X * np.sin(np.sqrt(np.abs(X))))