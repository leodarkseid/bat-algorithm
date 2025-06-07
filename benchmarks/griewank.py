import numpy as np

def griewank(X):
    n = X.size
    sum_sq = np.sum(X**2) / 4000.0

    # product term over i = 1..n; note: i is 1-based index here
    idx = np.arange(1, n + 1)
    prod_cos = np.prod(np.cos(X / np.sqrt(idx)))

    return 1.0 + sum_sq - prod_cos