import numpy as np

def ackley(X):
    n = X.size
    a = 20.0
    b = 0.2
    c = 2 * np.pi

    sum_sq = np.sum(X**2)
    sum_cos = np.sum(np.cos(c * X))

    term1 = -a * np.exp(-b * np.sqrt(sum_sq / n))
    term2 = -np.exp(sum_cos / n)

    return term1 + term2 + a + np.e