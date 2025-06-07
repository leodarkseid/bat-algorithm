from main2 import bat_algorithm
from mainO import bat_algorithm_original
import numpy as np


def rastrigin(X):
    A = 10
    return A * len(X) + sum([(x ** 2 - A * np.cos(2 * np.pi * x)) for x in X])

lb = -5.12
ub = 5.12
n_bats = 30
dim = 10
Sol = np.random.uniform(lb, ub, (n_bats, dim)) 
print(Sol)
n_iter=5000

b1, bf1 = bat_algorithm_original(rastrigin, Sol=Sol, dim=dim, n_bats=n_bats, lb=5.12, ub=5.12, n_iter=n_iter)
b2, bf2 = bat_algorithm(rastrigin, Sol=Sol, dim=dim, n_bats=n_bats, n_iter=n_iter)

print ('b1', b1)
print ('b2', b2)
print ('bf1', bf1)
print ('bf2', bf2)


