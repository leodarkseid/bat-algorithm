import numpy as np
import random


def rastrigin(X):
    """
    Rastrigin function: returns a scalar fitness for vector X.
    Global minimum at X = [0, 0, ..., 0] with value 0.
    """
    A = 10
    return A * len(X) + sum([(x ** 2 - A * np.cos(2 * np.pi * x)) for x in X])


def bat_algorithm(
    obj_func,
    Sol_init,
    dim,
    n_bats,
    n_iter=5000,
    f_min=0.0,
    f_max=2.0,
    A=0.9,
    r=0.5,
    lb=-5.12,
    ub=5.12
):
    """
    Tawhid & Dsouza (2018) style Bat Algorithm.

    Parameters:
        obj_func  : objective function accepting a vector and returning a scalar
        Sol_init  : initial positions (np.ndarray of shape (n_bats, dim))
        dim       : problem dimension (int)
        n_bats    : number of bats (int)
        n_iter    : number of iterations (int)
        f_min     : minimum frequency (float)
        f_max     : maximum frequency (float)
        A         : loudness (constant float)
        r         : pulse emission rate (constant float)
        lb, ub    : scalar lower/upper bounds for each dimension

    Returns:
        best            : best-found position (1D np.ndarray of length dim)
        best_fitness    : corresponding fitness (float)
    """
    # Initialize frequency, velocity, solution, and fitness arrays
    Q = np.zeros(n_bats)
    v = np.zeros((n_bats, dim))
    Sol = Sol_init.copy()
    Fitness = np.array([obj_func(sol) for sol in Sol])

    # Determine initial global best
    best_idx = np.argmin(Fitness)
    best = Sol[best_idx].copy()
    best_fitness = Fitness[best_idx]

    # Main optimization loop
    for t in range(n_iter):
        for i in range(n_bats):
            # 1) Update frequency
            Q[i] = f_min + (f_max - f_min) * np.random.rand()

            # 2) Update velocity & position
            v[i] = v[i] + (Sol[i] - best) * Q[i]
            S = Sol[i] + v[i]

            # 3) Apply simple bounds
            S = np.clip(S, lb, ub)

            # 4) Local random walk (with probability 1 - r)
            if np.random.rand() > r:
                epsilon = np.random.normal(0, 1, dim)
                S = best + 0.001 * epsilon
                S = np.clip(S, lb, ub)

            # 5) Evaluate new solution
            Fnew = obj_func(S)

            # 6) Accept if improved and random < A
            if (Fnew <= Fitness[i]) and (np.random.rand() < A):
                Sol[i] = S.copy()
                Fitness[i] = Fnew

            # 7) Update global best
            if Fnew < best_fitness:
                best = S.copy()
                best_fitness = Fnew

    return best, best_fitness


def bat_algorithm_original(
    obj_func,
    Sol_init,
    dim,
    n_bats,
    n_iter=5000,
    f_min_val=0.0,
    f_max_val=2.0,
    alpha=0.9,
    gamma=0.9,
    A0=1.0,
    r0=0.5,
    lb=-5.12,
    ub=5.12
):
    """
    Original Bat Algorithm (Yang, 2010) modified to accept initial positions.

    Parameters:
        obj_func  : objective function accepting a vector and returning a scalar
        Sol_init  : initial positions (np.ndarray of shape (n_bats, dim))
        dim       : problem dimension (int)
        n_bats    : number of bats (int)
        n_iter    : number of iterations (int)
        f_min_val : minimum frequency (float)
        f_max_val : maximum frequency (float)
        alpha     : loudness decay factor (float, e.g. 0.9)
        gamma     : pulse rate growth factor (float, e.g. 0.9)
        A0        : initial loudness (float)
        r0        : initial pulse rate (float)
        lb, ub    : scalar lower/upper bounds for each dimension

    Returns:
        best            : best-found position (1D np.ndarray of length dim)
        best_fitness    : corresponding fitness (float)
    """
    # Initialize frequency, velocity, solution, fitness, loudness, and pulse rate
    Q = np.zeros(n_bats)
    v = np.zeros((n_bats, dim))
    Sol = Sol_init.copy()
    Fitness = np.array([obj_func(sol) for sol in Sol])

    # Initial best bat
    best_idx = np.argmin(Fitness)
    best = Sol[best_idx].copy()
    best_fitness = Fitness[best_idx]

    A = np.full(n_bats, A0)  # loudness array
    r = np.full(n_bats, r0)  # pulse rate array

    for t in range(n_iter):
        for i in range(n_bats):
            # 1) Frequency update
            beta = np.random.rand()
            Q[i] = f_min_val + (f_max_val - f_min_val) * beta

            # 2) Velocity & position update
            v[i] = v[i] + (Sol[i] - best) * Q[i]
            S = Sol[i] + v[i]
            S = np.clip(S, lb, ub)

            # 3) Local search if rand > r[i]
            if np.random.rand() > r[i]:
                epsilon = np.random.normal(0, 1, dim)
                S = best + epsilon * np.mean(A)
                S = np.clip(S, lb, ub)

            # 4) Evaluate new candidate
            Fnew = obj_func(S)

            # 5) Accept new solution if better and rand < A[i]
            if (Fnew <= Fitness[i]) and (np.random.rand() < A[i]):
                Sol[i] = S.copy()
                Fitness[i] = Fnew
                # Update loudness and pulse rate
                A[i] *= alpha
                r[i] = r0 * (1 - np.exp(-gamma * t))

            # 6) Update global best
            if Fnew < best_fitness:
                best = S.copy()
                best_fitness = Fnew

    return best, best_fitness


# -----------------------------------------------------------------------------
# Exposed functions (for import):
#   - rastrigin
#   - bat_algorithm            (Tawhid & Dsouza, 2018 variant)
#   - bat_algorithm_original   (Yang, 2010 original variant)
# -----------------------------------------------------------------------------
