import numpy as np

# Rastrigin function as objective
def rastrigin(X):
    A = 10
    return A * len(X) + sum([(x ** 2 - A * np.cos(2 * np.pi * x)) for x in X])

def bat_algorithm_original(
    obj_func,
    Sol,
    dim=10,
    n_bats=30,
    n_iter=5000,
    f_min=0.0,
    f_max=2.0,
    alpha=0.9,
    gamma=0.9,
    A0=1.0,
    r0=0.5,
    lb=-5.12,
    ub=5.12
):
    # Initialize
    Q = np.zeros(n_bats)                        # frequency
    v = np.zeros((n_bats, dim))                 # velocities
    # Sol = np.random.uniform(lb, ub, (n_bats, dim))  # positions
    Fitness = np.array([obj_func(sol) for sol in Sol])
    
    best_idx = np.argmin(Fitness)
    best = Sol[best_idx].copy()
    best_fitness = Fitness[best_idx]

    A = np.full(n_bats, A0)                     # loudness
    r = np.full(n_bats, r0)                     # pulse rate

    for t in range(n_iter):
        for i in range(n_bats):
            beta = np.random.rand()
            Q[i] = f_min + (f_max - f_min) * beta
            v[i] = v[i] + (Sol[i] - best) * Q[i]
            S = Sol[i] + v[i]
            S = np.clip(S, lb, ub)

            if np.random.rand() > r[i]:
                epsilon = np.random.normal(0, 1, dim)
                S = best + epsilon * np.mean(A)

            Fnew = obj_func(S)

            if (Fnew <= Fitness[i]) and (np.random.rand() < A[i]):
                Sol[i] = S
                Fitness[i] = Fnew
                A[i] *= alpha
                r[i] = r0 * (1 - np.exp(-gamma * t))

            if Fnew < best_fitness:
                best = S
                best_fitness = Fnew

    return best, best_fitness


# # Run on Rastrigin
# best_sol, best_fit = bat_algorithm_original(rastrigin)
# print("Best solution:", best_sol)
# print("Best fitness:", best_fit)