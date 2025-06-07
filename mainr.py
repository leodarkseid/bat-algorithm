import numpy as np

# Rastrigin function
def rastrigin(X):
    A = 10
    return A * len(X) + sum([(x ** 2 - A * np.cos(2 * np.pi * x)) for x in X])

# Bat Algorithm (Standard, corrected and improved)
def bat_algorithm(
    obj_func,
    dim=10,          # Dimension of the problem
    n_bats=30,       # Number of bats
    n_iter=500,      # Number of iterations
    f_min=0,         # Minimum frequency
    f_max=2,         # Maximum frequency
    A_init=0.9,      # Initial loudness
    r_init=0.5,      # Initial pulse rate
    alpha=0.9,       # Loudness decay factor
    gamma=0.9,       # Pulse rate increase factor
    lb=-5.12,        # Lower bound
    ub=5.12          # Upper bound
):
    # Initialize bat population
    Q = np.zeros(n_bats)  # Frequencies
    v = np.zeros((n_bats, dim))  # Velocities
    Sol = np.random.uniform(lb, ub, (n_bats, dim))  # Positions
    Fitness = np.array([obj_func(sol) for sol in Sol])  # Fitness values
    A = np.full(n_bats, A_init)  # Loudness for each bat
    r = np.full(n_bats, r_init)  # Pulse rate for each bat

    # Find initial best solution
    best_idx = np.argmin(Fitness)
    best = Sol[best_idx].copy()  # Best position
    best_fitness = Fitness[best_idx]  # Best fitness

    # Main loop
    for t in range(n_iter):
        for i in range(n_bats):
            # Update frequency
            Q[i] = f_min + (f_max - f_min) * np.random.rand()
            
            # Update velocity and position (corrected formula)
            v[i] = v[i] + (best - Sol[i]) * Q[i]  # Move toward best solution
            S = Sol[i] + v[i]  # New position
            
            # Apply bounds
            S = np.clip(S, lb, ub)

            # Local search around best solution
            if np.random.rand() < r[i]:  # Use pulse rate to decide local search
                epsilon = np.random.normal(0, 1, dim)
                S = best + A[i] * epsilon  # Step size proportional to loudness

            # Evaluate new solution
            Fnew = obj_func(S)

            # Accept new solution based on fitness and loudness
            if (Fnew <= Fitness[i]) and (np.random.rand() < A[i]):
                Sol[i] = S
                Fitness[i] = Fnew
                # Update loudness and pulse rate
                A[i] *= alpha  # Decrease loudness
                r[i] = r_init * (1 - np.exp(-gamma * (t + 1)))  # Increase pulse rate

            # Update global best
            if Fnew < best_fitness:  # Strict inequality for better convergence
                best = S.copy()
                best_fitness = Fnew

    return best, best_fitness

# Run on Rastrigin
np.random.seed(42)  # For reproducibility
best_sol, best_fit = bat_algorithm(rastrigin)
print("Best solution:", best_sol)
print("Best fitness:", best_fit)