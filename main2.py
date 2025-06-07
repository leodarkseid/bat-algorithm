import numpy as np

# Rastrigin function
def rastrigin(X):
    A = 10
    return A * len(X) + sum([(x ** 2 - A * np.cos(2 * np.pi * x)) for x in X])

# Bat Algorithm (Tawhid & Dsouza style)
def bat_algorithm(
    obj_func,
    Sol,
    dim=10,
    n_bats=30,
    n_iter=5000,
    f_min=0,
    f_max=2,
    A=0.9,# Loudness decay factor
    r=0.5,# Pulse rate increase factor
    lb=-5.12,# Lower bound constraint
    ub=5.12 # Upper bound constraint
):
    # Frequency of echolation 
    freq = np.zeros(n_bats)
    # simlar to pso, a 2d rep of velocities in all dimensions
    velocities = np.zeros((n_bats, dim))
    # initial random position in n-dimension of space for each bat
    # Sol = np.random.uniform(lb, ub, (n_bats, dim))
    # fitness of each bat
    Fitness = np.array([obj_func(sol) for sol in Sol])
    # print(Fitness)

#index of the best bat based on evaluation of the obj-func
    best_idx = np.argmin(Fitness)
# actual random position that's closest to the goal 
    best = Sol[best_idx]
    best_fitness = Fitness[best_idx]
    # print("Original Best Fitness", best_fitness)

    for t in range(n_iter):
        for i in range(n_bats):
            freq[i] = f_min + (f_max - f_min) * np.random.rand()
            velocities[i] = velocities[i] + (Sol[i] - best) * freq[i]
            S = Sol[i] + velocities[i]

            # Apply bounds
            S = np.clip(S, lb, ub)

            # Pulse rate-based local search
            if np.random.rand() > r:
                epsilon = np.random.normal(0, 1, dim)
                S = best + 0.001 * epsilon

            Fnew = obj_func(S)

            if (Fnew <= Fitness[i]) and (np.random.rand() < A):
                Sol[i] = S
                Fitness[i] = Fnew

            # Update global best
            
            if Fnew <= best_fitness:
                # print("New Fitness", Fnew)

                best = S
                best_fitness = Fnew

    return best, best_fitness

# # Run on Rastrigin
# best_sol, best_fit = bat_algorithm(rastrigin)
# print("Best solution:", best_sol)
# print("Best fitness:", best_fit)
