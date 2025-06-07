from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
from rich.live import Live
from rich.console import Console
import numpy as np

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
    ub=5.12,
    update_interval=10
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
    
    
        
    progress = Progress(
        TextColumn("[bold blue]Optimization"),
        BarColumn(),
        TextColumn("Iter: {task.completed}/{task.total}"),
        TextColumn("• Best: {task.fields[fitness]:.5f}"),
        TimeElapsedColumn(),
        expand=True
    )
    
    task = progress.add_task("Running", total=n_iter, fitness=best_fitness)

    with progress:
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
            if t % update_interval == 0 or t == n_iter - 1:
                            progress.update(task, advance=update_interval, fitness=best_fitness)
    return best, best_fitness