from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
from rich.live import Live
from rich.console import Console
import numpy as np

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
    ub=5.12,
    update_interval=10
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
    
       
    progress = Progress(
        TextColumn("[bold blue]Optimization"),
        BarColumn(),
        TextColumn("Iter: {task.completed}/{task.total}"),
        TextColumn("• Best: {task.fields[fitness]:.5f}"),
        TimeElapsedColumn(),
        expand=True
    )


    task = progress.add_task("Running", total=n_iter, fitness=best_fitness)
    # Main optimization loop
    with progress:
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
            if t % update_interval == 0 or t == n_iter - 1:
                progress.update(task, advance=update_interval, fitness=best_fitness)
    return best, best_fitness