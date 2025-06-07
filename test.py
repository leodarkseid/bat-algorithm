
import sys
import numpy as np
import time
import questionary
from runTawhid import run_tawhid
from runYang import run_yang
from yang.main import bat_algorithm_original
from tawhidDsouza.main import bat_algorithm
from benchmarks.ackley import ackley 
from benchmarks.griewank import griewank
from benchmarks.rastrigin import rastrigin 
from benchmarks.rosenbrock import rosenbrock 
from benchmarks.schwefel import schwefel
from benchmarks.sphere import sphere
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table
# from benchmark import ackley, griewank, rastrigin, rosenbrock, schwefel, sphere



def main(benchmark, benchmark_name, dim, n_bats, n_iter, lb, ub):

    print(f"\nRunning benchmark: {benchmark_name}\n")
    # ------------------------------
    # 1) Setup: fixed random seed
    # ------------------------------
    seed = 42
    np.random.seed(seed)
    

    # ------------------------------
    # 2) Common parameters
    # ------------------------------
    # dim = 10            # problem dimension
    # n_bats = 30         # number of bats
    # n_iter = 5000       # number of iterations
    # lb, ub = -5.12, 5.12 # search space bounds

    # ------------------------------
    # 3) Generate a single initial population
    # ------------------------------
    # All bats share the same starting positions for a fair comparison:
    Sol_init = np.random.uniform(lb, ub, (n_bats, dim))
    
    _, best1, fitness1, elapsed1, original1 = run_yang(benchmark=benchmark, dim=dim, n_bats=n_bats, n_iter=n_iter, lb=lb, ub=ub, Sol_init=Sol_init)

    # ------------------------------
    # 5) Reseed RNG and run Tawhid & Dsouza (2018) variant
    # ------------------------------
    # # Reseeding ensures the same random‐number sequence for fairness
    np.random.seed(seed)
   
    _, best2, fitness2, elapsed2, original2 = run_tawhid(benchmark=benchmark, dim=dim, n_bats=n_bats, n_iter=n_iter, lb=lb, ub=ub, Sol_init=Sol_init)
    
    
    def print_results(title, best, original_fitness, best_fitness, elapsed):
        table = Table.grid(padding=(0, 2))
        table.add_column(justify="right", style="bold cyan")
        table.add_column()

        table.add_row("Best solution (rounded):", str(np.round(best, 4)))
        table.add_row("Original fitness:", f"{original_fitness:.6f}")
        table.add_row("Best fitness:", f"{best_fitness:.6f}")
        table.add_row("Time taken:", f"{elapsed:.4f} seconds")

        rprint(Panel(table, title=f"[bold green]{title}", expand=False))


# Usage:
    print_results("Original Yang (2010) Variant", best1, original1, fitness1, elapsed1)
    print_results("Tawhid & Dsouza (2018) Variant", best2, original2, fitness2, elapsed2)



    print("Note: Both used the same initial population and random seed reset for fairness.\n")


def safe_ask(q):
    try:
        answer = q.ask()
        if answer is None:
            print("\nCancelled by user.")
            sys.exit(0)
        return answer
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")
        sys.exit(0)


if __name__ == "__main__":
     # Benchmark options
    benchmarks = {'ackley':ackley,
    'griewank':griewank,
    'rastrigin':rastrigin,
    'rosenbrock':rosenbrock,
    'schwefel':schwefel,
    'sphere':sphere}
    selected = safe_ask(questionary.select(
        "Select a benchmark function:",
        choices=list(benchmarks),
        default='rastrigin',
    ))

    dimension = safe_ask(questionary.text("Dimension", default='10', validate=lambda val: val.replace('-', '').isdigit()))
    n_bats = safe_ask(questionary.text("Number Of Bats", default='30', validate=lambda val: val.isdigit()))
    n_iter = safe_ask(questionary.text("Number Of Iteration", default='5000', validate=lambda val: val.isdigit()))
    lb = safe_ask(questionary.text("Low Bound", default='-5.12'))
    ub = safe_ask(questionary.text("Upper Bound", default='5.12'))

    if selected:
        main(benchmarks[selected], selected, dim=int(dimension), n_bats=int(n_bats), n_iter=int(n_iter), lb=float(lb), ub=float(ub))
   
   
   
   