
import sys
import numpy as np
import time
import questionary
from runTawhid import run_tawhid
from runYang import run_yang
from yang.main import bat_algorithm_original
from tawhidDsouza.main import bat_algorithm
from benchmark.ackley import ackley 
from benchmark.griewank import griewank
from benchmark.rastrigin import rastrigin 
from benchmark.rosenbrock import rosenbrock 
from benchmark.schwefel import schwefel
from benchmark.sphere import sphere
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

    # ------------------------------
    # 4) Run the Original Yang (2010) Bat Algorithm
    # ------------------------------
    # start_time_1 = time.time()
    # best1, fitness1 = bat_algorithm_original(
    #     obj_func=benchmark,
    #     Sol_init=Sol_init,
    #     dim=dim,
    #     n_bats=n_bats,
    #     n_iter=n_iter,
    #     f_min_val=0.0,
    #     f_max_val=2.0,
    #     alpha=0.9,
    #     gamma=0.9,
    #     A0=1.0,
    #     r0=0.5,
    #     lb=lb,
    #     ub=ub
    # )
    # elapsed1 = time.time() - start_time_1
    
    name, best1, fitness1, elapsed1 = run_yang(benchmark=benchmark, dim=dim, n_bats=n_bats, n_iter=n_iter, lb=lb, ub=ub, Sol_init=Sol_init)

    # ------------------------------
    # 5) Reseed RNG and run Tawhid & Dsouza (2018) variant
    # ------------------------------
    # # Reseeding ensures the same random‐number sequence for fairness
    np.random.seed(seed)
    # start_time_2 = time.time()
    # best2, fitness2 = bat_algorithm(
    #     obj_func=rastrigin,
    #     Sol_init=Sol_init,
    #     dim=dim,
    #     n_bats=n_bats,
    #     n_iter=n_iter,
    #     f_min=0.0,
    #     f_max=2.0,
    #     A=0.9,
    #     r=0.5,
    #     lb=lb,
    #     ub=ub
    # )
    # elapsed2 = time.time() - start_time_2
    name, best2, fitness2, elapsed2 = run_tawhid(benchmark=benchmark, dim=dim, n_bats=n_bats, n_iter=n_iter, lb=lb, ub=ub, Sol_init=Sol_init)

    # ------------------------------
    # 6) Print & compare results
    # ------------------------------
    print("=== Bat Algorithm Comparison ===\n")
    print("Original Yang (2010) Variant:")
    print(f"  Best solution (rounded): {np.round(best1, 4)}")
    print(f"  Best fitness:            {fitness1:.6f}")
    print(f"  Time taken:              {elapsed1:.4f} seconds\n")

    print("Tawhid & Dsouza (2018) Variant:")
    print(f"  Best solution (rounded): {np.round(best2, 4)}")
    print(f"  Best fitness:            {fitness2:.6f}\n")
    print(f"  Time taken:              {elapsed2:.4f} seconds\n")

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
        default='rastrigin'
    ))

    dimension = safe_ask(questionary.text("Dimension", default='10', validate=lambda val: val.replace('-', '').isdigit()))
    n_bats = safe_ask(questionary.text("Number Of Bats", default='30', validate=lambda val: val.isdigit()))
    n_iter = safe_ask(questionary.text("Number Of Iteration", default='5000', validate=lambda val: val.isdigit()))
    lb = safe_ask(questionary.text("Low Bound", default='-5.12'))
    ub = safe_ask(questionary.text("Upper Bound", default='5.12'))

    if selected:
        main(benchmarks[selected], selected, dim=int(dimension), n_bats=int(n_bats), n_iter=int(n_iter), lb=float(lb), ub=float(ub))
   
   
   
   