import sys
import numpy as np
import time

import questionary
from tools import safe_ask
from yang.main import bat_algorithm_original
from benchmark.ackley import ackley 
from benchmark.griewank import griewank
from benchmark.rastrigin import rastrigin 
from benchmark.rosenbrock import rosenbrock 
from benchmark.schwefel import schwefel
from benchmark.sphere import sphere

def run_yang(benchmark, dim, n_bats, n_iter, lb, ub, seed=None, Sol_init=None ):
    if seed is not None:
        np.random.seed(seed)
    if Sol_init is None:
        Sol_init = np.random.uniform(lb, ub, (n_bats, dim))
    start = time.time()
    best, fitness = bat_algorithm_original(
        obj_func=benchmark,
        Sol_init=Sol_init,
        dim=dim,
        n_bats=n_bats,
        n_iter=n_iter,
        f_min_val=0.0,
        f_max_val=2.0,
        alpha=0.9,
        gamma=0.9,
        A0=1.0,
        r0=0.5,
        lb=lb,
        ub=ub
    )
    elapsed = time.time() - start
    return 'Yang', best, fitness, elapsed

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
    run_yang(benchmark=benchmarks[selected], dim=int(dimension), n_bats=int(n_bats), n_iter=int(n_iter), lb=float(lb), ub=float(ub))