import pandas as pd
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.factory import get_termination
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize

from SuDE import MyDe
from config import Config
from model import delete_files, MySelection, binary_tournament
from optimizer import Optimizer
from termination import ObjectiveTermination


def main():
    global config
    problem = Optimizer(n_var=config.dimension, function_name=config.benchmark)
    if config.algorithm =="GA":
        algorithm = GA(
            pop_size=config.pop_size,
            sampling=FloatRandomSampling(),
            selection=MySelection(pressure=2, func_comp=binary_tournament, problem=problem,config=config)
        )
    elif config.algorithm =="DE":
        algorithm = MyDe(
            pop_size=config.pop_size,
            variant="DE/rand/1/exp",
            CR=0.9,
            dither="vector",
            jitter=False,
            n_diff = 3,
            problem=problem,
            config=config,
            prob_mut=0.1,
            F=1.5
        )
        PolynomialMutation()
        # algorithm = DE(100,sampling=LHS(),
        #     variant="DE/rand/1/bin",
        #     CR=0.9,
        #     dither="vector",
        #     jitter=False,
        #     n_diff = 2)
        try:
            best_solution = pd.read_csv(f"output/best_{config.algorithm}_{config.benchmark}_{config.dimension}.csv")
            best_solution=best_solution["last_objective"][0]
        except:
            res = minimize(problem,
                           DE(100,sampling=LHS(),
            variant="DE/rand/1/bin",
            CR=0.9,
            dither="vector",
            jitter=False,
            n_diff = 2),
                           seed=1,
                           verbose=False,termination=get_termination("n_gen", 300))
            df = pd.DataFrame(
                {"run": [0], "last_objective": res.F, "usage_number": [0], "generation": [300]})
            df.to_csv(f"output/best_{config.algorithm}_{config.benchmark}_{Config.dimension}.csv")
            best_solution=problem.func.evaluate(res.X)





    res = minimize(problem,
                   algorithm,
                   seed=1,
                   verbose=False,termination=ObjectiveTermination(best_solution=best_solution,**{"config":config,"problem":problem,"n_max_gen":config.generations*2
                                                                                                  }))
    # res = minimize(problem,
    #                algorithm,
    #                seed=1,
    #                verbose=False, termination=get_termination("n_gen", config.generations))

    F_last = problem.func.evaluate(res.X)
    config.from_csv()
    print(f"last objective {F_last}")
    return F_last


run = []
F_last = []
counters = []
generations =[]
for i in range(10):
    delete_files()
    config = Config()
    last_objective = main()
    run.append(i)
    F_last.append(last_objective)
    counters.append(config.counter)
    generations.append(config.current_gen)
    df = pd.DataFrame({"run": run, "last_objective": F_last, "usage_number": counters,"generation":generations})
    df.to_csv(f"{config.benchmark}_{config.algorithm}_{Config.dimension}.csv")
    config.reset_params()
