import pandas as pd
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.factory import get_termination
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation, PM
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize

from SuDE import MyDe
from config import Config
from model import delete_files, MySelection, binary_tournament
from optimizer import Optimizer
from utils import MyRounder
from termination import ObjectiveTermination


def main(run,function):
    global config
    problem = Optimizer(n_var=config.dimension, function_name=function)
    if config.algorithm =="GA":
        algorithm = GA(
            pop_size=config.pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.1, eta=20.0, vtype=float, repair=MyRounder()),
            mutation=PM(prob=0.1, eta=20.0, vtype=float, repair=MyRounder()),
            eliminate_duplicates=True,
            # selection=MySelection(pressure=2, func_comp=binary_tournament, problem=problem,config=config)
        )
    elif config.algorithm =="DE":
        # algorithm = MyDe(
        #     pop_size=config.pop_size,
        #     variant=f"DE/"+("best1" if config.best else "best")+"/1/exp",
        #     CR=0.9,
        #     dither="vector",
        #     jitter=False,
        #     n_diff = 3,
        #     problem=problem,
        #     config=config,
        #     prob_mut=0.1,
        #     F=0.5,
        #     **{"best":config.best}
        # )
        PolynomialMutation()
        algorithm = DE(100,sampling=LHS(),
            variant="DE/rand/1/bin",
            CR=0.9,
            dither="vector",
            jitter=False,
            n_diff = 2,prob_mut=0.1,
            F=0.5,)






    res = minimize(problem,
                   algorithm,
                   seed=run,
                   verbose=False,termination=get_termination("n_gen", config.generations) )
    # res = minimize(problem,
    #                algorithm,
    #                seed=run,
    #                verbose=False, termination=get_termination("n_gen", config.generations))
    try:
        F_last = problem.func.evaluate(res.X)
    except:
        F_last = problem.evaluate(res.X)
    config.from_csv()
    print(f"last objective {F_last}")
    return F_last

def get_best_solution(config):
    try:
        best_solution = pd.read_csv(f"output/best_{config.benchmark}_{config.algorithm}_{config.dimension}.csv")
        best_solution = best_solution['last_objective'].mean()
        return best_solution
    except:
        return 0


run = []
F_last = []
counters = []
generations =[]
from cec2017.functions import all_functions

for f in all_functions:
    for i in range(31):
        delete_files()

        config = Config()
        best_solution = get_best_solution(config)
        last_objective = main(i,f)
        run.append(i)
        F_last.append(last_objective)
        counters.append(config.counter)
        generations.append(config.current_gen)
        df = pd.DataFrame({"run": run, "last_objective": F_last, "usage_number": counters,"generation":generations})
        if not config.best:
            df.to_csv(f"output/{config.benchmark}_{config.algorithm}_{Config.dimension}.csv")
        else:
            df.to_csv(f"output/best_{config.benchmark}_{config.algorithm}_{Config.dimension}.csv")
        config.reset_params()
