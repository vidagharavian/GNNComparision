import pandas as pd
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from optimizer import Optimizer
from utils import MyRounder

generation = 1


def binary_tournament(pop, P=(100*100,2), **kwargs):
    global generation
    # The P input defines the tournaments and competitors
    n_tournaments, n_competitors = P.shape
    if n_competitors != 2:
        raise Exception("Only pressure=2 allowed for binary tournament!")
    source =[]
    target = []
    label =[]
    # the result this function returns
    import numpy as np
    S = np.full(n_tournaments, -1, dtype=int)
    # now do all the tournaments
    for i in range(n_tournaments):
        a, b = P[i]
        source.append(pop[a].X)
        target.append( pop[b].X)
        # if the first individual is better, choose it
        if pop[a].F < pop[b].F:
            label.append(1)
            S[i] = a

        # otherwise take the other individual
        else:
            label.append(0)
            S[i] = b
    df=pd.DataFrame.from_dict({"source":source,"target":target,"label":label})
    df.to_csv(f"generations/{generation}.csv")
    generation+=1
    return S
class MySelection(TournamentSelection):

    def _do(self, _, pop, n_select, n_parents=1, **kwargs):
        return super()._do(_, pop, 2000, n_parents, **kwargs)


def main():
    problem = Optimizer()
    algorithm = GA(
        pop_size=100,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=1.0, eta=3.0, vtype=float,repair=MyRounder()),
        mutation=PM(prob=1.0, eta=3.0, vtype=float,repair=MyRounder()),
        eliminate_duplicates=True,
        selection=MySelection(pressure=2, func_comp=binary_tournament)
    )

    res = minimize(problem,
                   algorithm,
                   seed=1,
                   verbose=False, termination=get_termination("n_gen", 200))

    print(res.X)

main()