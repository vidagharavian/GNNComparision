
import os
import random
import shutil

import numpy as np
import pandas as pd

from pymoo.algorithms.soo.nonconvex.ga import GA

from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from config import Config
from optimizer import Optimizer
from ranker.main import train_in_generation, test_in_generation, pred_in_generation


def update_F(x, optimizer):
    x.F = [optimizer.func.evaluate(x.X)]
    return x


def create_P_test_train(P, gen):
    percentage = Config.get_test_split(gen)
    test_set = []
    train_set = []
    m = P.flatten()
    m = np.unique(m)
    selection = random.sample(list(m), k=int(percentage * len(m)))
    for a, b in P:
        if (int(a) in selection) and (int(b) in selection):
            train_set.append([a, b])
        else:
            test_set.append([a, b])
    return np.array(test_set), np.array(train_set)


def update_test_f(pop, test_set, problem, gen):
    source = []
    target = []
    label = []
    S = []
    for a, b in test_set:
        pop[a] = update_F(pop[a], problem)
        pop[b] = update_F(pop[b], problem)
        source.append(pop[a].X)
        target.append(pop[b].X)
        label.append(1 if pop[a].F < pop[b].F else 0)
        S.append(a if pop[a].F < pop[b].F else b)
    df = pd.DataFrame.from_dict({"source": source, "target": target, "label": label})
    feature = config.create_feature_vector(df, False)
    feature.to_csv("../ranker/features.csv", index=False)
    m = config.create_edge_vector_generation(df)
    try:
        last_df = pd.read_csv(f"../ranker/generations/{gen}.csv")
        m = pd.concat([m, last_df])
    except:
        pass
    m.to_csv(f"../ranker/generations/{gen}.csv", index=False)
    generation_roc = test_in_generation(gen, config.last_model, config.pred)
    config.last_model_test_accuracy = generation_roc
    return S


def calculate_pred_f(x, pop):
    pop[int(x["Src"])].F = [x["Weight"]]
    pop[int(x["Dst"])].F = [1 - x['Weight']]


def update_pred_f(pop, pred_set, gen):
    source = []
    target = []
    label = []

    for a, b in pred_set:
        source.append(pop[a].X)
        target.append(pop[b].X)
        label.append(1)
    df = pd.DataFrame.from_dict({"source": source, "target": target, "label": label})
    feature = config.create_feature_vector(df, False)
    feature.to_csv("../ranker/features.csv", index=False)
    m = config.create_edge_vector_generation(df)
    last_df = pd.read_csv(f"../ranker/generations/{gen}.csv")
    df_list = pd.concat([m, last_df])
    # df.to_csv(f"../ranker/generations/{generation}.csv",index=False)
    generation_pred = pred_in_generation(m, config.last_model, df_list, config.pred)
    s = []
    for p, c in zip(pred_set, generation_pred.cpu().numpy()):
        s.append(p[0] if c > 0.5 else p[1])
    # df.apply(lambda x: calculate_pred_f(x, pop), axis=1)
    config.counter += 1
    return s


def binary_tournament(pop, P=(100 * 100, 2), **kwargs):
    global config
    gen = config.current_gen
    problem = kwargs['problem']
    # The P input defines the tournaments and competitors
    n_tournaments, n_competitors = P.shape
    if n_competitors != 2:
        raise Exception("Only pressure=2 allowed for binary tournament!")
    if gen > 3:
        pred_set, test_set = create_P_test_train(P, gen)
        S1 = update_test_f(pop, test_set, problem, gen)
        if config.last_model_test_accuracy > 0.8:
            S2 = update_pred_f(pop, pred_set, gen)
        else:
            S2 = update_test_f(pop, pred_set, problem, gen)
        S1.extend(S2)
        S = np.array(S1)
    else:
        source = []
        target = []
        label = []
        # the result this function returns
        S = np.full(n_tournaments, -1, dtype=int)
        # now do all the tournaments
        for i in range(n_tournaments):
            a, b = P[i]
            source.append(pop[a].X)
            target.append(pop[b].X)
            pop[a] = update_F(pop[a], problem)
            pop[b] = update_F(pop[b], problem)
            # if the first individual is better, choose it
            if pop[a].F < pop[b].F:
                label.append(1)
                S[i] = a

            # otherwise take the other individual
            else:
                label.append(0)
                S[i] = b
        df = pd.DataFrame.from_dict({"source": source, "target": target, "label": label})
        feature = config.create_feature_vector(df, False)
        feature.to_csv("../ranker/features.csv", index=False)
        df = config.create_edge_vector_generation(df)
        df.to_csv(f"../ranker/generations/{gen}.csv", index=False)
    config.last_model = train_in_generation(gen, config.last_model, config.pred,config.optimizer)
    config.current_gen += 1
    return S




class MySelection(TournamentSelection):

    def __init__(self, func_comp=None, pressure=2, problem=None, **kwargs):
        self.problem = problem
        super().__init__(func_comp, pressure, **kwargs)

    def _do(self, _, pop, n_select, n_parents=1, **kwargs):
        return super()._do(_, pop, n_select*4, n_parents, **{"problem": self.problem})


def main():
    problem = Optimizer(n_var=config.dimension, function_name=config.benchmark)
    algorithm = GA(
        pop_size=config.pop_size,
        sampling=FloatRandomSampling(),
        selection=MySelection(pressure=2, func_comp=binary_tournament, problem=problem)
    )
    # algorithm = DE(
    #     pop_size=100,
    #     sampling=LHS(),
    #     variant="DE/best/0/bin",
    #     CR=0.9,
    #     dither="vector",
    #     jitter=False,
    #     n_diff = 2
    #
    # )

    res = minimize(problem,
                   algorithm,
                   seed=1,
                   verbose=False, termination=get_termination("n_gen", 50))

    F_last = problem.func.evaluate(res.X)
    print(f"last objective {F_last}")
    return F_last


def delete_files():
    dir = "generations"

    # Path
    location = "C:/Users/vghar/PycharmProjects/GNNComparision/ranker/"
    path = os.path.join(location, dir)
    path2 = os.path.join(location, "features.csv")
    # Remove the specified
    # file path
    shutil.rmtree(path, ignore_errors=True)
    os.remove(path2)
    print("% s has been removed successfully" % dir)
    os.mkdir(path)


run = []
F_last = []
counters = []
for i in range(10):
    config = Config()
    last_objective = main()
    run.append(i)
    F_last.append(last_objective)
    counters.append(config.counter)
    df = pd.DataFrame({"run": run, "last_objective": F_last, "usage_number": counters})
    df.to_csv(f"{config.benchmark}_{Config.dimension}.csv")
    config.reset_params()
    delete_files()

"""
rosenbrock last objective 10 : [6.60381283]
rosenbrock last objective 20 :[18.16060861]
ackley last objective 10 : 0.02869612
ackley last objective 20 : [0.53113252]

"""
