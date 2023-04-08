import itertools
import os
import random
import shutil

import numpy as np
import pandas as pd
import torch
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.sampling.lhs import LHS

from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize
from pymoo.termination import get_termination

import config
from config import benchmark, dimension, pop_size, generations, get_test_split
from optimizer import Optimizer

from ranker.data_preparations import create_feature_vector, create_edge_vector_generation
from ranker.main import train_in_generation, test_in_generation, pred_in_generation

generation = 1
problem = Optimizer(function_name=benchmark, n_var=dimension)
counter = 0


def save_generation(df, generation, fitness_function, d):
    save_path = f"{fitness_function}/d{d}/"
    current_directory = os.getcwd()
    path = os.path.join(current_directory, save_path)
    if not os.path.exists(path):
        # Create the directory
        # 'GeeksForGeeks' in
        # '/home / User / Documents'
        # with mode 0o666
        mode = 0o777
        temp_path = os.path.join(current_directory, f"{fitness_function}/")
        os.mkdir(temp_path, mode)
        os.mkdir(path, mode)
    try:
        old_df = pd.read_csv(f"{path}{generation}.csv")
        df = pd.concat([old_df, df])
    except:
        pass
    df.to_csv(f"{path}{generation}.csv", index=False)


def update_F(x, optimizer):
    x.F = [optimizer.func.evaluate(x.X)]
    return x


def create_P_test_train(P):
    percentage = get_test_split(generation)
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


def update_test_f(pop, test_set):
    source = []
    target = []
    label = []

    for a, b in test_set:
        pop[a] = update_F(pop[a], problem)
        pop[b] = update_F(pop[b], problem)
        source.append(pop[a].X)
        target.append(pop[b].X)
        label.append(1 if pop[a].F < pop[b].F else 0)
    df = pd.DataFrame.from_dict({"source": source, "target": target, "label": label})
    feature = create_feature_vector(df, False)
    feature.to_csv("../ranker/features.csv", index=False)
    df = create_edge_vector_generation(df)
    try:
        last_df = pd.read_csv(f"../ranker/generations/{generation}.csv")
        df = pd.concat([df, last_df])
    except:
        pass
    df.to_csv(f"../ranker/generations/{generation}.csv", index=False)
    generation_roc = test_in_generation(generation, config.last_model, config.pred)
    config.last_model_test_accuracy = generation_roc
    return pop, df


def calculate_pred_f(x, pop):
    pop[int(x["Src"])].F = [x["Weight"]]
    pop[int(x["Dst"])].F = [1 - x['Weight']]


def update_pred_f(pop, pred_set):
    global counter
    source = []
    target = []
    label = []
    for a, b in pred_set:
        source.append(pop[a].X)
        target.append(pop[b].X)
        label.append(1)
    df = pd.DataFrame.from_dict({"source": source, "target": target, "label": label})
    feature = create_feature_vector(df, False)
    feature.to_csv("../ranker/features.csv", index=False)
    df = create_edge_vector_generation(df)
    last_df = pd.read_csv(f"../ranker/generations/{generation}.csv")
    df_list = pd.concat([df, last_df])
    # df.to_csv(f"../ranker/generations/{generation}.csv",index=False)
    generation_pred = pred_in_generation(df, config.last_model, df_list, config.pred)
    df['Weight'] = generation_pred.cpu().numpy()
    df.apply(lambda x: calculate_pred_f(x, pop), axis=1)
    counter += 1
    return pop


def binary_tournament(pop, P=(100 * 100, 2), **kwargs):
    global generation
    # The P input defines the tournaments and competitors
    n_tournaments, n_competitors = P.shape
    if n_competitors != 2:
        raise Exception("Only pressure=2 allowed for binary tournament!")
    source = []
    target = []
    label = []
    if generation > 3:
        up_F = False
        pred_set, test_set = create_P_test_train(P)
        import numpy as np
        pop, df = update_test_f(pop, test_set)
        if config.last_model_test_accuracy > 0.7:
            update_pred_f(pop, pred_set)
        else:
            update_test_f(pop, pred_set)

    else:
        up_F = True
    # the result this function returns
    import numpy as np
    S = np.full(n_tournaments, -1, dtype=int)
    # now do all the tournaments
    for i in range(n_tournaments):
        a, b = P[i]
        source.append(pop[a].X)
        target.append(pop[b].X)
        if up_F:
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
    if up_F:
        df = pd.DataFrame.from_dict({"source": source, "target": target, "label": label})
        feature = create_feature_vector(df, False)
        feature.to_csv("../ranker/features.csv", index=False)
        df = create_edge_vector_generation(df)
        df.to_csv(f"../ranker/generations/{generation}.csv", index=False)
    model = train_in_generation(generation, config.last_model, config.pred)
    config.last_model = model
    generation += 1
    return S


class MySelection(TournamentSelection):

    def _do(self, _, pop, n_select, n_parents=1, **kwargs):
        return super()._do(_, pop, int((pop_size * (pop_size - 1)) / (n_parents * 2)), n_parents, **kwargs)


def main():
    algorithm = GA(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        selection=MySelection(pressure=2, func_comp=binary_tournament)
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
                   verbose=False, termination=get_termination("n_gen", generations))

    F_last = problem.func.evaluate(res.X)
    print(f"last objective {F_last}")
    return F_last


def delete_files():
    dir = "generations"

    # Path
    location = "C:/Users/vghar/PycharmProjects/GNNComparision/ranker/"
    path = os.path.join(location, dir)

    # Remove the specified
    # file path
    shutil.rmtree(path, ignore_errors=True)
    print("% s has been removed successfully" % dir)
    os.mkdir(path)


run = []
F_last = []
counters = []
for i in range(10):
    last_objective = main()
    run.append(i)
    F_last.append(last_objective)
    generation = 1
    config.last_model.reset_params()
    config.pred.reset_params()
    config.optimizer = torch.optim.Adam(itertools.chain(config.last_model.parameters(), config.pred.parameters()),
                                        lr=0.001)
    counters.append(counter)
    df = pd.DataFrame({"run": run, "last_objective": F_last, "usage_number": counters})
    df.to_csv(f"{config.benchmark}_{dimension}.csv")
    counter = 0
    delete_files()

"""
rosenbrock last objective 10 : [6.60381283]
rosenbrock last objective 20 :[18.16060861]
ackley last objective 10 : 0.02869612
ackley last objective 20 : [0.53113252]

"""
