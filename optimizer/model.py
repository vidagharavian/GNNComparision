import os
import random

import numpy as np
import pandas as pd
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize
from pymoo.termination import get_termination

import config
from config import benchmark, dimension, pop_size, generations, get_test_split
from optimizer import Optimizer
from ranker.MyData import GraphSAGE
from ranker.data_preparations import create_feature_vector, create_edge_vector_generation
from ranker.main import train_in_generation, test_in_generation, pred_in_generation
from utils import MyRounder

generation = 1
problem = Optimizer(function_name=benchmark, n_var=dimension)


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
    feature =create_feature_vector(df,False)
    feature.to_csv("../ranker/features.csv",index=False)
    df = create_edge_vector_generation(df)
    df.to_csv(f"../ranker/generations/{generation}.csv",index=False)
    generation_roc = test_in_generation(generation, config.last_model)
    config.last_model_test_accuracy = generation_roc
    return pop


def calculate_pred_f(x, pop):
    pop[int(x["Src"])].F = [1 -x["Weight"]]
    pop[int(x["Dst"])].F = [x['Weight']]


def update_pred_f(pop, pred_set):
    source = []
    target = []
    label = []
    for a, b in pred_set:
        source.append(pop[a].X)
        target.append(pop[b].X)
        label.append(1)
    df = pd.DataFrame.from_dict({"source": source, "target": target, "label": label})
    feature = create_feature_vector(df, False)
    feature.to_csv("../ranker/features.csv",index=False)
    df = create_edge_vector_generation(df)
    df.to_csv(f"../ranker/generations/{generation}.csv",index=False)
    generation_pred = pred_in_generation(df, config.last_model)
    df['Weight'] =generation_pred.cpu().numpy()
    df.apply(lambda x:calculate_pred_f(x,pop),axis=1)
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
        up_F=False
        pred_set,test_set  = create_P_test_train(P)
        update_test_f(pop,test_set)
        if config.last_model_test_accuracy >0.7:
            update_pred_f(pop,pred_set)
        else:
            update_test_f(pop,pred_set)

    else:
        up_F=True


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
    df = pd.DataFrame.from_dict({"source": source, "target": target, "label": label})
    feature = create_feature_vector(df, False)
    feature.to_csv("../ranker/features.csv",index=False)
    df = create_edge_vector_generation(df)
    df.to_csv(f"../ranker/generations/{generation}.csv",index=False)
    model = train_in_generation(generation,config.last_model)
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
        crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=MyRounder()),
        mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=MyRounder()),
        eliminate_duplicates=True,
        selection=MySelection(pressure=2, func_comp=binary_tournament)
    )

    res = minimize(problem,
                   algorithm,
                   seed=1,
                   verbose=False, termination=get_termination("n_gen", generations))

    F_last =problem.func.evaluate(res.X)
    print(f"last objective {F_last}")
    return F_last


for j in [10,20]:
    run = []
    F_last = []
    config.dimension=j
    for i in range(10):
        last_objective =main()
        run.append(i)
        F_last.append(last_objective)
        generation=1
        config.last_model= GraphSAGE(dimension, 64,32,0.2)
        df = pd.DataFrame({"run":run,"last_objective":F_last})
        df.to_csv(f"{config.benchmark}_{dimension}.csv")

"""
rosenbrock last objective 10 : [6.60381283]
rosenbrock last objective 20 :[18.16060861]
ackley last objective 10 : 0.02869612
ackley last objective 20 : [0.53113252]

"""
