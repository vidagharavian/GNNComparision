import os
import random
import shutil

import numpy as np
import pandas as pd

from pymoo.operators.selection.tournament import TournamentSelection
from sklearn.metrics import roc_auc_score

from config import Config
from ranker.DGL_presentation import create_archive

from ranker.main import train_in_generation, pred_in_generation


def update_F(x, optimizer):
    x.F=([optimizer.func.evaluate(x.X)])
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


def update_test_f(pop, test_set, problem, gen, config):
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
    label = m['Weight'].values.copy()
    last_df = create_archive(gen-1,archive_size=config.archive_size)
    prediction_score = get_prediction_score(m,gen,config,last_df)

    config.last_model_test_accuracy =roc_auc_score(label, prediction_score)
    print(f"generation:{gen} , accuracy:{config.last_model_test_accuracy}")
    # generation_roc = test_in_generation(gen, config.last_model, config.pred)
    # config.last_model_test_accuracy = generation_roc
    return S


def calculate_pred_f(x, pop):
    pop[int(x["Src"])].F = [x["Weight"]]
    pop[int(x["Dst"])].F = [1 - x['Weight']]

def get_prediction_score(m, gen, config,last_df=None):
    m["Weight"] =np.full(len(m["Weight"]),0)
    if last_df is None:
        last_df = pd.read_csv(f"../ranker/generations/{gen}.csv")
    df_list = pd.concat([m, last_df])
    # df.to_csv(f"../ranker/generations/{generation}.csv",index=False)
    generation_pred = pred_in_generation(m, config.last_model, df_list, config.pred)
    return generation_pred


def update_pred_f(pop, pred_set, gen, config):
    source = []
    target = []
    label = []

    for a, b in pred_set:
        source.append(pop[a].X)
        target.append(pop[b].X)
        label.append(0)
    df = pd.DataFrame.from_dict({"source": source, "target": target, "label": label})
    feature = config.create_feature_vector(df, False)
    feature.to_csv("../ranker/features.csv", index=False)
    m = config.create_edge_vector_generation(df)
    generation_pred = get_prediction_score(m,gen,config)
    s = []
    for p, c in zip(pred_set, generation_pred.cpu().numpy()):
        s.append(p[0] if c > 0.5 else p[1])
    # df.apply(lambda x: calculate_pred_f(x, pop), axis=1)
    config.counter += 1
    return s


def binary_tournament(pop, P=(100 * 100, 2), **kwargs):
    config = kwargs["config"]
    gen = config.current_gen
    problem = kwargs['problem']
    # The P input defines the tournaments and competitors
    n_tournaments, n_competitors = P.shape
    if n_competitors != 2:
        raise Exception("Only pressure=2 allowed for binary tournament!")
    if gen > 3:
        pred_set, test_set = create_P_test_train(P, gen)
        S1 = update_test_f(pop, test_set, problem, gen, config)
        if config.last_model_test_accuracy > 0.8:
            S2 = update_pred_f(pop, pred_set, gen, config)
        else:
            S2 = update_test_f(pop, pred_set, problem, gen, config)
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
    config.last_model = train_in_generation(gen, config.last_model, config.pred, config.optimizer, config.archive_size)
    config.current_gen += 1
    return S


class MySelection(TournamentSelection):

    def __init__(self, func_comp=None, pressure=2, problem=None, config=None, **kwargs):
        self.problem = problem
        self.config = config
        super().__init__(func_comp, pressure, **kwargs)

    def _do(self, _, pop, n_select, n_parents=1, **kwargs):
        return super()._do(_, pop, n_select * 4, n_parents, **{"problem": self.problem, "config": self.config})


def delete_files():
    dir = "generations"

    # Path
    location = "../ranker/"
    path = os.path.join(location, dir)
    path2 = os.path.join(location, "features.csv")
    # Remove the specified
    # file path
    try:
        shutil.rmtree(path, ignore_errors=True)
        os.remove(path2)
        print("% s has been removed successfully" % dir)
    except FileNotFoundError:
        print("% s has been removed successfully" % dir)
    os.mkdir(path)


"""
rosenbrock last objective 10 : [6.60381283]
rosenbrock last objective 20 :[18.16060861]
ackley last objective 10 : 0.02869612
ackley last objective 20 : [0.53113252]

"""
