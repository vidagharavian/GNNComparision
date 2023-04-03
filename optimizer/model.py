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

from optimizer import Optimizer
from ranker.data_preparations import create_edge_vector_generation, create_feature_vector
from ranker.train import Trainer, args, random_seed, save_name_base, method_name
from utils import MyRounder

generation = 1
d = 10
fitness_function = "RosenBrock"
pop_size = 5 * d
generations = 30
problem = Optimizer(n_var=d)


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
        df = pd.concat([old_df,df])
    except:
        pass
    df.to_csv(f"{path}{generation}.csv", index=False)


def select_for_fitness_eval(generation):
    if generation < 2:
        percentage = 1
    elif generation < 0.25 * generations:
        percentage = 0.2
    elif generation < 0.7 * generations:
        percentage = 0.3
    else:
        percentage = 0.1
    return percentage


def update_F(x, optimizer):
    x.F = [optimizer.func.evaluate(x.X)]
    return x


def create_P_test_train(P):
    percentage = select_for_fitness_eval(generation)
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


def update_train_set(P, pop):
    n_tournaments, n_competitors = P.shape
    source = []
    target = []
    for i in range(n_tournaments):
        a, b = P[i]
        source.append(pop[a].X)
        target.append(pop[b].X)
    df = pd.DataFrame.from_dict({"source": source, "target": target})






def use_label_test(P, pop, train, last_F=False):
    global generation
    n_tournaments, n_competitors = P.shape
    source = []
    target = []
    label = []
    import numpy as np
    S = np.full(n_tournaments, -1, dtype=int)

    # now do all the tournaments
    for i in range(n_tournaments):
        a, b = P[i]
        if not last_F:
            pop[a] = update_F(pop[a], problem)
            pop[b] = update_F(pop[b], problem)
        source.append(pop[a].X)
        target.append(pop[b].X)
        # if the first individual is better, choose it
        if pop[a].F < pop[b].F:
            label.append(1)
            S[i] = a

        # otherwise take the other individual
        else:
            label.append(0)
            S[i] = b
    df = pd.DataFrame.from_dict({"source": source, "target": target, "label": label})
    create_feature_vector(df)
    df = create_edge_vector_generation(df)
    save_generation(df, generation, fitness_function, d)
    if train:
        train = Trainer(args, random_seed, save_name_base)
        model, split, log_path = train.train(method_name)
        config.model = model
        config.last_model_save = log_path
    else:
        train = Trainer(args, random_seed, save_name_base, True)
        train.testing(0, method_name, args, log_path=config.last_model_save, model=config.model)
    generation += 1


def binary_tournament(pop, P, **kwargs):
    global generation
    global d
    # The P input defines the tournaments and competitors
    n_tournaments, n_competitors = P.shape
    if n_competitors != 2:
        raise Exception("Only pressure=2 allowed for binary tournament!")
    if config.use_traditional:
        source = []
        target = []
        label = []
        import numpy as np
        S = np.full(n_tournaments, -1, dtype=int)
        # now do all the tournaments
        for i in range(n_tournaments):
            a, b = P[i]
            pop[a] = update_F(pop[a], problem)
            pop[b] = update_F(pop[b], problem)
            source.append(pop[a].X)
            target.append(pop[b].X)
            # if the first individual is better, choose it
            if pop[a].F < pop[b].F:
                label.append(1)
                S[i] = a

            # otherwise take the other individual
            else:
                label.append(0)
                S[i] = b
        df = pd.DataFrame.from_dict({"source": source, "target": target, "label": label})
        create_feature_vector(df)
        df = create_edge_vector_generation(df)
        save_generation(df, generation, fitness_function, d)
        train = Trainer(args, random_seed, save_name_base)
        model, split, log_path = train.train(method_name)
        config.model = model
        config.last_model_save = log_path
        generation += 1
        args.generation = generation
        config.use_traditional = False
    else:
        test_set, train_set = create_P_test_train(P)
        n_tournaments, n_competitors = test_set.shape

    source = []
    target = []
    label = []
    train_source = []
    train_target = []
    # the result this function returns

    return S


class MySelection(TournamentSelection):

    def _do(self, _, pop, n_select, n_parents=2, **kwargs):
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

    print(res.F)


main()
