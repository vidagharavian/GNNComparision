

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.core.individual import Individual
from pymoo.operators.control import NoParameterControl
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.util.display.single import SingleObjectiveOutput

import numpy as np
import pandas as pd
from pymoo.algorithms.soo.nonconvex.de import Variant
from pymoo.core.population import Population
from pymoo.core.variable import get
from pymoo.operators.control import EvolutionaryParameterControl
from pymoo.operators.crossover.binx import mut_binomial
from pymoo.operators.crossover.dex import de_differential
from pymoo.operators.crossover.expx import mut_exp
from pymoo.operators.repair.bounds_repair import repair_random_init

from pymoo.operators.selection.rnd import fast_fill_random

from pymoo.util.misc import where_is_what

from model import create_P_test_train, update_test_f, update_pred_f, update_F
from ranker.main import train_in_generation


class MyVariant(Variant):



    def __init__(self, selection="best", n_diffs=1, F=0.5, crossover="bin", CR=0.2, jitter=False, prob_mut=0.1,
                 control=EvolutionaryParameterControl,problem=None,config=None,**kwargs):
        self.problem=problem
        self.config = config
        super().__init__(selection, n_diffs, F, crossover, CR, jitter, prob_mut, control, **kwargs)

    def do(self, problem, pop, n_offsprings, algorithm=None, **kwargs):
        control = self.control

        # let the parameter control now some information
        control.tell(pop=pop)

        # set the controlled parameter for the desired number of offsprings
        control.do(n_offsprings)

        # find the different groups of selection schemes and order them by category
        sel, n_diffs = get(self.selection, self.n_diffs, size=n_offsprings)
        H = where_is_what(zip(sel, n_diffs))

        # get the parameters used for reproduction during the crossover
        F, CR, jitter = get(self.F, self.CR, self.jitter, size=n_offsprings)

        # the `target` vectors which will be recombined
        X = pop.get("X")

        # the `donor` vector which will be obtained through the differential equation
        donor = np.full((n_offsprings, problem.n_var), np.nan)

        # for each type defined by the type and number of differentials
        for (sel_type, n_diffs), targets in H.items():

            # the number of offsprings created in this run
            n_matings, n_parents = len(targets), 1 + 2 * n_diffs

            # create the parents array
            P = np.full([n_matings, n_parents], -1)

            itself = np.array(targets)[:, None]


            if sel_type == "rand" or "best":
                fast_fill_random(P, len(pop), columns=range(n_parents), Xp=itself)
            elif sel_type == "target-to-best":
                best = lambda: np.random.choice(np.where(pop.get("rank") == 0)[0], replace=True, size=n_matings)
                P[:, 0] = targets
                P[:, 1] = best()
                fast_fill_random(P, len(pop), columns=range(2, n_parents), Xp=itself)
            else:
                raise Exception("Unknown selection method.")

            # get the values of the parents in the design space
            XX = np.swapaxes(X[P], 0, 1)

            # do the differential crossover to create the donor vector
            Xp = de_differential(XX, F[targets], jitter[targets])

            # make sure everything stays in bounds
            if problem.has_bounds():
                Xp = repair_random_init(Xp, XX[0], *problem.bounds())

            # set the donors (the one we have created in this step)
            donor[targets] = Xp

        # the `trial` created by by recombining target and donor
        trial = np.full((n_offsprings, problem.n_var), np.nan)

        crossover = get(self.crossover, size=n_offsprings)
        for name, K in where_is_what(crossover).items():

            _target = X[K]
            _donor = donor[K]
            _CR = CR[K]

            if name == "bin":
                M = mut_binomial(len(K), problem.n_var, _CR, at_least_once=True)
                _trial = np.copy(_target)
                _trial[M] = _donor[M]
            elif name == "exp":
                M = mut_exp(n_offsprings, problem.n_var, _CR, at_least_once=True)
                _trial = np.copy(_target)
                _trial[M] = _donor[M]
            elif name == "line":
                w = np.random.random((len(K), 1)) * _CR[:, None]
                _trial = _target + w * (_donor - _target)
            elif name == "hypercube":
                w = np.random.random((len(K), _target.shape[1])) * _CR[:, None]
                _trial = _target + w * (_donor - _target)
            elif name == "surrogate":
                pass
            else:
                raise Exception(f"Unknown crossover variant: {name}")

            trial[K] = _trial

        # create the population
        off = Population.new(X=trial)


        # do the mutation which helps to add some more diversity
        off = self.mutation(problem, off)


        # repair the individuals if necessary - disabled if repair is NoRepair
        off = self.repair(problem, off, **kwargs)
        # advance the parameter control by attaching them to the offsprings
        off = self.my_selection(off, pop)
        # control.advance(off)

        return off


    def my_selection(self, pop1:Population, pop2:Population):
        pair1=pop1.get("X")
        pair2=pop2.get("X")
        pop = np.concatenate((pair1, pair2), axis=0)
        indiv =[Individual(**{"X":individual})for individual in pop]
        pop = Population(individuals=indiv)
        x = np.arange(0,len(pair1)).reshape(len(pair1),1)
        y = np.arange(len(pair1), (2 * len(pair1))).reshape(len(pair1),1)
        train_pair =[]
        for p1 in x:
            for p2 in y:
                train_pair.append([p1[0],p2[0]])
        train_pair = np.array(train_pair)
        train_pair = train_pair.reshape(train_pair.shape[0],2)
        S = self.binary_tournament(pop, train_pair, **{"problem": self.problem})
        pop = pop[S]
        return pop

    def creat_key_set(self,S ,test_set):
        return_dict={}
        for i,s in enumerate(S):
            return_dict[f"{test_set[i][0]}_{test_set[i][1]}"]=s
        return return_dict



    def binary_tournament(self, pop, P=(100 * 100, 2), **kwargs):
        gen = self.config.current_gen
        problem = kwargs['problem']
        pred_set, test_set = create_P_test_train(P, gen)
        if self.config.last_model is not None:
            S1 = update_test_f(pop, test_set, problem, gen, self.config)
            if gen > 3 and self.config.last_model_test_accuracy > 0.7:
                S2 = update_pred_f(pop, pred_set, gen, self.config)
            else:
                S2 = update_test_f(pop, pred_set, problem, gen, self.config,False)
            S1.extend(S2)
            S = np.array(S1)
            S = self.ranker(len(pop),S)
        else:
            source = []
            target = []
            label = []
            n_tournaments, n_competitors = P.shape
            if n_competitors != 2:
                raise Exception("Only pressure=2 allowed for binary tournament!")
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
            feature = self.config.create_feature_vector(df, False)
            feature.to_csv("../ranker/features.csv", index=False)
            df = self.config.create_edge_vector_generation(df)
            df.to_csv(f"../ranker/generations/{gen}.csv", index=False)
        self.config.last_model = train_in_generation(gen, self.config.last_model, self.config.pred, self.config.optimizer)
        self.config.current_gen += 1
        self.config.to_csv()
        return S

    def ranker(self,pop_size,s):
        u, count = np.unique(s, return_counts=True)
        count_sort_ind = np.argsort(-count)

        return u[count_sort_ind][:int(pop_size/2)]




class MyDe(GeneticAlgorithm):

    def __init__(self, pop_size=100, n_offsprings=None, sampling=FloatRandomSampling(), variant="DE/best/1/bin",
                 output=SingleObjectiveOutput(), generation=1,config=None, **kwargs):
        self.config = config
        self.problem = kwargs["problem"]
        if variant is None:
            if "control" not in kwargs:
                kwargs["control"] = NoParameterControl
            variant = Variant(**kwargs)

        elif isinstance(variant, str):
            _, selection, n_diffs, crossover = variant.split("/")
            if "control" not in kwargs:
                    kwargs["control"] = NoParameterControl
            self.variant = MyVariant(selection=selection, n_diffs=int(n_diffs), crossover=crossover,
                                    problem=self.problem, generation=generation,config=self.config)


        super().__init__(pop_size=pop_size,
                         n_offsprings=n_offsprings,
                         sampling=sampling,
                         mating=self.variant,
                         survival=None,
                         output=output,
                         eliminate_duplicates=False,
                         **kwargs)

        self.termination = DefaultSingleObjectiveTermination()

    def _initialize_advance(self, infills=None, **kwargs):
        FitnessSurvival().do(self.problem, self.pop, return_indices=True)

    def _infill(self):
        infills = self.mating.do(self.problem, self.pop, self.n_offsprings, algorithm=self)
        # tag each individual with an index - if a steady state version is executed
        index = np.arange(len(infills))

        # if number of offsprings is set lower than pop_size - randomly select
        if self.n_offsprings < self.pop_size:
            index = np.random.permutation(len(infills))[:self.n_offsprings]
            infills = infills[index]

        infills.set("index", index)

        return infills



    def _advance(self, infills=None, **kwargs):
        assert infills is not None, "This algorithms uses the AskAndTell interface thus infills must to be provided."

        # get the indices where each offspring is originating from
        I = infills.get("index")

        F = [self.config.generations - self.config.current_gen for x in range(len(infills))]
        F = np.array(F).reshape(len(F), 1)
        infills.set(**{"F":F})
        # replace the individuals with the corresponding parents from the mating
        self.pop[I] = infills

        # update the information regarding the current population
        FitnessSurvival().do(self.problem, self.pop, return_indices=True)

    def _set_optimum(self, **kwargs):
        k = self.pop.get("rank") == 0
        self.opt = self.pop[k]


