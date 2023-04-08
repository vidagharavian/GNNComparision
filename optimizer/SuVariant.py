import numpy as np
import pandas as pd
from pymoo.algorithms.soo.nonconvex.de import DE, Variant
from pymoo.core.population import Population
from pymoo.core.variable import get
from pymoo.operators.control import NoParameterControl
from pymoo.operators.crossover.binx import mut_binomial
from pymoo.operators.crossover.dex import de_differential
from pymoo.operators.crossover.expx import mut_exp
from pymoo.operators.repair.bounds_repair import repair_random_init
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.rnd import fast_fill_random
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.util.display.single import SingleObjectiveOutput
from pymoo.util.misc import where_is_what

import config
from model import update_test_f, update_pred_f, update_F
from ranker.data_preparations import create_feature_vector, create_edge_vector_generation
from ranker.main import train_in_generation


class SuVariant(Variant):

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

            best = lambda: np.random.choice(np.where(pop.get("rank") == 0)[0], replace=True, size=n_matings)

            if sel_type == "rand":
                fast_fill_random(P, len(pop), columns=range(n_parents), Xp=itself)
            elif sel_type == "best":
                P[:, 0] = best()
                fast_fill_random(P, len(pop), columns=range(1, n_parents), Xp=itself)
            elif sel_type == "target-to-best":
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
            elif name =="surrogate":
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
        control.advance(off)

        return off

    def my_selection(self,pair1,pair2):
        pop = np.concatenate((pair1,pair2),axis=0)
        pop = Population(pop)
        P = np.concatenate((np.arange(0,len(pair1)-1),np.arange(len(pair1),2*len(pair1))),axis=1)
        return self.binary_tournament(pop,self.generation,self.optimizer,P)

    def binary_tournament(self,pop,generation,optimizer, P=(100 * 100, 2)):
        # The P input defines the tournaments and competitors
        n_tournaments, n_competitors = P.shape
        if n_competitors != 2:
            raise Exception("Only pressure=2 allowed for binary tournament!")
        source = []
        target = []
        label = []
        if generation > 3:
            up_F = False
            from optimizer.model import create_P_test_train
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
                pop[a] = update_F(pop[a], optimizer)
                pop[b] = update_F(pop[b], optimizer)
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

class SuDE(DE):

    def __init__(self, pop_size=100, n_offsprings=None, sampling=FloatRandomSampling(), variant="DE/best/1/bin",
                 output=SingleObjectiveOutput(),generation=1, **kwargs):
        if variant is None:
            if "control" not in kwargs:
                kwargs["control"] = NoParameterControl
            variant = Variant(**kwargs)

        elif isinstance(variant, str):
            try:
                _, selection, n_diffs, crossover = variant.split("/")
                if "control" not in kwargs:
                    kwargs["control"] = NoParameterControl
                variant = SuVariant(selection=selection, n_diffs=int(n_diffs), crossover=crossover,optimizer = self.problem,generation=generation)
            except:
                raise Exception("Please provide a valid variant: DE/<selection>/<n_diffs>/<crossover>")

        super().__init__(pop_size=pop_size,
                         n_offsprings=n_offsprings,
                         sampling=sampling,
                         mating=variant,
                         survival=None,
                         output=output,
                         eliminate_duplicates=False,
                         **kwargs)

        self.termination = DefaultSingleObjectiveTermination()