import numpy as np
from pymoo.termination.default import DefaultSingleObjectiveTermination


class ObjectiveTermination(DefaultSingleObjectiveTermination):

    def __init__(self, xtol=1e-8, cvtol=1e-8, ftol=1e-6, period=600,best_solution=0,config=None,problem=None, **kwargs) -> None:
        self.best_solution = best_solution
        self.config = config
        self.problem = problem
        super().__init__(xtol, cvtol, ftol, period, **kwargs)

    def update(self, algorithm):
        # algorithm.evaluator.n_eval = int((self.config.counter*0.3 + (algorithm.n_gen-self.config.counter))*self.config.pop_size)
        F= [self.problem.func.evaluate(i.x) for i in algorithm.off]
        min_f = min(F)
        print(min_f)
        if min_f <= self.best_solution:
            self.force_termination=True
        return super().update(algorithm)




