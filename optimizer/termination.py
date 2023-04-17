import numpy as np
from pymoo.termination.default import DefaultSingleObjectiveTermination


class ObjectiveTermination(DefaultSingleObjectiveTermination):

    def __init__(self, xtol=1e-8, cvtol=1e-8, ftol=1e-6, period=30,best_solution=0,config=None, **kwargs) -> None:
        self.best_solution = best_solution
        self.config = config
        super().__init__(xtol, cvtol, ftol, period, **kwargs)

    def update(self, algorithm):
        # algorithm.evaluator.n_eval = int((self.config.counter*0.3 + (algorithm.n_gen-self.config.counter))*self.config.pop_size)
        F= [i.f for i in algorithm.off]
        min_f = min(F)
        if min_f <= self.best_solution:
            self.force_termination=True
        return super().update(algorithm)



