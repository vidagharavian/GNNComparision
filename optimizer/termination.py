from pymoo.termination.default import DefaultSingleObjectiveTermination


class ObjectiveTermination(DefaultSingleObjectiveTermination):

    def __init__(self, xtol=1e-8, cvtol=1e-8, ftol=1e-6, period=30,best_solution=0, **kwargs) -> None:
        self.best_solution = best_solution
        super().__init__(xtol, cvtol, ftol, period, **kwargs)

    def update(self, algorithm):
        if self.f <= self.best_solution:
            self.force_termination=True
        return super().update(algorithm)



