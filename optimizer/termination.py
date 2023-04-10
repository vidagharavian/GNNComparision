from pymoo.termination.default import DefaultSingleObjectiveTermination


class ObjectiveTermination(DefaultSingleObjectiveTermination):

    def __init__(self, xtol=1e-8, cvtol=1e-8, ftol=1e-6, period=30, **kwargs) -> None:
        super().__init__(xtol, cvtol, ftol, period, **kwargs)