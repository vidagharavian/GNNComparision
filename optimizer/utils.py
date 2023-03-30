import numpy as np
from pymoo.operators.repair.rounding import RoundingRepair

class MyRounder(RoundingRepair):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _do(self, problem, X, **kwargs):
        return np.around(X,3)

