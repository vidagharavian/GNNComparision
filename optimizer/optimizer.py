
from pymoo.core.problem import Problem, ElementwiseProblem
import benchmark_functions as bf
from opfunu.cec_based.cec2022 import F12022,F22022

from benchmarks import RosenBrock


class Optimizer(ElementwiseProblem):
    func =None
    def __init__(self, n_var=20,
                         n_obj=1,
                         xl= -2.048,
                         xu=2.048,**kwargs):
        self.func = RosenBrock(dim=n_var)
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         xl=xl,
                         xu=xu, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):

        out["F"] =1




#GA d10 24.01358129 RosenBrock