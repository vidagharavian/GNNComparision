
from pymoo.core.problem import Problem, ElementwiseProblem
import benchmark_functions as bf
from opfunu.cec_based.cec2022 import F12022
class Optimizer(ElementwiseProblem):
    func =None
    def __init__(self, n_var=20,
                         n_obj=1,
                         xl=-5,
                         xu=5,**kwargs):
        self.func = F12022(ndim=n_var)
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         xl=xl,
                         xu=xu, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):

        out["F"] =self.func.evaluate(x)



