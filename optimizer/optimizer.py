
from pymoo.core.problem import Problem, ElementwiseProblem
import benchmark_functions as bf
from opfunu.cec_based.cec2022 import F12022

from benchmarks import RosenBrock


class Optimizer(ElementwiseProblem):
    func =None
    def __init__(self, n_var=10,
                         n_obj=1,
                         xl=-5,
                         xu=5,function_name='F12022',**kwargs):
        if function_name == 'F12022':
            self.func = F12022(ndim=n_var)
        if function_name == 'RosenBrock':
            xl = -2.048
            xu = 2.048
            self.func = RosenBrock(dim=n_var)
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         xl=xl,
                         xu=xu, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):

        out["F"] =self.func.evaluate(x)



