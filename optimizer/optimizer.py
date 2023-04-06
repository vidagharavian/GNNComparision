
from pymoo.core.problem import Problem, ElementwiseProblem

from benchmarks import RosenBrock, Ackley


class Optimizer(ElementwiseProblem):
    func =None
    def __init__(self, n_var=10,
                         n_obj=1,
                         xl=-5,
                         xu=5,function_name='F12022',**kwargs):
        if function_name == 'RosenBrock':
            xl = -2.048
            xu = 2.048
            self.func = RosenBrock(dim=n_var)
        if function_name == "Ackley":
            xl = -32.768
            xu = 32.768
            self.func = Ackley()

        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         xl=xl,
                         xu=xu, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):

        out["F"] =self.func.evaluate(x)



