from deap import benchmarks


class RosenBrock():

  def __init__(self,dim) -> None:
    self.dim = dim


  def evaluate(self,x):
    return benchmarks.rosenbrock(x)[0]

class Ackley:
  def evaluate(self,x):
    return benchmarks.ackley(x)[0]


class Griewank:
  def evaluate(self,x):
    return benchmarks.griewank(x)[0]