class RosenBrock():

  def __init__(self,dim) -> None:
    self.dim = dim


  def evaluate(self,x):
    dim = self.dim
    sum = 0.0
    for i in range(dim-1):
      a = 100 * ((x[i+1] - x[i]**2)**2)
      b = (1 - x[i])**2
      sum += a + b
    return sum