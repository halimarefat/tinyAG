import math
from tinyAG.engine import Value

class optimizers(Value):

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1.0 - t**2.0) * out.grad
        out._backward = _backward

        return out
        