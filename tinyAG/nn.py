import random
from tinyAG.engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0
        
    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, inpN, nonlinear=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(inpN)]
        self.b = Value(0.0)
        self.nonlinear = nonlinear

    def __call__(self, x):
        actFun = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return actFun.tanh() if self.nonlinear else actFun
    
    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'tanh' if self.nonlinear else 'linear'} Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, inpN, outN, **kwargs):
        self.neurons = [Neuron(inpN, **kwargs) for _ in range(outN)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}] \n"

class MLP(Module):

    def __init__(self, inpN, outNarr):
        self.layerSz = [inpN] + outNarr
        self.layers = [Layer(self.layerSz[i], self.layerSz[i+1], nonlinear=i!=len(outNarr)-1) for i in range(len(outNarr))]
    
    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x
    
    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]
    
    def __repr__(self):
        return f"MLP of [\n {' '.join(str(l) for l in self.layers)}]"

