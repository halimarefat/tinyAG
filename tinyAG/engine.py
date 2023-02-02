import math


class Value:

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda : None
        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only support int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * self.data**(other - 1.0) * out.grad
        out._backward = _backward

        return out

    def __truediv__(self, other):
        return self * other**(-1)

    def __rmul__(self, other):
        return self * other
    
    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def tanh(self):
        t = (math.exp(2*self.data) - 1) / (math.exp(2*self.data) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1.0 - t**2.0) * out.grad
        out._backward = _backward

        return out
    
    def relu(self):
        out = Value(max(self.data, 0), (self,), 'ReLU')

        def _backward():
            self.grad += 1.0 * out.grad if out.data > 0.0 else 0.0
        out._backward = _backward

        return out

    def sigmoid(self):
        x = math.exp(self.data * -1.0)
        out = Value(1.0 / (1.0 + x), (self,), 'sigmoid')

        def _bakcward():
            self.grad += (x / (1.0 + x)**2.0) * out.grad
        out._backward = _bakcward

        return out

    def arctan(self):
        out = Value(math.atan(self.data), (self,), 'ArcTan')

        def _backward():
            self.grad += (1.0 / (1.0 + self.data**2.0)) * out.grad
        out._backward = _backward

        return out

    def softplus(self):
        out = Value(math.log(1 + math.exp(self.data)), (self,), 'SoftPlus')

        def _backward():
            self.grad += (1.0 / (1.0 + math.exp(self.data * -1.0))) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(n):
            if n not in visited:
                visited.add(n)
                for c in n._prev:
                    build_topo(c)
                topo.append(n)
        build_topo(self)

        self.grad = 1.0
        for n in reversed(topo):
            n._backward()
