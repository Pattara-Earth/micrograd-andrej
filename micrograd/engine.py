import math

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0 # by default this variable not changing or effect the loss function
        self._backward = lambda: None
        self._prev = set(_children) # set of tuple: unchange and ordered
        self._op = _op
        self.label = label
        
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad # += mean chain rule multivariable case
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __radd__(self, other): # other.__add__(self) -> self.__add__(other)
        return self + other
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self, other): # other.__mul__(self) -> self.__mul__(other)
        return self * other
    
    def __neg__(self): # negative self
        return -1 * self
    
    def __sub__(self, other): # self - other
        return self + (-other)
    
    def __rsub__(self, other): # other.__sub__(self) -> self.__sub__(other)
        return self - other
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), f"Input type should be int/float: {type(other)}"
        out = Value(self.data**other, (self, ), f"**{other}")
        def _backward():
            self.grad += other * (self.data**(other-1)) * out.grad
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        return self * (other**-1)
    
    def __rtruediv__(self, other):
        return other * (self**-1)
    
    def exp(self):
        x = self.data
        exp = math.exp(x)
        out = Value(exp, (self, ), 'exp')
        def _backward():
            self.grad += exp * out.grad
        out._backward = _backward
        return out
    
    # activation funcitons
    def tanh(self): 
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        def _backward():
            self.grad = (1-t**2) *out.grad
        out._backward = _backward
        return out
    
    def relu(self):
        out = Value(self.data if self.data>0 else 0, (self, ), 'relu')
        def _backward():
            self.grad += (out.data>0) * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        # build topological sort graph
        stack = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                    visited.add(v)
                    for adjs_v in v._prev:
                        build_topo(adjs_v)
                    stack.append(v)
                    
        build_topo(self)
        stack.reverse()
        
        # backward apply chain rule to get gradient
        self.grad = 1.0
        for v in stack:
            v._backward()