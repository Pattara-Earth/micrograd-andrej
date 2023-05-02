import random
from micrograd.engine import Value

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def step(self, lr):
        for p in self.parameters():
            p.data += -lr*p.grad

    def parameters(self):
        pass        

class Neuron(Module):
    def __init__(self, n_in: int):
        self.weight = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.bias = Value(random.uniform(-1, 1))
    
    def __repr__(self):
        return f"[w:{[i for i in self.weight]} b:{self.bias}]"
        
    def __call__(self, x):
        out = sum([wi*xi for wi, xi in zip(self.weight, x)], self.bias)
        act_out = out.tanh()
        return act_out
    
    def parameters(self):
        return self.weight + [self.bias]

class Layer(Module): # for connect with other layer
    def __init__(self, n_in: int, n_out: int):
        self.neurons = [Neuron(n_in) for _ in range(n_out)]
            
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out
    
    def __repr__(self):
        return f"Layer:[{', '.join(str(n) for n in self.neurons)}]"
    
    def parameters(self):
        return  [p for n in self.neurons for p in n.parameters()]
    
class MLP(Module):
    def __init__(self, n_in: int, n_out: int, hiddens: list): 
        sz = [n_in] + hiddens + [n_out]
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(sz)-1)]
    
    def __repr__(self):
        return f"MLP:[{', '.join(str(l) for l in self.layers)}]"
    
    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x
    
    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]