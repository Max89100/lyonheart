from deeplearning_library import InitMethod, core
from .module import Module

class Linear(Module):
    def __init__(self, in_features, out_features, init_method=InitMethod.Kaiming):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.init_method = init_method
        self.layer = core.Linear(self.in_features,self.out_features,init_method)
        self.weight = self.layer.weight
        self.bias = self.layer.bias

class Sequential(Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self,x):
        for i in range(0,len(self.layers)):
            x = self.layers[i].forward(x)
        return x
       

class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.layer = core.ReLU()

class Softmax(Module):
    def __init__(self):
        super().__init__()
        self.layer = core.Softmax()