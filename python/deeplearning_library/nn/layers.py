from deeplearning_library import core
import numpy as np

class Layer:
    def __init__(self):
        self.layer = NotImplementedError

    def forward(self,x):
        return self.layer.forward(x)

    def parameters(self):
        return []

class Linear(Layer):
    def __init__(self, in_features, out_features, init_method=core.InitMethod.Kaiming):
        self.in_features = in_features
        self.out_features = out_features
        self.init_method = init_method
        self.layer = core.Linear(self.in_features,self.out_features,init_method)

    def parameters(self):
        return self.layer.parameters()

class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.layer = core.ReLU()

class Softmax(Layer):
    def __init__(self):
        super().__init__()
        self.layer = core.Softmax()

class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, input):
        return self.forward(input)

    def forward(self,x):
        for i in range(0,len(self.layers)):
            x = self.layers[i].forward(x)
        return x
    
    def parameters(self):
        parameters = []
        for layer in self.layers:
            layer_parameter = layer.parameters()
            if layer_parameter != []:
                parameters.append(layer_parameter[0])
                parameters.append(layer_parameter[1])
        return parameters



       