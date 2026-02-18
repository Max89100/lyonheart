from deeplearning_library import core

class Layer:
    def __init__(self):
        self.layer = NotImplementedError

    def forward(self,x):
        return self.layer.forward(x)

class Linear(Layer):
    def __init__(self, in_features, out_features, init_method=core.InitMethod.Kaiming):
        self.in_features = in_features
        self.out_features = out_features
        self.init_method = init_method
        self.layer = core.Linear(self.in_features,self.out_features,init_method)

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

       