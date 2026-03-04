class SGD:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr
    
    def step(self):
        #[W1,b1,W2,b2,...]
        # p -= grad * lr  
        for p in self.params:
           g = p.grad
           p -= g.mul_scalar(self.lr)