class SGD:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr
    
    def step(self):
        #[W1,b1,W2,b2,...]
        for i in range(len(self.params)):
           g = self.params[i].grad
           g_scaled = g.mul_scalar(self.lr)
           self.params[i].sub_assign(g_scaled)    