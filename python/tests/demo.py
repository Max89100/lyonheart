import torch as pt
import lyonheart as lh
import numpy as np

from lyonheart import *
from torch import *



def example():
        # LyonHeart
    x = lh.randn(32,784)
    y = lh.randn(32,1)
    model = lh.Sequential([lh.Linear(784,128),lh.ReLU(),lh.Linear(128,1), lh.Sigmoid()])
    criterion = lh.MSELoss()
    optimizer = lh.SGD(model.parameters(),0.01)
    loss = criterion(model(x),y)
    model.backward(loss)
    optimizer.step()

    # PyTorch
    x = pt.randn(32,784)
    y = pt.randn(32,1)
    model = pt.nn.Sequential(pt.nn.Linear(784,128), pt.nn.ReLU(), pt.nn.Linear(128,1),pt.nn.Sigmoid())
    criterion = pt.nn.MSELoss()
    optimizer = pt.optim.SGD(model.parameters(),0.01)
    loss = criterion(model(x),y)
    loss.backward()
    optimizer.step()

    # MyNetwork
    class MyNetwork(lh.Module):
        def __init__(self):
            super().__init__()
            self.layer = Sequential([Linear(2,4), ReLU(), Linear(4,1), Sigmoid()])
        
        def forward(self, x):
            return super().forward(x)
    x = lh.tensor([[1,2],[3,4]])
    model = MyNetwork()
    y = model(x)
    print(y)
    #CoreTensor([0.056530166, 0.002974218], device=Cpu)


if __name__ == "__main__":
    dataset = lh.data.datasets.MNIST("../../data/mnist", train=True,one_hot=True)
    train_loader = DataLoader(dataset,64,True)
    test_loader = DataLoader(dataset,64,False)
    model = Sequential([Linear(784,128), ReLU(), Linear(128,10)])
    criterion = lh.losses.LogSoftmax()
    optimizer = lh.optim.SGD(model.parameters(),lr = 0.1)
    trainer = Trainer(model,optimizer,[Accuracy()])
    trainer.train(10,train_loader,criterion)
    trainer.evaluate(test_loader)


