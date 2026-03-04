import tqdm as tqdm
import lyonheart as lh
from ..nn.module import Module
from ..data import DataLoader
from .metrics import Metrics
import numpy as np

class Trainer():
    def __init__(self,model:Module,optimizer, metrics:Metrics=None):
        self.model = model
        self.optimizer = optimizer
        self.metrics = metrics or []

    def logs(self,y_pred,y_target,loss=None):
        logs = {}
        if loss:
            logs = {"loss": f"{loss.to_numpy().item():.4f}"}
        for metric_fn in self.metrics:
                    metric_fn.update(y_pred,y_target)
                    val = metric_fn.compute()
                    logs[metric_fn.__class__.__name__.lower()] = f"{val:.4f}"
        return logs

    def train(self,epochs, train_loader:DataLoader, criterion):
        self.model.train()
        for epoch in range(epochs):
            pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for (x,y) in pbar:
                y_pred = self.model(lh.tensor(x))
                y_target = lh.tensor(y)
                loss = criterion(y_pred,y_target)
                self.model.backward(loss)
                self.optimizer.step()
                pbar.set_postfix(self.logs(y_pred,y_target,loss))
                
    def evaluate(self,test_loader:DataLoader):
        self.model.eval()
        pbar = tqdm.tqdm(test_loader,desc=f"Evaluation")
        for (x,y) in pbar:
            y_pred = self.model(lh.tensor(x))
            y_target = lh.tensor(y)
            pbar.set_postfix(self.logs(y_pred,y_target))

    
        



    
