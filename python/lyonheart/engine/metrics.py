import numpy as np

class Metrics:
    def update():
        NotImplementedError

    def compute():
        NotImplementedError

    def reset(self):
        for key in self.__dict__:
            setattr(self, key, 0)

class Accuracy(Metrics):
    def __init__(self):
        super().__init__()
        self.valids = 0
        self.total = 0

    def update(self, y_pred, y_target):
        preds = np.argmax(y_pred.to_numpy(), axis=1)
        targets = np.argmax(y_target.to_numpy(), axis=1)
        self.valids += np.sum(preds == targets)
        self.total += len(preds)

    def compute(self):
        return self.valids / self.total if self.total > 0 else 0