from ._lyonheart_core import CoreTensor, Parameter, InitMethod
from . import _lyonheart_core as core
from . import nn
from . import losses
from . import optim 
from . import data
from . import engine
from .losses import CrossEntropyLoss, MSELoss, LogSoftmax
from .data import Dataset, DataLoader, datasets
from .nn import Linear, ReLU, Sequential, Softmax, Module, Sigmoid
from .optim import SGD
from .engine import Trainer, Metrics, Accuracy
import numpy as np
import pickle

def save(model, path:str):
    state = model.state_dict()
    try:
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"--- Model successfully saved to: {path} ---")
    except Exception as e:
        print(f"--- Error during saving: {e} ---")

def load(path:str):
        try:
            with open(path,"+rb") as f:
                state:dict = pickle.load(f)
            print(f"--- Model successfully loaded from : {path} ---")
        except Exception as e:
            print(f"--- Error during loading: {e} ---")
        return state

def tensor(data) -> CoreTensor:
    """Helper pour créer un CoreTensor plus rapidement. Ne prends en charge que la 2D : [[1,2,3], [4,5,6]]. Applatis le tenseur en vecteur 1D."""
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.float32)
    return CoreTensor(data)

def zeros(*shape) -> CoreTensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    return CoreTensor.zeros(shape)

def ones(*shape: tuple) -> CoreTensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    return CoreTensor.ones(shape)

def randn(*shape: tuple) -> CoreTensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    return CoreTensor.randn(shape)

def rand(*shape: tuple) -> CoreTensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    return CoreTensor.rand(shape)

def zeros_like(self:CoreTensor) -> CoreTensor:
    return self.zeros_like()



# Optionnel : définir ce qui est visible lors d'un "from deeplearning_library import *"
__all__ = [
    "CoreTensor", "Parameter", "tensor", "zeros","zeros_like","ones","randn","rand", 
    "nn", "losses", "optim", "data",
    "CrossEntropyLoss", "MSELoss", "LogSoftmax", "SGD", 
    "DataLoader", "Dataset", "datasets", "Linear", "ReLU", "Softmax", "Sequential", "Module", "Sigmoid", "Trainer", "Accuracy", "Metrics"
]
