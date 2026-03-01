from . import _deeplearning_core as core
from . import datasets
from .data import Dataset, DataLoader
from .nn.layers import Linear, ReLU, Layer, Sequential, Softmax
from .optim import SGD
from ._deeplearning_core import CoreTensor
import numpy as np

def tensor(data) -> CoreTensor:
    """Helper pour créer un CoreTensor plus rapidement. Ne prends en charge que la 2D : [[1,2,3], [4,5,6]]"""
    
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.float32)
    return CoreTensor(data)

def zeros(shape: tuple) -> CoreTensor:
    return tensor(np.zeros(shape, dtype=np.float32))

def ones(shape: tuple) -> CoreTensor:
    return tensor(np.ones(shape, dtype=np.float32))

def randn(shape: tuple) -> CoreTensor:
    return tensor(np.random.randn(*shape).astype(np.float32))


# Optionnel : définir ce qui est visible lors d'un "from deeplearning_library import *"
__all__ = ["core", "datasets", "Dataset", "DataLoader", "Sequential", "Linear", "ReLU","SGD"]
