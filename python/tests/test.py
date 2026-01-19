import deeplearning_library as dl
import numpy as np
from hypothesis import settings, Verbosity, given
from hypothesis import strategies as st
from sympy.ntheory import isprime
  
@given(s=st.integers(min_value=1, max_value=2**10))
@settings(verbosity=Verbosity.normal, max_examples=500)
def test_is_prime(s):
    assert isprime(s) == dl.is_prime(s)

def test_tensor_communication():
   res = dl.py_computation(np.array([[1,2],[3,4]],dtype=np.float32))
   print(res)

def test_layers():
    l1 = dl.Linear(2,4)
    l2 = dl.Linear(4, 1)
    x = dl.GpuTensor(np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32))
    h = l1.forward(x).relu()
    y = l2.forward(h).sigmoid()
    res = y.to_numpy()
    print("Prédictions XOR (Avant entraînement) :")
    print(res)

def test_MLP_XOR():
    x = dl.GpuTensor(np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32))
    l1 = dl.Linear(2,4)
    l2 = dl.Linear(4, 1)
    h = l1.forward(x).relu()
    y_pred = l2.forward(h).sigmoid()
    loss = dl.LossFunction.mse(y_pred,dl.GpuTensor(np.array([[0,0],[1,1],[1,1],[0,0]], dtype=np.float32)))
    grads = loss.backward()
    


if __name__ == "__main__":
       test_layers()


