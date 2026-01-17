import deeplearning_library
import numpy as np
from hypothesis import settings, Verbosity, given
from hypothesis import strategies as st
from sympy.ntheory import isprime
  
@given(s=st.integers(min_value=1, max_value=2**10))
@settings(verbosity=Verbosity.normal, max_examples=500)
def test_is_prime(s):
    assert isprime(s) == deeplearning_library.is_prime(s)

def test_tensor_communication():
   res = deeplearning_library.py_computation(np.array([[1,2],[3,4]],dtype=np.float32))
   print(res)

if __name__ == "__main__":
       test_tensor_communication()


