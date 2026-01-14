import deeplearning_library
from hypothesis import settings, Verbosity, given
from hypothesis import strategies as st
from sympy.ntheory import isprime
  
@given(s=st.integers(min_value=1, max_value=2**10))
@settings(verbosity=Verbosity.normal, max_examples=500)
def test_is_prime(s):
    assert isprime(s) == deeplearning_library.is_prime(s)

def test_tensor_communication():
    data = [[2.0, 3.0], [4.0, 5.0]]
    result = deeplearning_library.py_computation(data)
    print(f"Type retourné par Rust : {type(result)}")
    print(f"Valeurs : {result}")

if __name__ == "__main__":
       test_tensor_communication()


