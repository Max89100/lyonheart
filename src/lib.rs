use burn::tensor::TensorData;
use pyo3::prelude::*;
use burn::tensor::{Tensor, backend::Backend};
use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use numpy::{PyArray, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};

/// Declared rust functions
#[pyfunction]
fn is_prime(num: u32) -> bool {
    match num {
        0 | 1 => false,
        _ => {
            let limit = (num as f32).sqrt() as u32; 

            (2..=limit).any(|i| num % i == 0) == false
        }
    }
}


// Elle ne sait pas si c'est du GPU, du CPU ou du TPU. Elle sait juste faire des maths.
// Fonction "Générique"
fn run_burn_logic<B: Backend>(data: TensorData, device: &B::Device) -> TensorData {
    // 1. On crée le tenseur à partir du TensorData
    let tensor1: Tensor<B, 2> = Tensor::from_data(data, device);
    let tensor2 = Tensor::ones_like(&tensor1);
    
    // 2. On fait l'opération et on repasse en TensorData pour le retour
    (tensor1 + tensor2).into_data()
}

#[pyfunction]
fn py_computation<'py>(py: Python<'py>, input: PyReadonlyArray2<'_, f32>) -> PyResult<Bound<'py, PyArray2<f32>>> {
    type MyBackend = Wgpu;
    let device = WgpuDevice::DefaultDevice;

    // A. Extraction : NumPy -> TensorData
    let shape = [input.shape()[0], input.shape()[1]];
    let data_vec = input.as_array().to_owned().into_raw_vec_and_offset().0;
    let input_data = TensorData::new(data_vec, shape);

    // B. Calcul sur
    let result_data = run_burn_logic::<MyBackend>(input_data, &device);

    let out_slice = result_data.as_slice::<f32>()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyTypeError, _>("Burn data conversion failed"))?;
    // On crée l'array 1D puis on le reshape en 2D pour correspondre à la shape d'origine
    let py_array_1d = PyArray::from_slice(py, out_slice);
    let py_array_2d = py_array_1d
        .reshape(shape)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Erreur de reshape: {:?}", e)))?;

    Ok(py_array_2d)
}





/// A Python module implemented in Rust.
/// We expose our Rust functions in the final Python module
#[pymodule]
fn deeplearning_library(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(is_prime, m)?)?;
    m.add_function(wrap_pyfunction!(py_computation,m)?)?;
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_test_false() {
        assert_eq!(is_prime(0), false);
        assert_eq!(is_prime(1), false);
        assert_eq!(is_prime(12), false)
    }
    #[test]
    fn simple_test_true() {
        assert_eq!(is_prime(2), true);
        assert_eq!(is_prime(3), true);
        assert_eq!(is_prime(41), true)
    }

}
