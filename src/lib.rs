use std::ops::Mul;

use burn::backend::autodiff::grads::Gradients;
use burn::tensor::TensorData;
use pyo3::{prelude::*, pyclass};
use burn::tensor::{Tensor, backend::Backend};
use burn::backend::wgpu::WgpuDevice;
use burn::backend::Autodiff;
use burn::backend::Wgpu;
use numpy::{PyArray, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
type MyBackend = Autodiff<Wgpu>;
type MyDevice = WgpuDevice;


/// Declared rust functions
fn run_burn_logic<B: Backend>(data: TensorData, device: &B::Device) -> TensorData {
    // 1. On crée le tenseur à partir du TensorData
    let tensor1: Tensor<B, 2> = Tensor::from_data(data, device);
    let tensor2: Tensor<B, 2> = Tensor::ones_like(&tensor1);
    
    // 2. On fait l'opération et on repasse en TensorData pour le retour
    (tensor1 + tensor2).into_data()
}

//CLASSES
#[pyclass]
pub struct GpuTensor {
    pub tensor: Tensor<MyBackend,2>,
}
#[pyclass]
pub struct Linear {
    pub weights: Tensor<MyBackend,2>,
    pub bias : Tensor<MyBackend,2>,
}

#[pyclass]
pub struct PyGradients {
    pub grads: burn::backend::autodiff::grads::Gradients,
}

#[pyclass]
pub struct LossFunction;

#[pymethods]
impl GpuTensor {
    #[new]
    fn new(input: PyReadonlyArray2<'_,f32>) -> Self {
        let shape: [usize; 2] = [input.shape()[0], input.shape()[1]];
        let data_vec: Vec<f32> = input.as_array().to_owned().into_raw_vec_and_offset().0;
        let input_data: TensorData = TensorData::new(data_vec, shape);
        Self { 
            tensor: Tensor::from_data(input_data, &MyDevice::DefaultDevice),
        }
    }
    
    
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let tensor_data: TensorData = self.tensor.clone().into_data();
        let out_slice: &[f32] = tensor_data.as_slice::<f32>()
            .map_err(|_e: burn::tensor::DataError| PyErr::new::<pyo3::exceptions::PyTypeError, _>("Burn data conversion failed"))?;
        let dims: Vec<usize> = self.tensor.shape().dims;
        let shape: [usize; 2] = [dims[0], dims[1]];
        let py_array: Bound<'_, PyArray<f32, numpy::ndarray::Dim<[usize; 2]>>> = PyArray::from_slice(py, out_slice)
            .reshape(shape)
            .map_err(|e: PyErr| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Erreur de reshape: {:?}", e)))?;
        Ok(py_array)
    }

    

    //activation functions
    pub fn relu(&self) -> PyResult<GpuTensor> {
        let tensor: Tensor<_, 2> = GpuTensor::_relu(&self.tensor);
        Ok(GpuTensor { tensor })
    }

    pub fn sigmoid(&self) -> PyResult<GpuTensor> {
        let tensor: Tensor<_, 2> = GpuTensor::_sigmoid(&self.tensor);
        Ok(GpuTensor { tensor } )
    }

    pub fn tanh(&self) -> PyResult<GpuTensor> { 
        let tensor: Tensor<_, 2> = GpuTensor::_tanh(&self.tensor);
        Ok(GpuTensor { tensor})
    }

    pub fn backward(&self) -> PyResult<PyGradients>{
        let grads = Self::_backward(&self.tensor);
        Ok(PyGradients {grads})
    }
        
}

impl GpuTensor {
    fn _relu(tensor:&Tensor<MyBackend,2>) -> Tensor<MyBackend,2> {
        return tensor.clone().clamp_min(0.0);
    }

    fn _sigmoid(tensor:&Tensor<MyBackend,2>) -> Tensor<MyBackend,2> {
        return tensor.clone().neg().exp().add_scalar(1.0).recip();
    }

    fn _tanh(tensor:&Tensor<MyBackend,2>) -> Tensor<MyBackend,2> {
        return Self::_sigmoid(&tensor.clone().mul_scalar(2.0)).mul_scalar(2.0).add_scalar(-1.0);
    }

    fn _backward(tensor:&Tensor<MyBackend,2>) -> Gradients {
        return tensor.clone().backward()
    }
}


#[pymethods]
impl Linear {
    #[new]
    fn new(input_size:usize, output_size:usize) -> Self {
        Self {
            weights: Tensor::<MyBackend,2>::random([input_size,output_size], burn::tensor::Distribution::Default, &MyDevice::DefaultDevice),
            bias: Tensor::<MyBackend, 2>::zeros([1, output_size], &MyDevice::DefaultDevice),
        }
    }
    fn forward(&self, input: &GpuTensor) -> PyResult<GpuTensor> {
        let y: Tensor<_, 2> = input.tensor.clone().matmul(self.weights.clone()) + self.bias.clone();
        Ok(GpuTensor {tensor: y})
    }

    fn update(&mut self, learning_rate: f32, pygrads: &PyGradients) {
        //Récupère le gradient du tenseur courant. 
        //grads est une sorte de dictionnaire avec tous les gradients.
        if let Some(grad_weights) = self.weights.grad(&pygrads.grads) {
            let scaled_grad: Tensor<_, 2> = grad_weights.mul_scalar(learning_rate);
            self.weights = self.weights.clone().sub(Tensor::from_inner(scaled_grad));
        }
        if let Some(grad_bias) = self.bias.grad(&pygrads.grads) {
            let scaled_bias: Tensor<_, 2> = grad_bias.mul_scalar(learning_rate);
            self.bias = self.weights.clone().sub(Tensor::from_inner(scaled_bias));
        }
    }
}

#[pymethods]
impl LossFunction {
    #[staticmethod]
    fn mse(pred:&GpuTensor, target: &GpuTensor) -> PyResult<GpuTensor> {
        let pred_tensor: Tensor<_, 2> = pred.tensor.clone();
        let target_tensor: Tensor<_, 2> = target.tensor.clone();
        let error: Tensor<_, 2> = pred_tensor.sub(target_tensor).powf_scalar(2.0);
        let mse: Tensor<_, 1> = error.mean();
        Ok( GpuTensor {tensor: mse.reshape([1,1])})
    }
}


//Python Functions
#[pyfunction]
fn py_computation<'py>(py: Python<'py>, input: PyReadonlyArray2<'_, f32>) -> PyResult<Bound<'py, PyArray2<f32>>> {

    // A. Extraction : NumPy -> TensorData
    let shape: [usize; 2] = [input.shape()[0], input.shape()[1]];
    let data_vec: Vec<f32> = input.as_array().to_owned().into_raw_vec_and_offset().0;
    let input_data: TensorData = TensorData::new(data_vec, shape);

    // B. Calcul sur
    let result_data: TensorData = run_burn_logic::<MyBackend>(input_data, &MyDevice::DefaultDevice);

    let out_slice: &[f32] = result_data.as_slice::<f32>()
        .map_err(|_e: burn::tensor::DataError| PyErr::new::<pyo3::exceptions::PyTypeError, _>("Burn data conversion failed"))?;
    // On crée l'array 1D puis on le reshape en 2D pour correspondre à la shape d'origine
    let py_array_1d: Bound<'_, PyArray<f32, numpy::ndarray::Dim<[usize; 1]>>> = PyArray::from_slice(py, out_slice);
    let py_array_2d: Bound<'_, PyArray<f32, numpy::ndarray::Dim<[usize; 2]>>> = py_array_1d
        .reshape(shape)
        .map_err(|e: PyErr| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Erreur de reshape: {:?}", e)))?;

    Ok(py_array_2d)
}

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



/// A Python module implemented in Rust.
/// We expose our Rust functions in the final Python module
#[pymodule]
fn deeplearning_library(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(is_prime, m)?)?;
    m.add_function(wrap_pyfunction!(py_computation,m)?)?;
    m.add_class::<GpuTensor>()?;
    m.add_class::<Linear>()?;
    Ok(())
}



// TESTS
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
