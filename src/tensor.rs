use crate::gradients::{LATEST_GRADS};
use crate::parameter::Parameter;
use burn::{optim::GradientsParams};
use burn::tensor::{Tensor};
use burn::tensor::TensorData;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::Autodiff;
use burn::backend::Wgpu;
use pyo3::{prelude::*, pyclass};
use pyo3::exceptions::PyRuntimeError;
use numpy::{PyArray, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};

type MyBackend = Autodiff<Wgpu>;
type MyDevice = WgpuDevice;


#[pyclass]
pub struct GpuTensor { //A Renommer CoreTensor ou BackendTensor
    pub tensor: Tensor<MyBackend,2>,
}

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
    pub fn backward(&self,all_parameters: Vec<Parameter>) -> PyResult<()>{
        let grads = GpuTensor::_backward(&self.tensor);
        let mut grads_params = GradientsParams::new();
        for p in all_parameters {
            // On demande au "sac en vrac" : "As-tu le gradient pour ce paramètre précis ?"
            if let Some(grad) = p.inner.grad(&grads) {
                // Si oui, on le range dans notre dictionnaire avec l'ID du paramètre
                grads_params.register(p.inner.id, grad);
            }
        }
        //on stocke le dictionnaire des gradients 
        //dans une variable globale mutable
        let mut storage = LATEST_GRADS.lock()
        .map_err(|_| PyRuntimeError::new_err("Le verrou des gradients est corrompu (Mutex poisoned)"))?;
        *storage = Some(grads_params);
        Ok(())
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

    pub fn softmax(&self) -> PyResult<GpuTensor> {
        let tensor = GpuTensor::_softmax(&self.tensor);
        Ok(GpuTensor { tensor })
    }
    

    //opérateurs de base exposés à Python pour plus de flexibilité
    pub fn add(&self,other:&GpuTensor) -> PyResult<GpuTensor> {
        let tensor = self.tensor.clone().add(other.tensor.clone());
        Ok(GpuTensor { tensor })
    }
    pub fn add_assign(&mut self,other:&GpuTensor) -> PyResult<()>{
        self.tensor = self.tensor.clone().add(other.tensor.clone());
        Ok(())
    }
    
    pub fn add_scalar(&self,other:f32) -> PyResult<GpuTensor> {
        let tensor = self.tensor.clone().add_scalar(other);
        Ok(GpuTensor { tensor })
    }
    pub fn add_scalar_assign(&mut self,other:f32) -> PyResult<()> {
        self.tensor = self.tensor.clone().add_scalar(other);
        Ok(())
    }

    pub fn mul(&self,other:&GpuTensor) -> PyResult<GpuTensor> {
        let tensor = self.tensor.clone().mul(other.tensor.clone());
        Ok(GpuTensor { tensor })
    }
    pub fn mul_assign(&mut self, other:&GpuTensor) -> PyResult<()> {
        self.tensor = self.tensor.clone().mul(other.tensor.clone());
        Ok(())
    }
    
    pub fn mul_scalar(&self,other:f32) -> PyResult<GpuTensor> {
        let tensor = self.tensor.clone().mul_scalar(other);
        Ok(GpuTensor { tensor })
    }
    pub fn mul_scalar_assign(&mut self,other:f32) -> PyResult<()> {
        self.tensor = self.tensor.clone().mul_scalar(other);
        Ok(())
    }


    pub fn div(&self,other:&GpuTensor) -> PyResult<GpuTensor> {
        let tensor = self.tensor.clone().div(other.tensor.clone());
        Ok(GpuTensor { tensor })
    }
    pub fn div_assign(&mut self, other:&GpuTensor) -> PyResult<()> {
        self.tensor = self.tensor.clone().div(other.tensor.clone());
        Ok(())
    }

    pub fn sub(&self,other:&GpuTensor) -> PyResult<GpuTensor> {
        let tensor = self.tensor.clone().sub(other.tensor.clone());
        Ok(GpuTensor { tensor })
    }
    pub fn sub_assign(&mut self,other:&GpuTensor) -> PyResult<()> {
        println!("ID avant : {:?}", self.tensor);
        let new_tensor = self.tensor.clone().sub(other.tensor.clone());
        self.tensor = new_tensor;
        println!("ID après : {:?}", self.tensor);
        //self.tensor.inplace(|tensor| tensor.sub(other.tensor.clone()));
        Ok(())
    }

    pub fn matmul(&self, other:&GpuTensor) -> PyResult<GpuTensor> {
        let tensor = self.tensor.clone().matmul(other.tensor.clone());
        Ok(GpuTensor { tensor })
    }
    pub fn matmul_assign(&mut self, other:&GpuTensor) -> PyResult<()> {
        self.tensor = self.tensor.clone().matmul(other.tensor.clone());
        Ok(())
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

    fn _softmax(tensor: &Tensor<MyBackend,2>) -> Tensor<MyBackend,2> {
        let max = tensor.clone().max_dim(1);
        let centered = tensor.clone() - max;
        let exp = centered.exp();
        let sum = exp.clone().sum_dim(1);
        return exp / sum;
    }

    pub fn _backward(tensor:&Tensor<MyBackend,2>) -> Gradients {
        return tensor.backward()
    }
}