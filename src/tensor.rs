use crate::parameter::LATEST_GRADS;
use crate::parameter::Parameter;
use burn::tensor::Distribution;
use burn::tensor::Shape;
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
use std::ops::Sub;

// type MyBackend = Autodiff<Wgpu>;
// type MyDevice = WgpuDevice;
// static DEVICE: burn::backend::wgpu::WgpuDevice = burn::backend::wgpu::WgpuDevice::DiscreteGpu(0); 

//FOR BENCHMARKING
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::NdArray;
type MyBackend = Autodiff<NdArray<f32>>;
type MyDevice = NdArrayDevice;
static DEVICE: burn::backend::ndarray::NdArrayDevice = burn::backend::ndarray::NdArrayDevice::Cpu;


#[pyclass]
pub struct CoreTensor {
    pub tensor: Tensor<MyBackend,2>,
}

impl Sub for CoreTensor {
    type Output = CoreTensor;
    fn sub(self, rhs: Self) -> Self::Output {
        CoreTensor { tensor: self.tensor.sub(rhs.tensor) }
    }
}

#[pymethods]
impl CoreTensor {
    #[new]
    pub fn new(input: PyReadonlyArray2<'_,f32>) -> Self {
        let shape: [usize; 2] = [input.shape()[0], input.shape()[1]];
        let data_vec: Vec<f32> = input.as_array().to_owned().into_raw_vec_and_offset().0;
        let input_data: TensorData = TensorData::new(data_vec, shape);
        Self { 
            //tensor: Tensor::from_data(input_data, &MyDevice::DefaultDevice),
            tensor: Tensor::from_data(input_data, &DEVICE),
        }
    }

    #[staticmethod]
    fn zeros(shape: Vec<usize>) -> PyResult<CoreTensor> {
        let tensor = Tensor::zeros(Shape::from(shape), &DEVICE);
        Ok(CoreTensor { tensor })
    }

    #[staticmethod]
    fn ones(shape: Vec<usize>) -> PyResult<CoreTensor> {
        let tensor = Tensor::ones(shape, &DEVICE);
        Ok(CoreTensor { tensor })
    }
    #[staticmethod]
    fn randn(shape: Vec<usize>) -> PyResult<CoreTensor> {
        let tensor = Tensor::random(shape, Distribution::Normal(0.0, 1.0), &DEVICE);
        Ok(CoreTensor { tensor })
    }
    #[staticmethod]
    fn rand(shape: Vec<usize>) -> PyResult<CoreTensor> {
        let tensor = Tensor::random(shape, Distribution::Default, &DEVICE);
        Ok(CoreTensor { tensor })
    }

    fn zeros_like(&self) -> PyResult<CoreTensor> {
        let tensor = Tensor::zeros_like(&self.tensor);
        Ok(CoreTensor { tensor })
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
        let grads = CoreTensor::_backward(&self.tensor);
        let mut grads_params = GradientsParams::new();

        for p in all_parameters {
            let p_inner = p.param.borrow(); 
            if let Some(grad) = p_inner.grad(&grads) {
                grads_params.register(p_inner.id.clone(), grad);
            }
        }
        let mut storage = LATEST_GRADS.lock()
        .map_err(|_| PyRuntimeError::new_err("Le verrou des gradients est corrompu (Mutex poisoned)"))?;
        *storage = Some(grads_params);
        Ok(())
    }

    


    // Opérations de base
    fn add(&self,other:&CoreTensor) -> PyResult<CoreTensor> {
        let tensor = self.tensor.clone().add(other.tensor.clone());
        Ok(CoreTensor { tensor })
    }

    fn add_assign(&mut self,other:&CoreTensor) -> PyResult<()>{
        self.tensor = self.tensor.clone().add(other.tensor.clone());
        Ok(())
    }

    fn add_scalar(&self,other:f32) -> PyResult<CoreTensor> {
        let tensor = self.tensor.clone().add_scalar(other);
        Ok(CoreTensor { tensor })
    }
    fn add_scalar_assign(&mut self,other:f32) -> PyResult<()> {
        self.tensor = self.tensor.clone().add_scalar(other);
        Ok(())
    }

    fn sub(&self, other: &CoreTensor) -> PyResult<CoreTensor> {
        Ok(CoreTensor { 
            tensor: self.tensor.clone().sub(other.tensor.clone()) 
        })
    }

    fn sub_assign(&mut self,other:&CoreTensor) -> PyResult<()> {
        let new_tensor = self.tensor.clone().sub(other.tensor.clone());
        self.tensor = new_tensor;
        Ok(())
    }

    fn sub_scalar(&self,other:f32) -> PyResult<CoreTensor> {
        let tensor = self.tensor.clone().sub_scalar(other);
        Ok(CoreTensor { tensor })
    }

    fn sub_scalar_assign(&mut self,other:f32) -> PyResult<()> {
        self.tensor = self.tensor.clone().sub_scalar(other);
        Ok(())
    }
    
    fn mul(&self,other:&CoreTensor) -> PyResult<CoreTensor> {
        let tensor = self.tensor.clone().mul(other.tensor.clone());
        Ok(CoreTensor { tensor })
    }

    fn mul_assign(&mut self, other:&CoreTensor) -> PyResult<()> {
        self.tensor = self.tensor.clone().mul(other.tensor.clone());
        Ok(())
    }

    fn mul_scalar(&self,other:f32) -> PyResult<CoreTensor> {
        let tensor = self.tensor.clone().mul_scalar(other);
        Ok(CoreTensor { tensor })
    }

    fn mul_scalar_assign(&mut self,other:f32) -> PyResult<()> {
        self.tensor = self.tensor.clone().mul_scalar(other);
        Ok(())
    }

    fn div(&self,other:&CoreTensor) -> PyResult<CoreTensor> {
        let tensor = self.tensor.clone().div(other.tensor.clone());
        Ok(CoreTensor { tensor })
    }

    fn div_assign(&mut self, other:&CoreTensor) -> PyResult<()> {
        self.tensor = self.tensor.clone().div(other.tensor.clone());
        Ok(())
    }

    fn div_scalar(&self,other:f32) -> PyResult<CoreTensor> {
        let tensor = self.tensor.clone().div_scalar(other);
        Ok(CoreTensor { tensor })
    }

    fn div_scalar_assign(&mut self,other:f32) -> PyResult<()> {
        self.tensor = self.tensor.clone().div_scalar(other);
        Ok(())
    }
    
    fn pow(&self, other: &CoreTensor) -> PyResult<CoreTensor> {
        let tensor = self.tensor.clone().powf(other.tensor.clone());
        Ok(CoreTensor { tensor })
    }

    fn pow_scalar(&self, other: f32) -> PyResult<CoreTensor> {
        let tensor = self.tensor.clone().powf_scalar(other);
        Ok(CoreTensor { tensor })
    }

    fn log(&self) -> PyResult<CoreTensor> {
        let tensor = self.tensor.clone().log();
        Ok(CoreTensor { tensor })
    }

    fn exp(&self) -> PyResult<CoreTensor> {
        let tensor = self.tensor.clone().exp();
        Ok(CoreTensor { tensor })
    }


    fn neg(&self) -> PyResult<CoreTensor> {
        let tensor = -self.tensor.clone();
        Ok(CoreTensor { tensor })
    }


    fn matmul(&self, other:&CoreTensor) -> PyResult<CoreTensor> {
        let tensor = self.tensor.clone().matmul(other.tensor.clone());
        Ok(CoreTensor { tensor })
    }
    
    fn matmul_assign(&mut self, other:&CoreTensor) -> PyResult<()> {
        self.tensor = self.tensor.clone().matmul(other.tensor.clone());
        Ok(())
    }

    fn mean(&self) -> PyResult<CoreTensor> {
        let tensor = self.tensor.clone().mean();
        Ok(CoreTensor { tensor: tensor.reshape([1,1]) })
    }

    fn sum(&self) -> PyResult<CoreTensor> {
        let tensor = self.tensor.clone().sum();
        Ok(CoreTensor { tensor: tensor.reshape([1,1]) })
    }
    fn sum_dim(&self, dim: i32) -> PyResult<CoreTensor> {
        let tensor = self.tensor.clone().sum_dim(dim);
        Ok(CoreTensor { tensor })
    }

    fn max_dim(&self, dim: i32) -> PyResult<CoreTensor> {
        let tensor = self.tensor.clone().max_dim(dim);
        Ok(CoreTensor { tensor })
    }




    //Magic Methods Python (surcharge des opérateurs python)
    fn __sub__(&self, other: &CoreTensor) -> PyResult<CoreTensor> {
        self.sub(other)
    }

    fn __add__(&self,other:&CoreTensor) -> PyResult<CoreTensor> {
        self.add(other)
    }

    fn __mul__(&self,other:&CoreTensor) -> PyResult<CoreTensor> {
        self.mul(other)
    }

    fn __truediv__(&self,other:&CoreTensor) -> PyResult<CoreTensor> {
        self.div(other)
    }

    fn __matmul__(&self, other:&CoreTensor) -> PyResult<CoreTensor> {
        self.matmul(other)
    }
    
    fn __neg__(&self) -> PyResult<CoreTensor> {
        self.neg()
    }
    
    fn __pow__(&self, other: &CoreTensor, _modulo:Option<PyObject>) -> PyResult<CoreTensor> {
        self.pow(other)
    }
    
    pub fn __iadd__(&mut self,other:&CoreTensor) -> PyResult<()>{
        self.add_assign(other)
    }

    pub fn __isub__(&mut self,other:&CoreTensor) -> PyResult<()> {
        self.sub_assign(other)
    }

    pub fn __itruediv__(&mut self, other:&CoreTensor) -> PyResult<()> {
        self.div_assign(other)
    }

    pub fn __imul__(&mut self, other:&CoreTensor) -> PyResult<()> {
        self.mul_assign(other)
    }

    fn __repr__(&self) -> String {
        let data = self.tensor.clone().into_data();
        format!("CoreTensor(\n{}\n, device={:?})", data, DEVICE)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }



    //activation functions
    pub fn relu(&self) -> PyResult<CoreTensor> {
        let tensor: Tensor<_, 2> = CoreTensor::_relu(&self.tensor);
        Ok(CoreTensor { tensor })
    }

    pub fn sigmoid(&self) -> PyResult<CoreTensor> {
        let tensor: Tensor<_, 2> = CoreTensor::_sigmoid(&self.tensor);
        Ok(CoreTensor { tensor } )
    }

    pub fn tanh(&self) -> PyResult<CoreTensor> { 
        let tensor: Tensor<_, 2> = CoreTensor::_tanh(&self.tensor);
        Ok(CoreTensor { tensor})
    }

    pub fn softmax(&self) -> PyResult<CoreTensor> {
        let tensor = CoreTensor::_softmax(&self.tensor);
        Ok(CoreTensor { tensor })
    }
    
}

impl CoreTensor {
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