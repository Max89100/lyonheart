use crate::tensor::CoreTensor;
use crate::parameter::Parameter;
use burn::prelude::ToElement;
use burn::tensor::{Tensor};
use burn::backend::Autodiff;
use pyo3::{prelude::*, pyclass};

// use burn::backend::wgpu::WgpuDevice;
// use burn::backend::Wgpu;
// type MyBackend = Autodiff<Wgpu>;
// type MyDevice = WgpuDevice;
// static DEVICE: MyDevice = MyDevice::DiscreteGpu(0);

//FOR BENCHMARKING
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::NdArray;
type MyBackend = Autodiff<NdArray<f32>>;
// type MyBackend = NdArray<f32>; //INFERENCE MODE
type MyDevice = NdArrayDevice;
static DEVICE: MyDevice = MyDevice::Cpu;


#[pyclass(unsendable)]
pub struct Linear {
    pub weight: Parameter,
    pub bias : Parameter
}

#[pyclass]
#[derive(Clone)]
pub enum InitMethod {
    Xavier,
    Kaiming,
    Default
}

#[pyclass]
pub struct ReLU();
#[pyclass]
pub struct Sigmoid();
#[pyclass]
pub struct Softmax();

#[pymethods]
impl Linear {
    #[new]
    fn new(input_size:usize, output_size:usize, initializer:InitMethod) -> Self {
        match initializer {
            // Xavier (uniform) permet d'optimiser l'initialisation des couches
            // qui utilisent Sigmoid ou Tanh comme fonction d'activation
            InitMethod::Xavier => {
                let alpha = (6.0 / (input_size + output_size).to_f32()).sqrt();
                return Self {
                    weight: Parameter::_alloc(Tensor::<MyBackend,2>::random([input_size,output_size], burn::tensor::Distribution::Uniform(-alpha.to_f64(), alpha.to_f64()), &DEVICE)),
                    bias: Parameter::_alloc(Tensor::<MyBackend, 2>::zeros([1, output_size], &DEVICE))

                }
            }
            // Kaiming (normal) permet d'optimiser l'initialsiation des couches
            // qui utilisent ReLU comme fonction d'activation
            InitMethod::Kaiming => {
                let sigma = (2.0 / input_size.to_f32()).sqrt();
                return Self {
                    weight: Parameter::_alloc(Tensor::<MyBackend,2>::random([input_size,output_size], burn::tensor::Distribution::Normal(0.0, sigma.to_f64()), &DEVICE)),
                    bias: Parameter::_alloc(Tensor::<MyBackend, 2>::full([1, output_size], 0.01, &DEVICE))
                }
            }
            InitMethod::Default => {
                return Self {
                    weight: Parameter::_alloc(Tensor::<MyBackend,2>::random([input_size,output_size], burn::tensor::Distribution::Default, &DEVICE)),
                    bias: Parameter::_alloc(Tensor::<MyBackend, 2>::zeros([1, output_size], &DEVICE))
                }
            }
        }
    }

    // fn forward(&self, input: &CoreTensor) -> PyResult<CoreTensor> {
    //     let w = self.weight._tensor();
    //     let b = self.bias._tensor();
    //     let y = input.tensor.clone().matmul(w).add(b);
    //     Ok(CoreTensor { tensor: y })
    // }
    fn forward<'py>(&self, input: &Bound<'py, CoreTensor>) -> PyResult<Bound<'py, CoreTensor>> {
        let w = self.weight._tensor();
        let b = self.bias._tensor();
        let y = input.borrow().tensor.clone(); 
        let tensor = y.matmul(w).add(b);
        Bound::new(input.py(), CoreTensor { tensor })
    }

    #[getter]
    fn parameters(&self) -> PyResult<Vec<Parameter>>{
        Ok(vec![
            self.weight.clone(),
            self.bias.clone(),
        ])
    }

    #[getter]
    fn weight(&self) -> PyResult<Parameter> {
        Ok(self.weight.clone())
    }

    #[getter]
    fn bias(&self) -> PyResult<Parameter> {
        Ok(self.bias.clone())
    }

}


#[pymethods]
impl ReLU {
    #[new]
    pub fn new() -> Self{
        ReLU {}
    }

    pub fn forward(&self,input: &CoreTensor) -> PyResult<CoreTensor> {
        input.relu()
    }
}

#[pymethods]
impl Sigmoid {
    #[new]
    pub fn new() -> Self{
        Sigmoid{}
    }
    pub fn forward(&self,input: &CoreTensor) -> PyResult<CoreTensor> {
        input.sigmoid()
    }
}
#[pymethods]
impl Softmax {
    #[new]
    pub fn new() -> Self{
        Softmax{}
    }

    pub fn forward(&self,input: &CoreTensor) -> PyResult<CoreTensor> {
        input.softmax()
    }
}