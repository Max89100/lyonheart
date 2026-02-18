use crate::tensor::GpuTensor;
use crate::gradients::LATEST_GRADS;


use burn::prelude::ToElement;
use burn::tensor::{Tensor};
use burn::backend::wgpu::WgpuDevice;
use burn::backend::Autodiff;
use burn::backend::Wgpu;
use pyo3::{prelude::*, pyclass};
use pyo3::exceptions::PyRuntimeError;

type MyBackend = Autodiff<Wgpu>;
type MyDevice = WgpuDevice;

#[pyclass]
pub struct Linear {
    pub weights: Tensor<MyBackend,2>,
    pub bias : Tensor<MyBackend,2>,
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
                    weights: Tensor::<MyBackend,2>::random([input_size,output_size], burn::tensor::Distribution::Uniform(-alpha.to_f64(), alpha.to_f64()), &MyDevice::DefaultDevice).set_require_grad(true),
                    bias: Tensor::<MyBackend, 2>::zeros([1, output_size], &MyDevice::DefaultDevice).set_require_grad(true),
                }
            }
            // Kaiming (normal) permet d'optimiser l'initialsiation des couches
            // qui utilisent ReLU comme fonction d'activation
            InitMethod::Kaiming => {
                let sigma = (2.0 / input_size.to_f32()).sqrt();
                return Self {
                    weights: Tensor::<MyBackend,2>::random([input_size,output_size], burn::tensor::Distribution::Normal(0.0, sigma.to_f64()), &MyDevice::DefaultDevice).set_require_grad(true),
                    bias: Tensor::<MyBackend, 2>::full([1, output_size], 0.01, &MyDevice::DefaultDevice).set_require_grad(true),
                }
            }
            InitMethod::Default => {
                return Self {
                    weights: Tensor::<MyBackend,2>::random([input_size,output_size], burn::tensor::Distribution::Default, &MyDevice::DefaultDevice).set_require_grad(true),
                    bias: Tensor::<MyBackend, 2>::zeros([1, output_size], &MyDevice::DefaultDevice).set_require_grad(true),
                }
            }
        }
    }
    fn forward(&self, input: &GpuTensor) -> PyResult<GpuTensor> {
        let y: Tensor<_, 2> = input.tensor.clone().matmul(self.weights.clone()) + self.bias.clone();
        Ok(GpuTensor {tensor: y})
    }
     fn update(&mut self, learning_rate: f32) -> PyResult<()> {
        let storage = LATEST_GRADS.lock()
        .map_err(|_| PyRuntimeError::new_err("Le verrou des gradients est corrompu (Mutex poisoned)"))?;
        //grads est une sorte de dictionnaire avec tous les gradients.
        //La formule de l'update est : poids = poids - gradxlr
        if let Some(grads) = storage.as_ref() {
            if let Some(grad_weights) = self.weights.grad(grads) {
                let scaled_grad: Tensor<_, 2> = grad_weights.mul_scalar(learning_rate);
                self.weights = self.weights.clone().sub(Tensor::from_inner(scaled_grad));
            }
            if let Some(grad_bias) = self.bias.grad(grads) {
                let scaled_bias: Tensor<_, 2> = grad_bias.mul_scalar(learning_rate);
                self.bias = self.bias.clone().sub(Tensor::from_inner(scaled_bias));
            }
            Ok(())
        }
        else {
            Err(PyRuntimeError::new_err("Tentative d'update avant d'avoir appelé backward() !"))
        }
        
    }
}

#[pymethods]
impl ReLU {
    #[new]
    pub fn new() -> Self{
        ReLU {}
    }

    pub fn forward(&self,input: &GpuTensor) -> PyResult<GpuTensor> {
        input.relu()
    }
}

#[pymethods]
impl Sigmoid {
    #[new]
    pub fn new() -> Self{
        Sigmoid{}
    }
    pub fn forward(&self,input: &GpuTensor) -> PyResult<GpuTensor> {
        input.sigmoid()
    }
}
#[pymethods]
impl Softmax {
    #[new]
    pub fn new() -> Self{
        Softmax{}
    }

    pub fn forward(&self,input: &GpuTensor) -> PyResult<GpuTensor> {
        input.softmax()
    }
}