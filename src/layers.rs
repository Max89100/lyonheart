use crate::tensor::GpuTensor;

use burn::tensor::{Tensor};
use burn::backend::wgpu::WgpuDevice;
use burn::backend::Autodiff;
use burn::backend::Wgpu;
use pyo3::{prelude::*, pyclass};

type MyBackend = Autodiff<Wgpu>;
type MyDevice = WgpuDevice;

#[pyclass]
pub struct Linear {
    pub weights: Tensor<MyBackend,2>,
    pub bias : Tensor<MyBackend,2>,
}

#[pymethods]
impl Linear {
    #[new]
    fn new(input_size:usize, output_size:usize) -> Self {
        Self {
            weights: Tensor::<MyBackend,2>::random([input_size,output_size], burn::tensor::Distribution::Default, &MyDevice::DefaultDevice).set_require_grad(true),
            bias: Tensor::<MyBackend, 2>::zeros([1, output_size], &MyDevice::DefaultDevice).set_require_grad(true),
        }
    }
    fn forward(&self, input: &GpuTensor) -> PyResult<GpuTensor> {
        let y: Tensor<_, 2> = input.tensor.clone().matmul(self.weights.clone()) + self.bias.clone();
        Ok(GpuTensor {tensor: y})
    }
     fn update(&mut self, learning_rate: f32, loss: &GpuTensor) {
         let grads = GpuTensor::_backward(&loss.tensor);
        //grads est une sorte de dictionnaire avec tous les gradients.
        //La formule de l'update est : poids = poids - gradxlr
        if let Some(grad_weights) = self.weights.grad(&grads) {
            let scaled_grad: Tensor<_, 2> = grad_weights.mul_scalar(learning_rate);
            self.weights = self.weights.clone().sub(Tensor::from_inner(scaled_grad));
        }
        if let Some(grad_bias) = self.bias.grad(&grads) {
            let scaled_bias: Tensor<_, 2> = grad_bias.mul_scalar(learning_rate);
            self.bias = self.weights.clone().sub(Tensor::from_inner(scaled_bias));
        }
    }
}