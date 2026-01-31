use crate::tensor::GpuTensor;

use burn::tensor::{Tensor};
use pyo3::{prelude::*, pyclass};


#[pyclass]
pub struct LossFunction;

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