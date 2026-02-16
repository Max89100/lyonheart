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

    #[staticmethod]
    fn cross_entropy(y_pred:&GpuTensor, y_target:&GpuTensor) -> PyResult<GpuTensor>{
        let y_pred_tensor = y_pred.tensor.clone();
        let y_target_tensor = y_target.tensor.clone();
        let eps  = 1e-10;
        let adjusted_tensor = y_pred_tensor.add_scalar(eps);
        let product = y_target_tensor * (adjusted_tensor.log());
        let loss = product.sum_dim(1).neg().mean();
        Ok(GpuTensor { tensor: loss.reshape([1,1])})
    }
}