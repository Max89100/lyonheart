use crate::tensor::CoreTensor;

use burn::tensor::{Tensor};
use pyo3::{prelude::*, pyclass};


#[pyclass]
pub struct LossFunction;

#[pymethods]
impl LossFunction {
    #[staticmethod]
    fn mse(pred:&CoreTensor, target: &CoreTensor) -> PyResult<CoreTensor> {
        let pred_tensor: Tensor<_, 2> = pred.tensor.clone();
        let target_tensor: Tensor<_, 2> = target.tensor.clone();
        let error: Tensor<_, 2> = pred_tensor.sub(target_tensor).powf_scalar(2.0);
        let mse: Tensor<_, 1> = error.mean();
        Ok( CoreTensor {tensor: mse.reshape([1,1])})
    }

    #[staticmethod]
    fn cross_entropy(y_pred:&CoreTensor, y_target:&CoreTensor) -> PyResult<CoreTensor>{
        let y_pred_tensor = y_pred.tensor.clone();
        let y_target_tensor = y_target.tensor.clone();
        let eps  = 1e-10;
        let adjusted_tensor = y_pred_tensor.add_scalar(eps);
        let product = y_target_tensor * (adjusted_tensor.log());
        let loss = product.sum_dim(1).neg().mean();
        Ok(CoreTensor { tensor: loss.reshape([1,1])})
    }
}