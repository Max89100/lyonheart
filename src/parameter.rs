use crate::gradients::{LATEST_GRADS};

use burn::module::Param;
use burn::backend::Autodiff;
use burn::backend::Wgpu;
use burn::tensor::{Tensor, TensorData};
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::{pyclass};
use pyo3::{prelude::*};
use burn::backend::wgpu::WgpuDevice;
use pyo3::exceptions::PyRuntimeError;

type MyBackend = Autodiff<Wgpu>;
type MyDevice = WgpuDevice;


#[pyclass(unsendable)]
#[derive(Clone)]
pub struct Parameter {
    pub inner: Param<Tensor<MyBackend,2>>
}

#[pymethods]
impl Parameter {
    #[new]
    fn new(input: PyReadonlyArray2<'_,f32>) -> Self {
        let shape: [usize; 2] = [input.shape()[0], input.shape()[1]];
        let data_vec: Vec<f32> = input.as_array().to_owned().into_raw_vec_and_offset().0;
        let input_data: TensorData = TensorData::new(data_vec, shape);
        Self { 
            inner: Param::from_tensor(Tensor::from_data(input_data, &MyDevice::DefaultDevice)),
        }
    }

    // #[getter]
    // fn grad(&self) -> PyResult<Option<GpuTensor>>{
    //     let storage = LATEST_GRADS.lock()
    //     .map_err(|_| PyRuntimeError::new_err("Le verrou des gradients est corrompu (Mutex poisoned)"))?;
    //     //grads est une sorte de dictionnaire avec tous les gradients.
    //     //La formule de l'update est : poids = poids - gradxlr
    //     if let Some(grads) = storage.as_ref() {
    //         if let Some(grad_tensor) = self.tensor.grad(grads) {
    //             let autodiff_grad = Tensor::<MyBackend,2>::from_inner(grad_tensor);
    //             return Ok(Some(GpuTensor{tensor:autodiff_grad}));
    //         }
    //         else {
    //             println!("Gradient NON TROUVÉ dans le dictionnaire pour cet ID.");
    //         }
    //     }
    //     Ok(None)
    // }
}

