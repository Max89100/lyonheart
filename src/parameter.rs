use std::cell::RefCell;
use std::rc::Rc;

use crate::gradients::LATEST_GRADS;
use crate::tensor::GpuTensor;
use burn::module::Param;
use burn::backend::Autodiff;
use burn::backend::Wgpu;
use burn::tensor::{Tensor, TensorData};
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::{pyclass};
use pyo3::{prelude::*};
use burn::backend::wgpu::WgpuDevice;
type MyBackend = Autodiff<Wgpu>;
type MyDevice = WgpuDevice;


#[pyclass(unsendable)]
#[derive(Clone)]
pub struct Parameter {
    pub param: Rc<RefCell<Param<Tensor<MyBackend, 2>>>>
}

#[pymethods]
impl Parameter {
    #[new]
    fn new(input: PyReadonlyArray2<'_,f32>) -> Self {
        let shape: [usize; 2] = [input.shape()[0], input.shape()[1]];
        let data_vec: Vec<f32> = input.as_array().to_owned().into_raw_vec_and_offset().0;
        let input_data: TensorData = TensorData::new(data_vec, shape);
        Self { 
            param: Rc::new(RefCell::new(Param::from_tensor(Tensor::from_data(input_data, &MyDevice::DefaultDevice)))),
        }
    }

    #[getter]
fn grad(&self) -> PyResult<Option<GpuTensor>> {
    let storage = LATEST_GRADS.lock()
        .map_err(|_| PyRuntimeError::new_err("Le verrou des gradients est corrompu"))?;

    if let Some(grads) = storage.as_ref() {
        // ÉTAPE CLÉ : On emprunte le Param pour lire son ID
        // On utilise .borrow() car on veut juste LIRE
        let param_borrow = self.param.borrow(); 
        let param_id = &param_borrow.id; // On accède à l'ID interne de Burn

        if let Some(grad_tensor) = grads.get(*param_id) {
            let autodiff_grad = Tensor::<MyBackend, 2>::from_inner(grad_tensor);
            return Ok(Some(GpuTensor { tensor: autodiff_grad }));
        } else {
            //println!("Gradient NON TROUVÉ pour l'ID: {:?}", param_id);
        }
    }
    Ok(None)
}

    #[getter]
    fn tensor(&self) -> PyResult<GpuTensor> {
        Ok(GpuTensor { tensor: self.param.borrow().val() })
    }

    pub fn sub_assign(&self, other: &GpuTensor) -> PyResult<()> {
        let mut p = self.param.borrow_mut(); // On demande l'accès en écriture
        let updated_value = p.val().sub(other.tensor.clone()).detach();
        let id = p.id.clone();
        // On remplace le Param à l'intérieur du RefCell
        *p = Param::initialized(id,updated_value).set_require_grad(true);
        Ok(())
    }
}

