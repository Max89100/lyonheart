use std::cell::RefCell;
use std::rc::Rc;
use crate::tensor::CoreTensor;
use burn::module::Param;
use burn::backend::Autodiff;
use burn::backend::Wgpu;
use burn::tensor::{Tensor};
use pyo3::exceptions::PyRuntimeError;
use pyo3::{pyclass};
use pyo3::{prelude::*};
use numpy::{PyReadonlyArray2};
use burn::{optim::GradientsParams};
use std::sync::Mutex;
use lazy_static::lazy_static;

// type MyBackend = Autodiff<Wgpu>;
// static DEVICE: burn::backend::wgpu::WgpuDevice = burn::backend::wgpu::WgpuDevice::DiscreteGpu(0);

//FOR BENCHMARKING
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::NdArray;
type MyBackend = Autodiff<NdArray<f32>>;
type MyDevice = NdArrayDevice;
static DEVICE: burn::backend::ndarray::NdArrayDevice = burn::backend::ndarray::NdArrayDevice::Cpu;

lazy_static! {
    // Un stockage global sécurisé pour le dernier calcul de gradient
    pub static ref LATEST_GRADS: Mutex<Option<GradientsParams>> = Mutex::new(None);
}

pub type SharedParam<B, const D: usize> = Rc<RefCell<Param<Tensor<B, D>>>>;

#[pyclass(unsendable)]
#[derive(Clone)]
pub struct Parameter {
    pub param: SharedParam<MyBackend, 2>
}

#[pymethods]
impl Parameter {
    fn set(&mut self, input: PyReadonlyArray2<'_,f32>) {
        let tensor = CoreTensor::new(input).tensor;
        self._update_value(tensor);
    }

    #[getter]
    fn grad(&self) -> PyResult<Option<CoreTensor>> {
        let storage = LATEST_GRADS.lock()
            .map_err(|_| PyRuntimeError::new_err("Le verrou des gradients est corrompu"))?;

        if let Some(grads) = storage.as_ref() {
            // .borrow() permet de juste lire, .borrow_mut d'écrire aussi
            let param_borrow = self.param.borrow(); 
            let param_id = &param_borrow.id; // On accède à l'ID interne de Burn

            if let Some(grad_tensor) = grads.get(*param_id) {
                let autodiff_grad = Tensor::<MyBackend, 2>::from_inner(grad_tensor);
                return Ok(Some(CoreTensor { tensor: autodiff_grad }));
            } else {
                //println!("Gradient NON TROUVÉ pour l'ID: {:?}", param_id);
            }
        }
        Ok(None)
    }

    #[getter]
    pub fn tensor(&self) -> PyResult<CoreTensor> {
        Ok(CoreTensor { tensor: self._tensor()})
    }


    pub fn add_assign(&self, other: &CoreTensor) -> PyResult<()> {
        let updated_value = {
            let p = self.param.borrow();
            p.val().add(other.tensor.clone()).detach()
        };
        self._update_value(updated_value);
        Ok(())
    }

    pub fn sub_assign(&self, other: &CoreTensor) -> PyResult<()> {
        let updated_value = {
            let p = self.param.borrow(); // On demande l'accès en lecture
            p.val().sub(other.tensor.clone()).detach()
        };
        self._update_value(updated_value);
        Ok(())
    }

    pub fn mul_assign(&self, other: &CoreTensor) -> PyResult<()> {
        let updated_value = {
            let p = self.param.borrow();
            p.val().mul(other.tensor.clone()).detach()
        };
        self._update_value(updated_value);
        Ok(())
    }

    pub fn div_assign(&self, other: &CoreTensor) -> PyResult<()> {
        let updated_value = {
            let p = self.param.borrow();
            p.val().div(other.tensor.clone()).detach()
        };
        self._update_value(updated_value);
        Ok(())
    }

    //Magic Methods Python 
    fn __iadd__(&self, other: &CoreTensor) -> PyResult<()> {
        self.add_assign(other)
    }

    fn __isub__(&self, other: &CoreTensor) -> PyResult<()> {
        self.sub_assign(other)
    }

    fn __imul__(&self, other: &CoreTensor) -> PyResult<()> {
        self.mul_assign(other)
    }

    fn __itruediv__(&self, other: &CoreTensor) -> PyResult<()> {
        self.div_assign(other)
    }

}

impl Parameter {
    fn _update_value(&self, new_value: Tensor<MyBackend, 2>) {
        let mut p = self.param.borrow_mut();
        let id = p.id.clone();
        *p = Param::initialized(id, new_value).set_require_grad(true);
    }

    pub fn _alloc(value: Tensor<MyBackend, 2>) -> Self {
        let param = Param::from_tensor(value).set_require_grad(true);
        Self {
            param: Rc::new(RefCell::new(param)),
        }
    }

    pub fn _tensor(&self) -> Tensor<MyBackend,2> {
        self.param.borrow().val()
    }
}


