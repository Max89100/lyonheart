use pyo3::{prelude::*};
use pyo3_stub_gen::define_stub_info_gatherer;
use crate::parameter::clear_grads;

pub mod tensor;
pub mod layers;
pub mod loss;
pub mod parameter;

define_stub_info_gatherer!(stub_info);

/// A Python module implemented in Rust.
/// We expose our Rust functions in the final Python module
#[pymodule]
fn _lyonheart_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<tensor::CoreTensor>()?;
    m.add_class::<parameter::Parameter>()?;
    m.add_class::<layers::Linear>()?;
    m.add_class::<layers::ReLU>()?;
    m.add_class::<layers::Sigmoid>()?;
    m.add_class::<layers::Softmax>()?;
    m.add_class::<layers::InitMethod>()?;
    m.add_function(wrap_pyfunction!(clear_grads,m)?)?;
    Ok(())
}


// TESTS
#[cfg(test)]
mod tests {
}
