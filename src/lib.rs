use pyo3::{prelude::*};

pub mod tensor;
pub mod layers;
pub mod loss;
pub mod gradients;
pub mod parameter;


/// A Python module implemented in Rust.
/// We expose our Rust functions in the final Python module
#[pymodule]
fn _deeplearning_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<tensor::GpuTensor>()?;
    m.add_class::<layers::Linear>()?;
    m.add_class::<layers::ReLU>()?;
    m.add_class::<layers::Sigmoid>()?;
    m.add_class::<layers::Softmax>()?;
    m.add_class::<layers::InitMethod>()?;
    m.add_class::<loss::LossFunction>()?;
    Ok(())
}


// TESTS
#[cfg(test)]
mod tests {
    
}
