use pyo3::{prelude::*};

pub mod tensor;
pub mod layers;
pub mod loss;


/// A Python module implemented in Rust.
/// We expose our Rust functions in the final Python module
#[pymodule]
fn deeplearning_library(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<tensor::GpuTensor>()?;
    m.add_class::<layers::Linear>()?;
    m.add_class::<loss::LossFunction>()?;
    Ok(())
}


// TESTS
#[cfg(test)]
mod tests {
    
}
