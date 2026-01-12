use pyo3::prelude::*;

/// Declared rust functions
#[pyfunction]
fn is_prime(num: u32) -> bool {
    match num {
        0 | 1 => false,
        _ => {
            let limit = (num as f32).sqrt() as u32; 

            (2..=limit).any(|i| num % i == 0) == false
        }
    }
}

/// A Python module implemented in Rust.
/// We expose our Rust functions in the final Python module
#[pymodule]
fn deeplearning_library(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(is_prime, m)?)?;
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_test_false() {
        assert_eq!(is_prime(0), false);
        assert_eq!(is_prime(1), false);
        assert_eq!(is_prime(12), false)
    }
    #[test]
    fn simple_test_true() {
        assert_eq!(is_prime(2), true);
        assert_eq!(is_prime(3), true);
        assert_eq!(is_prime(41), true)
    }

}
