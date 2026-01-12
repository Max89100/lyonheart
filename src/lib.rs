use pyo3::prelude::*;
use burn::tensor::{Tensor, backend::Backend};
use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;

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


// Elle ne sait pas si c'est du GPU, du CPU ou du TPU. Elle sait juste faire des maths.
// Fonction "Générique"
fn computation<B: Backend>(device: &B::Device) {
    let tensor1: Tensor<B, 2> = Tensor::from_floats([[2.0, 3.0], [4.0, 5.0]], device);
    let tensor2 = Tensor::ones_like(&tensor1);
    let result = tensor1 + tensor2;
    println!("Résultat du calcul : {}", result);
}

// Ici, on "verrouille" le type pour Python.
#[pyfunction]
fn py_computation() -> PyResult<()> {
    // On force l'utilisation de Wgpu (le GPU via WebGPU)
    type MyBackend = Wgpu; 
    let device = WgpuDevice::DefaultDevice;
    // On appelle la fonction générique en lui disant d'utiliser MyBackend
    computation::<MyBackend>(&device);
    Ok(())
}




/// A Python module implemented in Rust.
/// We expose our Rust functions in the final Python module
#[pymodule]
fn deeplearning_library(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(is_prime, m)?)?;
    m.add_function(wrap_pyfunction!(py_computation,m)?)?;
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
