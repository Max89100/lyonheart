use burn::backend::autodiff::grads::Gradients;
use std::sync::Mutex;
use lazy_static::lazy_static;

lazy_static! {
    // Un stockage global sécurisé pour le dernier calcul de gradient
    pub static ref LATEST_GRADS: Mutex<Option<Gradients>> = Mutex::new(None);
}