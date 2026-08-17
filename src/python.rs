use super::*;

#cfg[(feature = "python")]
use pyo3::prelude::*;
#cfg[(feature = "python")]
use pyo3::exceptions::*;
#cfg[(feature = "python")]
use pyo3::types::*;

pub mod chebyshev;
pub mod rootfinders;
pub mod polish;



#[cfg(feature = "python")]
#[pymodule]
mod pyacpr {
    #[pymodule_export]
    use super::chebyshev_coefficients_py;
}

pub fn chebyshev_coefficients_py<'py>(f: fn(f64), a: f64, b: f64) -> (DVector<f64>, f64) {
    chebyshev_adaptive()
}