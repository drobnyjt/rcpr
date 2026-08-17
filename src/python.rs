use super::*;

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::exceptions::*;
#[cfg(feature = "python")]
use pyo3::types::*;

use crate::chebyshev::{chebyshev_adaptive, chebyshev_approximate, chebyshev_subdivide};
use crate::rootfinders::*;

#[cfg(feature = "python")]
#[pymodule]
mod pyacpr {
    #[pymodule_export]
    use super::chebyshev_coefficients_py;

    #[pymodule_export]
    use super::chebyshev_approximate_py;

    #[pymodule_export]
    use super::chebyshev_subdivide_py;

    #[pymodule_export]
    use super::find_roots_py;
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (f, a, b, epsilon=1e-6, n0=2, n_max=1000))]
pub fn chebyshev_coefficients_py<'py>(_python: Python<'py>, f: &Bound<'py, PyAny>, a: f64, b: f64, epsilon: f64, n0: usize, n_max: usize) -> PyResult<(Vec<f64>, f64)> {
    
    let f = {|x| f.clone().call1((x,)).unwrap().extract().unwrap()};
    let (result, error) = chebyshev_adaptive(&f, a, b, n0, epsilon, n_max);
    Ok((result.iter().map(|x| *x).collect::<Vec<f64>>(), error))
}

#[cfg(feature = "python")]
#[pyfunction]
pub fn chebyshev_approximate_py(a_j: Vec<f64>, a: f64, b: f64, x: f64) -> PyResult<f64> {
    Ok(chebyshev_approximate(a_j.try_into()?, a, b, x))
}

#[cfg(feature = "python")]
#[pyfunction]
pub fn chebyshev_subdivide_py<'py>(_python: Python<'py>, f: &Bound<'py, PyAny>, a: f64, b: f64, epsilon: f64, n0: usize, n_max: usize, interval_limit: f64) -> PyResult<(Vec<(f64, f64)>, Vec<Vec<f64>>)> {
    let f = {|x| f.clone().call1((x,)).unwrap().extract().unwrap()};
    let (intervals, coefficients) = chebyshev_subdivide(&f, vec![(a, b)], n0, epsilon, n_max, interval_limit).map_err(|e| PyRuntimeError::new_err(format!("Chebyshev subdivide failed: {}", e)))?;
    return Ok((intervals, coefficients.iter().map(|v| v.iter().map(|x| *x).collect::<Vec<f64>>()).collect()))
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (f, a, b, epsilon=1e-6, n0=2, n_max=200, interval_limit=0.0, complex_threshold=0.0, truncation_threshold=0.0, far_from_zero=f64::MAX))]
pub fn find_roots_py<'py>(_python: Python<'py>, f: &Bound<'py, PyAny>, a: f64, b: f64, epsilon: f64, n0: usize, n_max: usize, interval_limit: f64, complex_threshold: f64, truncation_threshold: f64, far_from_zero: f64) -> PyResult<Vec<f64>> {
    let f = {|x| f.clone().call1((x,)).unwrap().extract().unwrap()};
    find_roots(&f, vec![(a, b)], n0, epsilon, n_max, complex_threshold, truncation_threshold, interval_limit, far_from_zero).map_err(|e| PyRuntimeError::new_err(format!("Chebyshev rootfinding failed: {}", e)))
}
