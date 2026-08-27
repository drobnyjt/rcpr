#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::exceptions::*;
#[cfg(feature = "python")]
use pythonize::*;

use crate::chebyshev::{chebyshev_adaptive, chebyshev_approximate, chebyshev_subdivide, ErrorCalc};
use crate::rootfinders::*;
use crate::polish::secant_polish;
use crate::lobatto_grid;

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

    #[pymodule_export]
    use super::secant_polish_py;

    #[pymodule_export]
    use super::lobatto_grid_py;
}

/// Calculates the Lobatto grid of degree N on [a, b].
/// 
/// # Arguments  
///  - `a`: left side of interval  
///  - `b`: right side of interval  
///  - `N`: degree  
///
/// # Returns  
///  - `x_k`: list of gridpoints x_k  
#[cfg(feature = "python")]
#[pyfunction]
pub fn lobatto_grid_py<'py>(_python: Python<'py>, a: f64, b: f64, N: usize) -> PyResult<Vec<f64>> {
    if (b > a) && (N > 0) {
        Ok(lobatto_grid(a, b, N))
    } else {
        Err(PyValueError::new_err(format!("Invalid input to Lobatto Grid.")))
    }
}

/// Performs adaptive Chebyshev interpolation of initial degree n0 and maximum degree n_max for f(x) on [a, b]
/// 
/// # Arguments  
///  - `f`: f(x); must return float  
///  - `a`: left side of interval  
///  - `b`: right side of interval  
///  - `epsilon`: relative error of interpolation, option, default `1e-6`
///  - `n0`: initial degree of Chebyshev interpolant, optional, default `2`
///  - `n_max`: maximum degree of Chebyshev interpolant, optional, default `1000`
///  - `relative_error`: whether to use relative error instead of absolute, optional, default `True`
///
/// # Returns  
///  - `(coefficients, error)`  
///  - `coefficients`: Chebyshev coefficients  
///  - `error`: estimated relative error of fit  
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (f, a, b, epsilon=1e-6, n0=2, n_max=1000, relative_error=true))]
pub fn chebyshev_coefficients_py<'py>(_python: Python<'py>, f: &Bound<'py, PyAny>, a: f64, b: f64, epsilon: f64, n0: usize, n_max: usize, relative_error: bool) -> PyResult<(Vec<f64>, f64)> {
    let f = |x| f.clone().call1((x,))?.extract();
    let error_calc = if relative_error { ErrorCalc::Relative } else { ErrorCalc::Absolute };
    let (result, error, _) = chebyshev_adaptive(&f, a, b, n0, epsilon, n_max, error_calc).map_err(|e| PyRuntimeError::new_err(format!("Secant polishing failed: {}", e)))?;
    Ok((result.iter().map(|x| *x).collect::<Vec<f64>>(), error))
}

/// Given a list of Chebyshev coefficients and a position x, returns the value of the approximated function at x through Curtis-Clenshaw recursion relation
///
/// # Arguments  
///  - `a_j`: list of Chebyshev coefficients  
///  - `a`: left side of interval  
///  - `b`: right side of interval  
///  - `x`: x to approximate f(x) at  
///
/// # Returns
///  - `f(x)`: approximated value of f(x)
#[cfg(feature = "python")]
#[pyfunction]
pub fn chebyshev_approximate_py(a_j: Vec<f64>, a: f64, b: f64, x: f64) -> PyResult<f64> {
    Ok(chebyshev_approximate(a_j.try_into()?, a, b, x))
}

/// Performs Chebyshev interpolation of initial degree n0 and maximum degree n_max with automatic subdivision for f(x) on \[a, b\]
/// 
/// # Arguments  
///  - `f`: f(x); must return float  
///  - `a`: left side of interval  
///  - `b`: right side of interval  
///  - `epsilon`: relative error of interpolation  
///  - `n0`: initial degree of Chebyshev interpolant  
///  - `n_max`: maximum degree of Chebyshev interpolant  
///  - `interval_limit`: minimum allowable interval
///  - `relative_error`: whether to use relative error, optional, default `True`
///
/// # Returns  
///  - `(intervals, coefficients)`  
///  - `coefficients`: Chebyshev coefficients  
///  - `error`: estimated relative error of fit  
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (f, a, b, epsilon=1e-6, n0=2, n_max=1000, interval_limit=1e-4, relative_error=true))]
pub fn chebyshev_subdivide_py<'py>(_python: Python<'py>, f: &Bound<'py, PyAny>, a: f64, b: f64, epsilon: f64, n0: usize, n_max: usize, interval_limit: f64, relative_error: bool) -> PyResult<(Vec<(f64, f64)>, Vec<Vec<f64>>)> {
    let f = |x| f.clone().call1((x,))?.extract();
    let error_calc = if relative_error { ErrorCalc::Relative } else { ErrorCalc::Absolute };
    let (intervals, coefficients, _) = chebyshev_subdivide(&f, vec![(a, b)], n0, epsilon, n_max, interval_limit, error_calc).map_err(|e| PyRuntimeError::new_err(format!("Chebyshev subdivide failed: {}", e)))?;
    return Ok((intervals, coefficients.iter().map(|v| v.iter().map(|x| *x).collect::<Vec<f64>>()).collect()))
}

/// Simultaneously finds multiple roots with adaptive Chebyshev proxy rootfinding with automatic subdivision for f(x) on \[a, b\]
/// 
/// # Arguments  
///  - `f`: f(x); must return float  
///  - `a`: left side of interval  
///  - `b`: right side of interval  
///  - `conifg`: options for rootfinder  
///
/// # Returns
///  - `roots`: list of Roots. roots are sorted but not deduplicated.  
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (f, a, b, config=None))]
pub fn find_roots_py<'py>(_python: Python<'py>, f: &Bound<'py, PyAny>, a: f64, b: f64, config: Option<&Bound<'py, PyAny>>) -> PyResult<Vec<f64>> {
    let config = match config {
        Some(x) => depythonize(x)?,
        None => Config::default()
    };
    let f = |x| f.clone().call1((x,))?.extract();
    find_roots(&f, vec![(a, b)], config).map_err(|e| PyRuntimeError::new_err(format!("Chebyshev rootfinding failed: {}", e)))
}

/// Polish a root for a function f(x) using the secant method to relative error `epsilon` not exceeding `iter_max` iterations
///
/// # Arguments  
///  - `f`: f(x); must return float  
///  - `x0`: initial guess  
///  - `epsilon`: hybrid error of secant polishing  
///  - `iter_max`: maximum iteration number  
///
/// # Returns  
///  - `root_polished`: polished root  
#[cfg(feature = "python")]
#[pyfunction]
pub fn secant_polish_py<'py>(_python: Python<'py>, f: &Bound<'py, PyAny>, x0: f64, epsilon: f64, iter_max: usize) -> PyResult<f64> {
    let f = |x| f.clone().call1((x,))?.extract();
    secant_polish(&f, x0, iter_max, epsilon).map_err(|e| PyRuntimeError::new_err(format!("Secant polishing failed: {}", e)))
}