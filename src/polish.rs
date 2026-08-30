use crate::chebyshev::ChebError;
use crate::chebyshev::*;
use crate::hyberr;

/// Polishes roots of a function f(x) using Newton's method to specified error delta; returns error if iter_max exceeded
pub fn newton_polish<F, D, E>(f: &F, df: &D, x0: f64, iter_max: usize, delta: f64) -> Result<f64, ChebError>
where F: Fn(f64) -> Result<f64, E>, D: Fn(f64) -> Result<f64, E>, E: std::error::Error + Send + Sync + 'static, {

    if !x0.is_finite() {
        return Err(ChebError::Numeric(NumericProblem::NonFinite))
    }

    let mut x = x0;
    let mut err = 0.;

    for _ in 1..=iter_max {
        let df_x = df(x).map_err(|e| ChebError::Function(format!("Failed to calculate df(x) for x={}: {}", x, e)))?;
        let xn = x - f(x).map_err(|e| ChebError::Function(format!("Failed to calculate f(x) for x={}: {}", x, e)))?/df_x;
        if !xn.is_finite() || !(1./df_x).is_finite() {
            return Err(ChebError::Numeric(NumericProblem::NonFinite))
        }
        err = hyberr(xn, x);
        x = xn;
        if err < delta {
            return Ok(x);
        }
    }
    Err(ChebError::NotConverged(NotConvergedInfo {
        function_name: "newton_polish",
        previous_error: err,
        num_iterations: iter_max
    }))
}

/// Polishes roots of a function f(x) using secant method to specified error delta; returns error if iter_max exceeded
pub fn secant_polish<F, E>(f: &F, x0: f64, iter_max: usize, delta: f64) -> Result<f64, ChebError> where F: Fn(f64) -> Result<f64, E>, E: std::error::Error + Send + Sync + 'static, {

    if !x0.is_finite() {
        return Err(ChebError::Numeric(NumericProblem::NonFinite))
    }

    let mut x1 = x0;
    let dx = x0.abs().max(1.0)*f64::EPSILON.sqrt();
    let mut x2 = x1 + dx;

    if hyberr(x2, x1) < delta {
        return Ok(x1)
    }
    
    for _ in 1..=iter_max {

        let f2 = f(x2).map_err(|e| ChebError::Function(format!("Failed to calculate f(x) for x={}: {}", x2, e)))?;
        let f1 = f(x1).map_err(|e| ChebError::Function(format!("Failed to calculate f(x) for x={}: {}", x1, e)))?;
        let df = f2 - f1;

        let x3 = x2 - f2*(x2 - x1)/df;

        //let err = (x3 - x2)*(x3 - x2);
        // This error is the absolute residual instead of the difference.
        let err = hyberr(x3, x2);

        if err < delta {
            return Ok(x3)
        }
        x1 = x2;
        x2 = x3;
    }
    Err(ChebError::NotConverged(NotConvergedInfo {
        function_name: "secant_polish",
        previous_error: hyberr(x2, x1),
        num_iterations: iter_max
    }))
}