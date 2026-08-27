use super::*;
use cached::*;
use serde::Deserialize;
use thiserror::Error;

#[derive(Debug, Copy, Clone, Deserialize)]
pub enum ErrorCalc {
    Absolute,
    Relative
}

#[derive(Error, Debug)]
pub enum ChebError {
    #[error("Failed to converge: {0:?}")]
    NotConverged(NotConvergedInfo),
    #[error("Failed to evaluate function {0}")]
    Function(String),
    #[error("Numeric issue detected: {0:?}")]
    Numeric(NumericProblem),
    #[error("Invalid input: {0:?}")]
    Input(InputProblem),
}

#[derive(Debug)]
pub struct NotConvergedInfo {
    pub function_name: &'static str,
    pub previous_error: f64,
    pub num_iterations: usize,
}

#[derive(Debug)]
pub enum InputProblem {
    IntervalInvalid((f64, f64)),
    IntervalLimitInvalid(f64),
    InitialDegreeInvalid(usize),
    MaxDegreeInvalid(usize),
    ComplexThresholdInvalid(f64),
    EpsilonInvalid(f64),
    FarFromZeroInvalid(f64),
}

#[derive(Debug)]
pub enum NumericProblem {
    NonFinite,
    IntervalTooSmall((f64, f64)),
    IntervalLimitReached((f64, f64)),
    Comparison
}

/// Performs adaptive Chebyshev interpolation of a function f(x), on the interval x=\[a, b\].
///
/// This function calculates the Chebyshev coefficients and associated error (absolute or relative) 
/// of a single-valued, real, continuous function on the interval [a, b], for the smallest degree N
/// that satisfies error < epsilon. Each iteration, N0 is doubled until the desired error is achieved
/// or N_max is reached. 
///
/// # Arguments
///  - `f`: the function f(x); must return `Result<f64, E>`
///  - `a`: left side of interval; `f64` 
///  - `b`: right side of interval; `f64`
///  - `N0`: initial degree of Chebyshev approximant; usize
///  - `epsilon`: error tolerance; `f64`
///  - `N_max`: maximum degree of Chebyshev approximant; usize
///  - `error_calc`: ErrorCalc::Absolute or ErrorCalc::Relative
///
/// # Returns
///  - `Ok((a_1, error, f_1))`
///  - `a_1`: Chebyshev coefficients; `DVector<f64>`
///  - `error`: absolute or relative error; `f64`
///  - `f_1`: function evaluations on the Lobatto grid used in interpolation
///
/// # Sources
/// This algorithm combines aspects of the inteprolation described in \[1\] §3.2 and \[2\] Eq. 4.1-4.3.
/// It also offers the option to use the relative error instead of the absolute error;
/// in the author's experience, this improves convergence for f(x) with large dynamic range.
///  - \[1\] J Boyd, Solving Transcendental Equations, SIAM, 2014, doi: 10.1137/1.9781611973525
///  - \[2\] J Boyd, Finding the Zeros of a Univariate Equation, SIAM Review, 2013, doi:10.1137/110838297
pub fn chebyshev_adaptive<F, E>(
    f: &F, a: f64, b: f64, N0: usize, epsilon: f64, N_max: usize, error_calc: ErrorCalc
    ) -> Result<(DVector<f64>, f64, DVector<f64>), ChebError>
    where
        F: Fn(f64) -> Result<f64, E>,
        E: std::error::Error + Send + Sync + 'static, 
    {
    //Adaptive Chebyshev approximation of the function f on the interval [a, b], which starts from degree N0 and doubles
    //the degree each iteration until the error is less than epsilon, starting with order N0 returning the Chebyshev coefficients a if
    //convergence is reached before the degree exceeds N_max.
    let (mut a_0, mut f_0) = chebyshev_coefficients_fast(f, a, b, N0, DVector::<f64>::from(vec![]))?;
    let mut N0 = N0;

    loop {
        let N1 = 2*N0;
        let (a_1, f_1) = chebyshev_coefficients_fast(f, a, b, N1, f_0)?;

        //Error is defined as sum(delta) where delta_2N = fN(x) - f2N(x)
        //Since the N0..2N0 terms of fN are zero, this sum can be split into two pieces
        let absolute_error = a_0.iter().enumerate().map(|(i, a)| (a - a_1[i]).abs()).sum::<f64>() + a_1.iter().enumerate().filter(|(i, _)| i >= &(N0 + 1)).map(|(_, a)| a.abs()).sum::<f64>();
        let error = match error_calc {
            ErrorCalc::Absolute => absolute_error,
            ErrorCalc::Relative => {
                let norm = f_1.iter().map(|&x| x.abs()).max_by(f64::total_cmp).ok_or(ChebError::Numeric(NumericProblem::Comparison))?;
                if norm == 0.0 {
                    absolute_error
                } else {
                    absolute_error/norm
                }
            }
        };

        if (error < epsilon) || (2*N1 > N_max) {
            return Ok((a_1, error, f_1))
        };

        a_0 = a_1;
        f_0 = f_1;
        N0 = N1;
    }
}

/// Approximates a function f(x) on an interval \[a, b\] from its Chebyshev coefficients.
/// 
/// # Arguments
///  - `a_j`: Chebyshev coefficients; `DVector<f64>`
///  - `a`: left side of interval; `f64`
///  - `b`: right side of interval; `f64`
///  - `x`: the position at which to approximate f(x); `f64`
///
/// # Returns
///  - `y`: approximate value of f(x); `f64`
///
/// # Sources
///
///  - \[1\] §B.2.1 Eqs. B.9-B.13 and Table B.2
///  - \[1\] J Boyd, Solving Transcendental Equations, SIAM, 2014, doi: 10.1137/1.9781611973525
pub fn chebyshev_approximate(a_j: DVector<f64>, a: f64, b: f64, x: f64) -> f64 {
    let N = a_j.len() - 1;

    let xi = (2.0 * x - (b + a)) / (b - a);
    let mut b0 = 0.0;
    let mut b1 = 0.0;
    let mut b2 = 0.0;
    let mut b3 = 0.0;

    // N+1 iterations, consuming a_N, a_{N-1}, ..., a_0
    for i in 1..=N + 1 {
        b0 = 2.0 * xi * b1 - b2 + a_j[N + 1 - i];
        b3 = b2;
        b2 = b1;
        b1 = b0;
    }

    (b0 - b3 + a_j[0]) / 2.0
}

/// Performs adaptive Chebyshev interpolation for f(x) on x=\[a, b\] with automatic subdivision.
///
/// # Arguments
///  - `f`: function to interpolate; must return `Result<f64, E>.`
///  - `intervals`: list of intervals on which to interpolate; `Vec<(f64, f64)>`
///  - `N0`: initial degree of Chebyshev approximant on each interval
///  - `epsilon`: error tolerance
///  - `interval_limit`: limit on interval width
///  - `error_calc`: `ErrorCalc::Absolute` or `ErrorCalc::Relative`
/// 
/// # Returns
///  - `Result<(intervals, coefficients, evaluations), ChebError>`
///  - `intervals`: intervals \[a, b\] found by automatic subdivision; `Vec<(f64, f64)>`
///  - `coefficients`: Chebyshev coefficients on each interval in `intervals`; `DVector<f64>`
///  - `evaluations`: Evaluations of f(x) on the Lobatto grid for each interval - saved for reuse
/// 
/// # Errors
/// Will return `ChebError::Numeric` if subdivided interval is smaller than specified
///
/// # Sources
/// Strategy described in \[1\] §2.9 and \[2\]
/// limit or smaller than machine precision.
///  - \[1\] J Boyd, Solving Transcendental Equations, SIAM, 2014, doi: 10.1137/1.9781611973525
///  - \[2\] J Boyd, Finding the Zeros of a Univariate Equation, SIAM Review, 2013, doi:10.1137/110838297
pub fn chebyshev_subdivide<F, E>(
    f: &F, intervals: Vec<(f64, f64)>, N0: usize, epsilon: f64, N_max: usize, interval_limit: f64, error_calc: ErrorCalc
    ) -> Result<(Vec<(f64, f64)>, Vec<DVector<f64>>, Vec<DVector<f64>>), ChebError> 
    where 
        F: Fn(f64) -> Result<f64, E>,
        E: std::error::Error + Send + Sync + 'static,
    {

    let mut coefficients: Vec<DVector<f64>> = Vec::new();
    let mut intervals_out: Vec<(f64, f64)> = Vec::new();
    let mut evaluations: Vec<DVector<f64>> = Vec::new();

    for interval in intervals {

        if (interval.1 - interval.0) < interval_limit {
            return Err(ChebError::Numeric(NumericProblem::IntervalLimitReached((interval.0, interval.1))))
        }

        let a = interval.0;
        let b = interval.1;

        let (a_0, error, f_0) = chebyshev_adaptive(f, a, b, N0, epsilon, N_max, error_calc)?;
        
        if error < epsilon {
            intervals_out.push(interval);
            coefficients.push(a_0);
            evaluations.push(f_0);

        } else {

            let mid = a + (b - a)/2.;
            if !(a < mid && mid < b) {
                return Err(ChebError::Numeric(NumericProblem::IntervalTooSmall((a, b))))
            }

            let result = chebyshev_subdivide(f, vec![(a, mid), (mid, b)], N0, epsilon, N_max, interval_limit, error_calc);

            if let Ok((intervals_new, coefficients_new, evlautions_new)) = result {
                for ((i, c), evaluation) in intervals_new.iter().zip(coefficients_new).zip(evlautions_new) {
                    intervals_out.push(*i);
                    coefficients.push(c.clone());
                    evaluations.push(evaluation);
                }
            } else {
                return result
            };
        }
    }
    Ok((intervals_out, coefficients, evaluations))
}

/// Calculates the Chebyshev coefficients of a function f(x) on the interval \[a, b\]
fn chebyshev_coefficients_fast<F, E>(f: &F, a: f64, b: f64, N: usize, previous: DVector<f64>) -> Result<(DVector<f64>, DVector<f64>), ChebError>
where F: Fn(f64) -> Result<f64, E>, E: std::error::Error + Send + Sync + 'static, {
    //Given a function f and an interval [a, b], returns a vector of the Chebyshev interpolation
    //coefficients on that interval of order N.
    let xk = lobatto_grid(a, b, N);
    let I_jk = interpolation_matrix(N);

    if previous.is_empty() {

        let f_xk = DVector::<f64>::from(xk.iter().map(|&x| f(x)).collect::<Result<Vec<f64>, E>>().map_err(|e| ChebError::Function(format!("Failed to calculate f(x): {}", e)))?);

        return Ok((I_jk*f_xk.clone(), f_xk))
    }

    let f_xk = DVector::<f64>::from(
        xk.into_iter()
        .enumerate()
        .map(|(i, x_i)| if i%2==0 {
            Ok(previous[i/2])
        } else {
            f(x_i)
        }
    ).collect::<Result<Vec<f64>, E>>().map_err(|e| ChebError::Function(format!("Failed to calculate f(x): {}", e)))?);
    
    Ok((I_jk*f_xk.clone(), f_xk))
}

/// Given a vector of Chebyshev coefficients [a_0, ... a_N], return [a_0 ... a_i] such that a_i is the
/// first element > estimated truncation error.
pub fn truncate_chebyshev_coefficients(a_j: DVector<f64>) -> Result<DVector<f64>, ChebError> {

    let truncation_error = (a_j.len() - 1) as f64 * f64::EPSILON * a_j.iter()
        .map(|x| x.abs())
        .max_by(f64::total_cmp)
        .ok_or(ChebError::Numeric(NumericProblem::Comparison))?;

    for (index, &a) in a_j.iter().rev().enumerate() {
        if a.abs() > truncation_error {

            let stop: usize = a_j.len() - index - 1;

            return Ok(DVector::from(
                a_j.iter()
                .enumerate()
                .filter(|(i, _)| i <= &stop)
                .map(|(_, &a)| a)
                .collect::<Vec<f64>>())
            )
        }
    }
    Ok(a_j)
}

/// Constructs the Chebyshev-Frobenius Companion Matrix from Chebyshev coefficients a_j.
/// \[2\] B.2-3
///  - \[1\] J Boyd, Solving Transcendental Equations, SIAM, 2014, doi: 10.1137/1.9781611973525
///  - \[2\] J Boyd, Finding the Zeros of a Univariate Equation, SIAM Review, 2013, doi:10.1137/110838297
pub fn chebyshev_frobenius_matrix(a_j: DVector<f64>) -> Result<DMatrix<f64>, ChebError> {

    let N: usize = a_j.len() - 1;

    // For trivial coeffs, need to not divide only element by 2
    if N == 1 {
        let element = -a_j[0]/a_j[1];
        if element.is_finite() {
            return Ok(DMatrix::from_vec(1, 1, vec![-a_j[0]/a_j[1]]))
        } else {
            return Err(ChebError::Numeric(NumericProblem::NonFinite))
        }
    }

    let mut A_jk: DMatrix<f64> = DMatrix::zeros(N, N);

    let inv_2_aj_N = 1./2./a_j[N];
    for k in 0..N {
        A_jk[(0, k)] = delta(1, k as i32);
        A_jk[(N - 1, k)] = -(a_j[k]*inv_2_aj_N) + (1./2.)*delta(k as i32, N as i32 - 2);
    }

    for j in 1..N - 1 {
        A_jk[(j, j - 1)] = 0.5;
        A_jk[(j, j + 1)] = 0.5;
    }

    if A_jk.iter().any(|x| !x.is_finite()) {
        Err(ChebError::Numeric(NumericProblem::NonFinite))
    } else {
        Ok(A_jk)
    }    
}

/// Helper function used in construction of matrices.
fn p(j: usize, N: usize) -> f64 {
    if (j == 0) || (j == N) {
        2.
    } else {
        1.
    }
}

/// Kronecker delta; one if j == k, else 0.
fn delta(j: i32, k: i32) -> f64 {
    if j == k {
        1.
    } else {
        0.
    }
}

#[concurrent_cached]
/// Chebyshev interpolation matrix of size (N + 1).
/// \[2\] A.3
///  - \ [1\] J Boyd, Solving Transcendental Equations, SIAM, 2014, doi: 10.1137/1.9781611973525
///  - \[2\] J Boyd, Finding the Zeros of a Univariate Equation, SIAM Review, 2013, doi:10.1137/110838297
fn interpolation_matrix(N: usize) -> DMatrix<f64> {

    let mut I_jk: DMatrix<f64> = DMatrix::zeros(N + 1, N + 1);

    for j in 0..=N {
        for k in 0..=N {
            I_jk[(j, k)] = 2./p(j, N)/p(k, N)/N as f64*(j as f64*PI*k as f64/N as f64).cos();
        }
    }
    I_jk
}