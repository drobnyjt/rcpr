pub use super::*;

use crate::chebyshev::*;//{chebyshev_subdivide, chebyshev_frobenius_matrix, truncate_chebyshev_coefficients, ErrorCalc, ChebError};
use crate::polish::*;
use serde::*;

const DEFAULT_EPSILON: f64 = 1e-6;
const DEFAULT_DELTA: f64 = 1e-9;

const fn default_epsilon() -> f64 {
    DEFAULT_EPSILON
}

const fn default_delta() -> f64 {
    DEFAULT_DELTA
}

const fn default_usize_2() -> usize {
    2
}

const fn default_usize_512() -> usize {
    512
}

const fn default_float_max() -> f64 {
    f64::MAX
}

const fn default_float_1_10000() -> f64 {
    1./10000.
}

const fn default_float_1_e_minus_12() -> f64 {
    1e-12
}

const fn default_error_calc() -> ErrorCalc {
    ErrorCalc::Absolute
}

/// Rootfinder configuration options
///
/// # Fields  
///  - `epsilon`: relative or absolute error of Chebyshev interpolation; `f64`  
///  - `delta`: hybrid error of polishers; `f64`  
///  - `N0`: initial degree of Chebyshev interpolation; `usize`  
///  - `N_max`: maximum degree of Chebyshev interpolation; `usize`  
///  - `complex_threshold`: magnitude of root complexity to ignore; `f64`  
///  - `far_from_zero`: skips intervals on which all Chebyshev coefficients > this value  
///  - `interval_limit`: limit on interval size after subdivision  
///  - `error_calc`: `ErrCalc::Absolute` or `ErrorCalc::Relative `   
#[derive(Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default = "default_epsilon")]
    pub epsilon: f64,
    #[serde(default = "default_delta")]
    pub delta: f64,
    #[serde(default = "default_usize_2")]
    pub N0: usize,
    #[serde(default = "default_usize_512")]
    pub N_max: usize,
    #[serde(default = "default_float_1_10000")]
    pub complex_threshold: f64,
    #[serde(default = "default_float_max")]
    pub far_from_zero: f64, 
    #[serde(default = "default_float_1_e_minus_12")]
    pub interval_limit: f64,
    #[serde(default = "default_error_calc")]
    pub error_calc: ErrorCalc
}

impl Default for Config {
    fn default() -> Config {
        Config {
            epsilon: default_epsilon(),
            delta: default_delta(),
            N0: default_usize_2(),
            N_max: default_usize_512(),
            complex_threshold: default_float_1_10000(),
            far_from_zero: default_float_max(),
            interval_limit: default_float_1_e_minus_12(),
            error_calc: default_error_calc(),
        }
    }
}

impl Config {
    pub fn new(
        epsilon: f64,
        delta: f64,
        N0: usize,
        N_max: usize,
        complex_threshold: f64,
        far_from_zero: f64,
        interval_limit: f64,
        error_calc: ErrorCalc
    ) -> Config {
        Config {
            epsilon,
            delta,
            N0,
            N_max,
            complex_threshold,
            far_from_zero,
            interval_limit,
            error_calc
        }
    }
}

/// Finds all roots of a function f(x) on the interval \[a, b\]
/// via adaptive Chebyshev proxy rootfinding with automatic subdivision.  
///
/// # Arguments  
///  - `f`: function to find roots of; must return `Result<f64, E>`  
///  - `intervals`: Vec of intervals [a_i, b_i], to, piecewise, find roots on; `Vec<(f64, f64)>`  
///  - `config`: `Config` struct that configures rootfinder.  
/// 
/// # Returns  
///  - `Result<roots, ChebError>`  
///  - `roots`: list of roots found, sorted. Roots are not deduplicated.  
/// 
/// # Sources  
/// Most complete, succinct description can be found in \[2\]. More discussion in \[1\].  
///  - \[1\] J Boyd, Solving Transcendental Equations, SIAM, 2014, doi: 10.1137/1.9781611973525  
///  - \[2\] J Boyd, Finding the Zeros of a Univariate Equation, SIAM Review, 2013, doi:10.1137/110838297  
pub fn find_roots<F, E>(f: &F, intervals: Vec<(f64, f64)>, config: Config) -> Result<Vec<f64>, ChebError> where F: Fn(f64) -> Result<f64, E>, E: std::error::Error + Send + Sync + 'static,  {

    let Config { epsilon, N0, N_max, complex_threshold, far_from_zero, interval_limit, error_calc, .. } = config;

    if N0==0 {
        return Err(ChebError::Input(InputProblem::InitialDegreeInvalid(N0)))
    }

    if N_max <= N0 {
        return Err(ChebError::Input(InputProblem::MaxDegreeInvalid(N_max)))
    }

    if complex_threshold < 0.0 {
        return Err(ChebError::Input(InputProblem::ComplexThresholdInvalid(complex_threshold)))
    }

    if interval_limit <= 0. {
        return Err(ChebError::Input(InputProblem::IntervalLimitInvalid(interval_limit)))
    }

    if far_from_zero <= 0. {
        return Err(ChebError::Input(InputProblem::FarFromZeroInvalid(far_from_zero)))
    }

    if intervals.is_empty() {
        return Err(ChebError::Input(InputProblem::EmptyIntervals))
    }
    let a = intervals[0].0;
    let b = intervals[intervals.len() - 1].1;

    if b <= a {
        return Err(ChebError::Input(InputProblem::IntervalInvalid((a, b))))
    }

    let (intervals, coefficients, evaluations) = chebyshev_subdivide(f, intervals, N0, epsilon, N_max, interval_limit, error_calc)?;
    let mut roots: Vec<f64> = Vec::new();

    for (index, ((i, c), fxk)) in intervals.iter().zip(coefficients).zip(evaluations).enumerate() {

        if c.is_empty() {
            continue
        }

        //Test if all chebyshev interpolants in this interval are far from zero
        //If yes, skip this interval
        let min = fxk.min();
        let max = fxk.max();
        if min > far_from_zero || max < -far_from_zero {
            continue
        }

        //Truncate trailing chebyshev coefficients below estimated threshold
        let a_j = truncate_chebyshev_coefficients(c)?;

        // If len(a_j) is 1, the function is a constant and the interval can be skipped.
        let N = a_j.len();
        if N == 1 {
            continue
        }

        // I'm deciding here to skip the interval if all chebyshev coefficients are zero
        // This may return an error later
        if a_j.iter().all(|&x| x==0.0) {
            continue
        }

        let mut A = chebyshev_frobenius_matrix(a_j)?;

        //Parlett-Reinsch balancing conditions the values of the matrix to avoid floating point errors
        //https://doi.org/10.1007/BF02165404
        balance_parlett_reinsch(&mut A);

        let eigenvalues = A.complex_eigenvalues();

        for eigenvalue in eigenvalues.iter() {
            if (eigenvalue.im.abs() <= complex_threshold) && (eigenvalue.re.abs() < 1.0 + config.epsilon) {
                if (index < intervals.len() - 1) && (1.0 - eigenvalue.re) <= f64::EPSILON {
                    // if not right-most interval, attempt to drop eigenvalues on right boundary
                    // this check needs to be more strict than above or it discards real roots
                    continue
                }
                roots.push(eigenvalue.re*(i.1 - i.0)/2. + (i.1 + i.0)/2.)
            }
        }
    }
    roots.sort_by(|a, b| a.total_cmp(b));
    Ok(roots)
}

/// Finds and Newton-polishes all roots of a function f(x) on intervals \[a_i, b_i\]
/// via adaptive Chebyshev proxy rootfinding with automatic subdivision  
///
/// # Arguments  
///  - `f`: function to find roots of; must return `Result<f64, E>`  
///  - `intervals`: Vec of intervals [a_i, b_i], to, piecewise, find roots on; `Vec<(f64, f64)>`  
///  - `config`: `Config` struct that configures rootfinder.  
/// 
/// # Returns  
///  - `Result<roots, ChebError>`  
///  - `roots`: list of roots found, sorted. Roots are not deduplicated.  
/// 
/// # Sources  
/// Most complete, succinct description can be found in \[2\]. More discussion in \[1\].  
///  - \[1\] J Boyd, Solving Transcendental Equations, SIAM, 2014, doi: 10.1137/1.9781611973525  
///  - \[2\] J Boyd, Finding the Zeros of a Univariate Equation, SIAM Review, 2013, doi:10.1137/110838297  
pub fn find_roots_piecewise_with_newton_polishing<F, G, D, E>(g: &G, f: &F, df: &D, intervals: Vec<(f64, f64)>, config: Config) -> Result<Vec<f64>, ChebError>
    where F: Fn(f64) -> Result<f64, E>, G: Fn(f64) -> Result<f64, E>, D: Fn(f64) -> Result<f64, E>, E: std::error::Error + Send + Sync + 'static,    {

    let Config {delta, ..} = config;

    let a = intervals[0].0;
    let b = intervals[intervals.len() - 1].1;

    if b <= a {
        return Err(ChebError::Input(InputProblem::IntervalInvalid((a, b))))
    }

    let roots = find_roots(g, intervals, config)?;
    let mut polished_roots: Vec<f64> = Vec::new();

    for root in roots.iter() {

        if let Ok(root_refined) = newton_polish(f, df, *root, NEWTON_MAX_ITERATIONS, delta){
            let correction = root_refined - *root;

            if ((correction/root_refined).abs() < 1.) && (root_refined >= a) && (root_refined <= b) {
                polished_roots.push(root_refined);
            }
        };
    }
    Ok(polished_roots)
}

/// Finds and Secant-polishes all roots of a function f(x) on interval \[a, b\]
/// via adaptive Chebyshev proxy rootfinding with automatic subdivision  
///
/// # Arguments  
///  - `f`: function to find roots of; must return `Result<f64, E>`  
///  - `intervals`: Vec of intervals [a_i, b_i], to, piecewise, find roots on; `Vec<(f64, f64)>`  
///  - `config`: `Config` struct that configures rootfinder.  
/// 
/// # Returns  
///  - `Result<roots, ChebError>`  
///  - `roots`: list of roots found, sorted. Roots are not deduplicated.  
/// 
/// # Sources  
/// Most complete, succinct description can be found in \[2\]. More discussion in \[1\].  
///  - \[1\] J Boyd, Solving Transcendental Equations, SIAM, 2014, doi: 10.1137/1.9781611973525  
///  - \[2\] J Boyd, Finding the Zeros of a Univariate Equation, SIAM Review, 2013, doi:10.1137/110838297
pub fn find_roots_with_secant_polishing<F, G, E>(g: &G, f: &F, a: f64, b: f64, config: Config) -> Result<Vec<f64>, ChebError> where F: Fn(f64) -> Result<f64, E>, G: Fn(f64) -> Result<f64, E>, E: std::error::Error + Send + Sync + 'static, {

    let Config {delta, ..} = config;

    let roots = find_roots(g, vec![(a, b)], config)?;
    let mut polished_roots: Vec<f64> = Vec::new();

    for root in roots.iter() {

        if let Ok(root_refined) = secant_polish(f, *root, SECANT_MAX_ITERATIONS, delta){
            let correction = root_refined - *root;

            if ((correction/root_refined).abs() < 1.) && (root_refined >= a) && (root_refined <= b) {
                polished_roots.push(root_refined);
            }
        };
    }
    Ok(polished_roots)
}

/// Finds and Newton-polishes all roots of a function f(x) on interval \[a, b\]
/// via adaptive Chebyshev proxy rootfinding with automatic subdivision  
///
/// # Arguments  
///  - `f`: function to find roots of; must return `Result<f64, E>`  
///  - `intervals`: Vec of intervals [a_i, b_i], to, piecewise, find roots on; `Vec<(f64, f64)>`  
///  - `config`: `Config` struct that configures rootfinder.  
/// 
/// # Returns
///  - `Result<roots, ChebError>`  
///  - `roots`: list of roots found, sorted. Roots are not deduplicated.  
/// 
/// # Sources  
/// Most complete, succinct description can be found in \[2\]. More discussion in \[1\].
///  - \[1\] J Boyd, Solving Transcendental Equations, SIAM, 2014, doi: 10.1137/1.9781611973525  
///  - \[2\] J Boyd, Finding the Zeros of a Univariate Equation, SIAM Review, 2013, doi:10.1137/110838297
pub fn find_roots_with_newton_polishing<F, G, D, E>(g: &G, f: &F, df: &D, a: f64, b: f64, config: Config) -> Result<Vec<f64>, ChebError>
    where F: Fn(f64) -> Result<f64, E>, G: Fn(f64) -> Result<f64, E>, D: Fn(f64) -> Result<f64, E>, E: std::error::Error + Send + Sync + 'static, {

    let Config {delta, ..} = config;

    let roots = find_roots(g, vec![(a, b)], config)?;
    let mut polished_roots: Vec<f64> = Vec::new();

    for root in roots.iter() {

        if let Ok(root_refined) = newton_polish(f, df, *root, NEWTON_MAX_ITERATIONS, delta){

            let correction = root_refined - *root;

            if ((correction/root_refined).abs() < 1.) && (root_refined >= a) && (root_refined <= b) {
                polished_roots.push(root_refined);
            }
        };
    }
    Ok(polished_roots)
}

/// Finds and Secant-polishes all roots of a function f(x) on intervals \[a_i, b_i\]
/// via adaptive Chebyshev proxy rootfinding with automatic subdivision  
///
/// # Arguments  
///  - `f`: function to find roots of; must return `Result<f64, E>`  
///  - `intervals`: Vec of intervals [a_i, b_i], to, piecewise, find roots on; `Vec<(f64, f64)>`  
///  - `config`: `Config` struct that configures rootfinder.  
/// 
/// # Returns  
///  - `Result<roots, ChebError>`  
///  - `roots`: list of roots found, sorted. Roots are not deduplicated.  
///   
/// # Sources  
/// Most complete, succinct description can be found in \[2\]. More discussion in \[1\].  
///  - \[1\] J Boyd, Solving Transcendental Equations, SIAM, 2014, doi: 10.1137/1.9781611973525  
///  - \[2\] J Boyd, Finding the Zeros of a Univariate Equation, SIAM Review, 2013, doi:10.1137/110838297  
pub fn find_roots_piecewise_with_secant_polishing<F, G, E>(g: &G, f: &F, intervals: Vec<(f64, f64)>, config: Config) -> Result<Vec<f64>, ChebError>
    where F: Fn(f64) -> Result<f64, E>, G: Fn(f64) -> Result<f64, E>, E: std::error::Error + Send + Sync + 'static, {

    let Config {delta, ..} = config;

    if intervals.is_empty() {
        return Err(ChebError::Input(InputProblem::EmptyIntervals));
    }

    let a = intervals[0].0;
    let b = intervals[intervals.len() - 1].1;

    let roots = find_roots(g, intervals, config)?;
    let mut polished_roots: Vec<f64> = Vec::new();
    for root in roots.iter() {

        if let Ok(root_refined) = secant_polish(f, *root, SECANT_MAX_ITERATIONS, delta){
            let correction = root_refined - *root;

            if ((correction/root_refined).abs() < 1.) && (root_refined >= a) && (root_refined <= b) {
                polished_roots.push(root_refined);
            }
        };
    }
    Ok(polished_roots)
}

/// Finds all roots of a polynomial via eigenvalues of the monomial Fiedler companion matrix
/// 
/// # Arguments  
///  - `c_j` list of coefficients in monomial basis; e.g., x^2 - 3.*x - 1.0 is \[1., -3, -1\]  
///
/// # Returns  
///  - `Result<roots, ChebError>`  
///  - `roots`: list of roots found; `Vec<f64>`  
pub fn real_polynomial_roots(c_j: Vec<f64>, complex_threshold: f64) -> Result<Vec<f64>, ChebError> {

    if c_j.len() < 3 {
        return Err(ChebError::Input(InputProblem::PolynomialDegreeInvalid))
    }

    let leading_coefficient = c_j[0];
    let scaled_c_j = c_j.iter().map(|&x| x/leading_coefficient).collect::<Vec<f64>>();

    let mut B_jk = monomial_fiedler_matrix(scaled_c_j.into())?;

    balance_parlett_reinsch(&mut B_jk);

    let mut roots: Vec<f64> = B_jk.complex_eigenvalues()
        .iter()
        .filter(|x| (x.im).abs() <= complex_threshold)
        .map(|x| x.re)
        .collect::<Vec<f64>>();

    roots.sort_by(|a, b| a.total_cmp(b));

    Ok(roots)
}