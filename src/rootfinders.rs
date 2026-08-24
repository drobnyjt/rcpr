pub use super::*;

use crate::chebyshev::{chebyshev_subdivide, chebyshev_frobenius_matrix, truncate_chebyshev_coefficients};
use crate::polish::*;
use serde::*;
use nalgebra::Schur;

const DEFAULT_EPSILON: f64 = 1e-12;
const DEFAULT_DELTA: f64 = 1e-13;
const SCHUR_DECOMPOSITION_EPSILON: f64 = 1e-16;
const SCHUR_DECOMPOSITION_MAX_ITERATIONS: usize = 128;

const fn default_epsilon() -> f64 {
    DEFAULT_EPSILON
}

const fn default_delta() -> f64 {
    DEFAULT_DELTA
}

const fn default_float_zero() -> f64 {
    0.0
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

#[derive(Clone, Copy, Deserialize)]
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
    #[serde(default = "default_float_zero")]
    pub interval_limit: f64,
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
        interval_limit: f64
    ) -> Config {
        Config {
            epsilon,
            delta,
            N0,
            N_max,
            complex_threshold,
            far_from_zero,
            interval_limit
        }
    }
}

pub fn find_roots<F: Fn(f64) -> f64>(f: &F, intervals: Vec<(f64, f64)>, config: Config) -> Result<Vec<f64>, anyhow::Error> {

    let Config { epsilon, N0, N_max, complex_threshold, far_from_zero, interval_limit, .. } = config;

    ensure!(N0 > 0, "N0 cannot be zero.");
    ensure!(N_max >= N0, "N_max cannot be smaller than N0.");
    ensure!(complex_threshold >= 0., "Complex threshold cannot be less than zero.");
    ensure!(interval_limit >= 0., "Interval limit cannot be less than zero.");
    ensure!(far_from_zero >= 0., "Far-from-zero threshold cannot be less than zero.");

    let a = intervals[0].0;
    let b = intervals[intervals.len() - 1].1;

    ensure!(b > a, "Invalid interval [{}, {}]", a, b);

    let (intervals, coefficients, evaluations) = chebyshev_subdivide(f, intervals, N0, epsilon, N_max, interval_limit)?;
    let mut roots: Vec<f64> = Vec::new();

    for (index, ((i, c), fxk)) in intervals.iter().zip(coefficients).zip(evaluations).enumerate() {

        if c.is_empty() {
            continue
        }

        //Test if all chebyshev interpolants in this interval are far from zero
        //If yes, skip this interval
        let min = fxk.max();
        let max = fxk.min();
        if min > far_from_zero || max < -far_from_zero {
            continue
        }

        //Truncate trailing chebyshev coefficients if below threshold
        let a_j = truncate_chebyshev_coefficients(c)?;

        //If len(a_j) is 1, then its eigenvalue is simply itself, and the interval can be skipped.
        if a_j.len() == 1 {
            // roots.push(a_j[0]*(i.1 - i.0)/2. + (i.1 + i.0)/2.);
            continue
        }

        let mut A = chebyshev_frobenius_matrix(a_j)?;

        //Parlett-Reinsch balancing conditions the values of the matrix to avoid floating point errors
        //https://doi.org/10.1007/BF02165404
        balance_parlett_reinsch(&mut A);

        // I mistakenly removed this because Issue #611 in nalgebra,
        // which requires this hack to prevent infinite loops, was resolved,
        // but the fix is not actually implemented in any release. Hence, back 
        // in it goes.
        if let Some(schur_matrix) = Schur::try_new(
            A,
            SCHUR_DECOMPOSITION_EPSILON,
            SCHUR_DECOMPOSITION_MAX_ITERATIONS
        ) {
            let eigenvalues = schur_matrix.complex_eigenvalues();
            for eigenvalue in eigenvalues.iter() {
                if (eigenvalue.im.abs() <= complex_threshold) && (eigenvalue.re.abs() < 1.0 + config.epsilon) {
                    if (index < intervals.len() - 1) && (1.0 - eigenvalue.re) <= f64::EPSILON {
                        // if not right-most interval, attempt to drop eigenvalues on right boundary
                        continue
                    }
                    roots.push(eigenvalue.re*(i.1 - i.0)/2. + (i.1 + i.0)/2.)
                }
            }
        } else {
            let subroots = find_roots(f, vec![(i.0, i.0 + (i.1 - i.0)/2.), (i.0 + (i.1 - i.0)/2., i.1)], config)?;
            for root in subroots {
                roots.push(root)
            }
        }
    }
    Ok(roots)
}

pub fn find_roots_piecewise_with_newton_polishing<F: Fn(f64) -> f64, G: Fn(f64) -> f64, D: Fn(f64) -> f64>(g: &G, f: &F, df: &D, intervals: Vec<(f64, f64)>, config: Config) -> Result<Vec<f64>, anyhow::Error> {

    let Config {delta, ..} = config;

    let a = intervals[0].0;
    let b = intervals[intervals.len() - 1].1;

    ensure!(b > a, "Invalid interval [{}, {}]", a, b);

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

pub fn find_roots_with_secant_polishing<F: Fn(f64) -> f64, G: Fn(f64) -> f64>(g: &G, f: &F, a: f64, b: f64, config: Config) -> Result<Vec<f64>, anyhow::Error> {

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

pub fn find_roots_with_newton_polishing<F: Fn(f64) -> f64, G: Fn(f64) -> f64, D: Fn(f64) -> f64>(g: &G, f: &F, df: &D, a: f64, b: f64, config: Config) -> Result<Vec<f64>, anyhow::Error> {

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

pub fn find_roots_piecewise_with_secant_polishing<F: Fn(f64) -> f64, G: Fn(f64) -> f64>(g: &G, f: &F, intervals: Vec<(f64, f64)>, config: Config) -> Result<Vec<f64>, anyhow::Error> {

    let Config {delta, ..} = config;

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

pub fn real_polynomial_roots(c_j: Vec<f64>, complex_threshold: f64) -> Result<Vec<f64>, anyhow::Error> {

    let mut B_jk = monomial_fiedler_matrix(c_j.into());

    balance_parlett_reinsch(&mut B_jk);

    let roots = B_jk.complex_eigenvalues();

    Ok(roots.iter().filter(|x| (x.im).abs() <= complex_threshold).map(|x| x.re).collect::<Vec<f64>>())
}