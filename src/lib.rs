#![allow(non_snake_case)]
const NEWTON_MAX_ITERATIONS: usize = 1000;
const SECANT_MAX_ITERATIONS: usize = 1000;

use nalgebra::{DMatrix, DVector};
use nalgebra::linalg::balancing::balance_parlett_reinsch;
use std::f64::consts::PI;

pub mod chebyshev;
pub mod rootfinders;
pub mod polish;
#[cfg(feature="python")]
pub mod python;

/// Constructs the monomial Frobenius companion matrix from monomial coefficients `c_j`
/// Assumes the leading coefficient is 1. and c_n is degree-0  
///
/// # Source  
/// A well known result; reproduced in \[3\] Eq. 1-2  
/// \[3\] M. Fiedler, A note on companion matrices, Lin. Alg. and its App., 2003, doi:10.1016/S0024-3795(03)00548-2  
pub fn monomial_frobenius_matrix(c_j: DVector<f64>) -> DMatrix<f64> {
    let N: usize = c_j.len() - 1;

    let mut A_jk: DMatrix<f64> = DMatrix::zeros(N, N);

    for k in 1..N {
        A_jk[(k, k - 1)] = 1.0;
    }

    for k in 0..N {
        A_jk[(k, N - 1)] = -c_j[N - k]
    }
    A_jk
}

/// Constructs the monomial Fiedler companion matrix from monomial coefficients `c_j`
/// Assumes the leading coefficient is 1. and c_n is degree-0  
///
/// # Source  
/// \[3\] Example 2.4 and surrounding discussion  
/// \[3\] M. Fiedler, A note on companion matrices, Lin. Alg. and its App., 2003, doi:10.1016/S0024-3795(03)00548-2  

fn monomial_fiedler_matrix(c_j: DVector<f64>) -> DMatrix<f64> {
    let N: usize = c_j.len() - 1;

    let mut A_jk: DMatrix<f64> = DMatrix::zeros(N, N);

    //Subdiagonals

    for k in (3..N).step_by(2) {
        A_jk[(k, k - 2)] = 1.0;
    }

    for k in (2..N).step_by(2) {
        A_jk[(k, k - 1)] = -c_j[k + 1];
    }

    //Superdiagonals

    for k in (0..N-2).step_by(2) {
        A_jk[(k, k + 2)] = 1.0;
    }

    for k in (0..N-1).step_by(2) {
        A_jk[(k, k + 1)] = -c_j[k + 2];
    }

    A_jk[(0, 0)] = -c_j[1];
    A_jk[(1, 0)] = 1.;

    A_jk
}

/// Calculates the values of a Lobatto grid on \[a, b\] of degree N
/// 
/// # Source
/// \[2\] A.1
/// \[2\] J Boyd, Finding the Zeros of a Univariate Equation, SIAM Review, 2013, doi:10.1137/110838297
pub fn lobatto_grid(a: f64, b: f64, N: usize) -> Vec<f64> {
    //Returns a Lobatto Grid on the interval [a, b] of order N.
    (0..=N).map(|k| (b - a)/2.*(PI*k as f64/N as f64).cos() + (b + a)/2.).collect::<Vec<f64>>()
}

#[cfg(test)]
mod tests;