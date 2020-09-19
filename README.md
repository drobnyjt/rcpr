# rcpr
Rust Chebyshev Proxy Rootfinder: A robust global rootfinder using adaptive Chebyshev interpolation with automatic subdivision that accurately finds all roots of a smooth function F(x) on [a, b] using the Chebyshev-Frobenius companion matrix.

rcpr has a number of dependencies. Most notably, ndarray, ndarray-linalg (which requires LAPACK and BLAS). rcpr has not yet succesfully been compiled on Windows.

You can use rcpr as follows:

Include the line:
  rcpr = { git = "https://github.com/drobnyjt/rcpr", branch= "master" }
in your Cargo.toml.

Include the line:
use rcpr::chebyshev::*;
in your rust source file.

Many users will simply need only two functions: 

find_roots_with_newton_polishing(G, F, DF, a, b, N0, epsilon, N_max, complex_threshold, truncation threshold, interval_limit, far_from_zero)

* G: &dyn Fn(f64) -> f64: the approriately scaled function to find roots of.
* F: &dyn Fn(f64) -> f64: the original function.
* DF: &dyn Fn(f64) -> f64: the derivative of the original funciton w.r.t. the independent variable.
* a, b: f64, f64: the lower and upper bounds of the interval to find roots in
* N0: usize: the initial degree of Chebyshev polynomial to approximate G
* epsilon: f64: the absolute tolerance of the Chebyshev approximation
* N_max: usize: the maximum degree of Chebyshev polynomial before the interval is subdivided
* complex_threshold: f64: the threshold of the imaginary part of roots that are near-real that is tolerated
* truncation_threshold: f64: trailing-degree Chebyshev polynomials with coefficients below this value will be ignored as negligible
* interval_limit: f64: if the subdivision algorithm produces an interval below this length, the code will panic and abort
* far_from_zero: f64: if G evaluated at all the Lobatto grid points is further than this value from zero, that interval will be assumed to have no roots

real_polynomial_roots(c, complex_threshold)

* c: the coefficients of the polynomial in monomial form, starting with degree n, n-1, ... 1, 0
* complex_threshold: f64: the threshold of the imaginary part of roots that are near-real that is tolerated
