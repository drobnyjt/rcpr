# rcpr
Rust Chebyshev Proxy Rootfinder: A robust global rootfinder using adaptive Chebyshev interpolation with automatic subdivision that accurately finds all roots of a smooth function F(x) on [a, b] using the Chebyshev-Frobenius companion matrix. Also included is a polynomial rootfinder for polynoials in monomial form via the monomial companion matrix. This work is based on the text _Solving Transcendental Equations_ (2013) by John P Boyd.

You can use rcpr as follows:

Include the line:
  `rcpr = { git = "https://github.com/drobnyjt/rcpr", branch= "master" }`
in your Cargo.toml.

Include the line:
`use rcpr::chebyshev::*;`
in your rust source file.

Many users will simply need one of three functions: `find_roots_with_newton_polishing`, `real_polynomial_roots`, `find_roots_with_secant_polishing`. These three functions are documented below:

`find_roots_with_newton_polishing(G, F, DF, a, b, N0, epsilon, N_max, complex_threshold, truncation threshold, interval_limit, far_from_zero) -> Result<Vec<f64>, anyhow::Error>`

Args:

* ` G: &dyn Fn(f64) -> f64`: the approriately scaled function G(x) = F(x)S(x), where F(x) is the orignial function and S(x) is a scaling function with no zeros on the interval [a,b], to find roots of. For polynomials of degree n on the interval [0, b] , a good general-purpose scaling function is `1/(1 + (r/a)^(m))` where a is an appropriate scaling factor to keep `r ~ O(1)` and `1 <= m <= n`. This is only important for functions with a very large range over a small domain; for well-behaved functions, S(x) can be 1.
* `F: &dyn Fn(f64) -> f64`: the original function to find roots of on the interval [a,b]
* `DF: &dyn Fn(f64) -> f64`: the derivative of the original funciton w.r.t. the independent variable.
* `a, b: f64, f64`: the lower and upper bounds of the interval to find roots of F(x) in.
* `N0: usize`: the initial degree of Chebyshev polynomial used to approximate G(x)
* `epsilon: f64`: the absolute tolerance of the Chebyshev approximation to G(x)
* `N_max: usize`: the maximum degree of Chebyshev polynomial before the interval is subdivided 
* `complex_threshold: f64`: the threshold of the imaginary part of roots that are near-real that is tolerated
* `truncation_threshold: f64`: trailing-degree Chebyshev polynomials with coefficients below this value will be ignored as negligible
* `interval_limit: f64`: if the subdivision algorithm produces an interval below this length, the function will return an Error and abort
* `far_from_zero: f64`: if G(x) evaluated at all the Lobatto grid points on an interval [c,d] is further than this value from zero, that interval will be assumed to have no roots contained within it

Returns:
* `Result<Vec<f64>, anyhow::Error>`: if successful, returns a vec of the real roots of F(x) on the interval [a,b], "polished" using Newton's method to absolute accuracy epsilon.

`real_polynomial_roots(c, complex_threshold) -> Result<Vec<f64>, anyhow::Error>`

Args:

* `c: Vec<f64>` the coefficients of the polynomial in monomial form, with the first coefficient being 1, starting with degree n, n-1, ... 1, 0. For example, for the polynomial `P(x) = x^2 + 5x + 2`, `c = vec![1., 5., 2.]`
* `complex_threshold: f64`: the threshold of the imaginary part of roots that are near-real that is tolerated

Returns:
* `Result<Vec<f64>, anyhow::Error>`: if successful, returns a vec of the real roots of P(x)


`find_roots_with_secant_polishing(G, F, a, b, N0, epsilon, N_max, complex_threshold, truncation threshold, interval_limit, far_from_zero) -> Result<Vec<f64>, anyhow::Error>`

Args:

* ` G: &dyn Fn(f64) -> f64`: the approriately scaled function G(x) = F(x)S(x), where F(x) is the orignial function and S(x) is a scaling function with no zeros on the interval [a,b], to find roots of. For polynomials of degree n on the interval [0, b] , a good general-purpose scaling function is `1/(1 + (r/a)^(m))` where a is an appropriate scaling factor to keep `r ~ O(1)` and `1 <= m <= n`. This is only important for functions with a very large range over a small domain; for well-behaved functions, S(x) can be 1.
* `F: &dyn Fn(f64) -> f64`: the original function to find roots of on the interval [a,b]
* `a, b: f64, f64`: the lower and upper bounds of the interval to find roots of F(x) in.
* `N0: usize`: the initial degree of Chebyshev polynomial used to approximate G(x)
* `epsilon: f64`: the absolute tolerance of the Chebyshev approximation to G(x)
* `N_max: usize`: the maximum degree of Chebyshev polynomial before the interval is subdivided 
* `complex_threshold: f64`: the threshold of the imaginary part of roots that are near-real that is tolerated
* `truncation_threshold: f64`: trailing-degree Chebyshev polynomials with coefficients below this value will be ignored as negligible
* `interval_limit: f64`: if the subdivision algorithm produces an interval below this length, the function will return an Error and abort
* `far_from_zero: f64`: if G(x) evaluated at all the Lobatto grid points on an interval [c,d] is further than this value from zero, that interval will be assumed to have no roots contained within it

Returns:
* `Result<Vec<f64>, anyhow::Error>`: if successful, returns a vec of the real roots of F(x) on the interval [a,b], "polished" using the Secant method to absolute accuracy epsilon.

