[![Rust](https://github.com/drobnyjt/rcpr/actions/workflows/workflow.yml/badge.svg)](https://github.com/drobnyjt/rcpr/actions/workflows/workflow.yml)

# rcpr
Rust Chebyshev Proxy Rootfinder: A robust global rootfinder using adaptive Chebyshev interpolation with automatic subdivision that accurately finds all roots of a smooth function F(x) on [a, b] using the Chebyshev-Frobenius companion matrix. Also included is a polynomial rootfinder for polynoials in monomial form via the monomial companion matrix. This work is based on the text _Solving Transcendental Equations_ (2013) by John P Boyd.

You can use rcpr as follows:

Include the line:
  `rcpr = { git = "https://github.com/drobnyjt/rcpr", branch= "main" }`
in your Cargo.toml.

Include the line:
`use rcpr::rootfinders::*;`
in your rust source file.

Many users will simply need the config struct and one of three functions: `find_roots`, `find_roots_with_newton_polishing`, `real_polynomial_roots`, `find_roots_with_secant_polishing`. These three functions are documented below:

## Configuring rootfinder(s)

`rcpr` uses a Conig struct that has the following structure:

```
Config {
    epsilon,
    delta,
    N0,
    N_max,
    complex_threshold,
    far_from_zero,
    interval_limit
}
```
and can be instantiated with:
```
let config = Config::new(
    epsilon,
    delta,
    N0,
    N_max,
    complex_threshold,
    far_from_zero,
    interval_limit
) 
```
These parameters are:
* `N0: usize`: the initial degree of Chebyshev polynomial used to approximate G(x) (default: 2)
* `epsilon: f64`: the absolute tolerance of the Chebyshev approximation to G(x) (default: 1e-6)
* `delta: f64`: the (hybrid) relative error on step size of root polishers (default: 1e-9)
* `N_max: usize`: the maximum degree of Chebyshev polynomial before the interval is subdivided (default: 512) 
* `complex_threshold: f64`: the threshold of the imaginary part of roots that are near-real that is tolerated (default: 1e-4)
* `interval_limit: f64`: if the subdivision algorithm produces an interval below this length, the function will return an Error and abort (default: 1e-4)
* `far_from_zero: f64`: if G(x) evaluated at all the Lobatto grid points on an interval `[c, d]` is further than this value from zero, that interval will be assumed to have no roots contained within it (default: float_max)

## Rootfinders

### Find roots on intervals

`find_roots(F, intervals, config) -> Result<Vec<f64>, ChebError>`
* ` F: Fn(f64) -> Result<f64, E>`

Args:

* `F: Fn(f64) -> Result<f64, E>`: the original function to find roots of on the interval `[a, b]`. Must be real, single-valued, and continuous.
* `intervals: Vec<(f64, f64)>`: intervals to look for roots on; a list of tuples `(a_i, b_i)` such that `b_i < a_i`.
* `config`: Config struct defined above. 

Returns:
* `Result<Vec<f64>, ChebError>`: if successful, returns a vec of the real roots of F(x) on the union of all intervals. 

### Find roots with Newton polishing (dF/dx available)

`find_roots_with_newton_polishing(G, F, DF, a, b, config) -> Result<Vec<f64>, ChebError>`

Args:

* ` G: Fn(f64) -> Result<f64, E>`: the approriately scaled function G(x) = F(x)S(x), where F(x) is the orignial function and S(x) is a scaling function with no zeros on the interval [a,b], to find roots of. For polynomials of degree n on the interval [0, b] , a good general-purpose scaling function is `1/(1 + (r/a)^(m))` where a is an appropriate scaling factor to keep `r ~ O(1)` and `1 <= m <= n`. This is only important for functions with a very large range over a small domain; for well-behaved functions, S(x) can be 1.
* `F: Fn(f64) -> Result<f64, E>`: the original function to find roots of on the interval `[a, b]`
* `DF: Fn(f64) -> Result<f64, E>`: the derivative of the original funciton w.r.t. the independent variable.
* `a, b: f64, f64`: the lower and upper bounds of the interval to find roots of F(x) in.
* `config`: Config struct defined above. 

Returns:
* `Result<Vec<f64>, ChebError>`: if successful, returns a vec of the real roots of F(x) on the interval `[a, b]`, "polished" using Newton's method to hybrid relative stepsize delta.

### Find roots with secant polishing (dF/dx not available)

`find_roots_with_secant_polishing(G, F, a, b, N0, epsilon, N_max, config) -> Result<Vec<f64>, ChebError>`

Args:

* ` G: Fn(f64) -> Result<f64, E>`: the approriately scaled function G(x) = F(x)S(x), where F(x) is the orignial function and S(x) is a scaling function with no zeros on the interval [a,b], to find roots of. For polynomials of degree n on the interval [0, b] , a good general-purpose scaling function is `1/(1 + (r/a)^(m))` where a is an appropriate scaling factor to keep `r ~ O(1)` and `1 <= m <= n`. This is only important for functions with a very large range over a small domain; for well-behaved functions, S(x) can be 1.
* `F: Fn(f64) -> Result<f64, E>`: the original function to find roots of on the interval [a,b]
* `a, b: f64, Result<f64, E>`: the lower and upper bounds of the interval to find roots of F(x) in.
* `N0: usize`: the initial degree of Chebyshev polynomial used to approximate G(x)

Returns:
* `Result<Vec<f64>, ChebError>`: if successful, returns a vec of the real roots of F(x) on the interval `[a, b]`, "polished" using the Secant method to hybrid relative stepsize delta.

### Find roots of a polynomial in monomial basis

`real_polynomial_roots(c, complex_threshold) -> Result<Vec<f64>, ChebError>`

Args:

* `c: Vec<f64>` the coefficients of the polynomial in monomial form, with the first coefficient being 1, starting with degree n, n-1, ... 1, 0. For example, for the polynomial `P(x) = x^2 + 5x + 2`, `c = vec![1., 5., 2.]`
* `complex_threshold: f64`: the threshold of the imaginary part of roots that are near-real that is tolerated.

Returns:
* `Result<Vec<f64>, ChebError>`: if successful, returns a vec of the real roots of P(x)