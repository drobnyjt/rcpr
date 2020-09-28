#![allow(non_snake_case)]

#[macro_use(s)]
extern crate ndarray;
extern crate ndarray_linalg;

#[cfg(feature = "openblas")]
extern crate openblas_src;
#[cfg(feature = "netlib")]
extern crate netlib_src;
#[cfg(feature = "intel_mkl")]
extern crate intel_mkl_src;

#[cfg(test)]
mod tests {

    #[cfg(feature = "openblas")]
    extern crate openblas_src;
    #[cfg(feature = "netlib")]
    extern crate netlib_src;
    #[cfg(feature = "intel_mkl")]
    extern crate intel_mkl_src;

    use ndarray::*;
    use std::f64::consts::PI;
    use ndarray_linalg::*;
    use cached::proc_macro::cached;
    pub use crate::chebyshev::*;

    fn g(x: f64) -> f64 {
        f(x)/(1. + x.powf(6.))
    }

    fn f(x: f64) -> f64 {
        (x - 2.)*(x + 3.)*(x - 8.)*(x + 1E-4)*(x - 1E-5)*(x + 1.)*(x + 10.)
    }

    fn df(x: f64) -> f64 {
        (6000000000.*x*x*x*x*x - 34999550000.*x*x*x*x - 92002520004.*x*x*x + 116993790021.*x*x + 108007020046.*x + 4859961.)/1000000000.
    }

    #[test]
    fn test_rootfinding_with_newton() {
        let a = -10.;
        let b = 10.;
        let N0 = 2;
        let epsilon = 1E-9;
        let truncation_threshold = 1E-13;
        let N_max = 1000;
        let complex_threshold = 1E-13;
        let interval_limit = 1E-9;
        let far_from_zero = 1E2;

        let roots = find_roots_with_newton_polishing(&g, &f, &df, a, b, N0, epsilon, N_max, complex_threshold, truncation_threshold, interval_limit, far_from_zero).unwrap();
        let num_roots = roots.len();

        println!("Identified {} roots.", num_roots);
        for root in roots.iter() {
            println!("Root: {}", root);
        }
    }

    #[test]
    fn test_rootfinding_with_secant() {
        let a = -10.;
        let b = 10.;
        let N0 = 2;
        let epsilon = 1E-9;
        let truncation_threshold = 1E-13;
        let N_max = 1000;
        let complex_threshold = 1E-13;
        let interval_limit = 1E-9;
        let far_from_zero = 1E2;

        let roots = find_roots_piecewise_with_secant_polishing(&g, &f, vec![(a, b/2.), (b/2., b)], N0, epsilon, N_max, complex_threshold, truncation_threshold, interval_limit, far_from_zero).unwrap();
        let num_roots = roots.len();

        println!("Identified {} roots.", num_roots);
        for root in roots.iter() {
            println!("Root: {}", root);
        }
    }

    #[test]
    fn test_polynom() {
        ///(x - 1)^2*(x + 3)*(x + 3.2)
        let g = |x: f64| x.powf(4.) + 4.2*x.powf(3.) - 1.8*x.powf(2.) - 13.*x + 9.6;
        let c_j: Vec<f64> = vec![1., 5.2, 3.4, -9.6];//, -9.6, 3.4, 5.2, 1.];

        let roots = real_polynomial_roots(c_j, 1E-12).unwrap();

        println!("Roots are: 1, -3, -3.2");

        for root in roots.iter() {
            println!("Found root: {}", root);
        }
    }
}

pub mod chebyshev {
    use ndarray::{Array2, Array1};
    use std::f64::consts::PI;
    use ndarray_linalg::*;
    use cached::proc_macro::cached;
    use anyhow::*;

    pub fn real_polynomial_roots(c_j: Vec<f64>, complex_threshold: f64) -> Result<Vec<f64>, anyhow::Error> {

        let A_jk = monomial_frobenius_matrix(c_j.into());

        if let Ok((roots, _)) = A_jk.eig() {
            Ok(roots.iter().filter(|x| x.im <= complex_threshold).map(|x| x.re).collect::<Vec<f64>>())
        } else {
            Err(anyhow!("Eigenvalue calculation failed to find roots. Check coefficients."))
        }
    }

    fn p(j: usize, N: usize) -> f64 {
        if (j == 0) || (j == N) {
            2.
        } else {
            1.
        }
    }

    fn delta(j: i32, k: i32) -> f64 {
        if j == k {
            1.
        } else {
            0.
        }
    }

    #[cached]
    fn interpolation_matrix(N: usize) -> Array2<f64> {

        let mut I_jk: Array2<f64> = Array2::zeros((N + 1, N + 1));

        for j in 0..=N {
            for k in 0..=N {
                I_jk[[j, k]] = 2./p(j, N)/p(k, N)/N as f64*(j as f64*PI*k as f64/N as f64).cos();
            }
        }
        I_jk
    }

    pub fn monomial_frobenius_matrix(c_j: Array1<f64>) -> Array2<f64> {
        let N: usize = c_j.len() - 1;

        let mut A_jk: Array2<f64> = Array2::zeros((N, N));

        for k in 1..N {
            A_jk[[k, k - 1]] = 1.0;
        }

        for k in 0..N {
            A_jk[[k, N - 1]] = -c_j[N - k]
        }
        A_jk
    }

    pub fn chebyshev_frobenius_matrix(a_j: Array1<f64>) -> Array2<f64> {
        let N: usize = a_j.len() - 1;
        let mut A_jk: Array2<f64> = Array2::zeros((N, N));

        for k in 0..N {
            A_jk[[0, k]] = delta(1, k as i32);
            A_jk[[N - 1, k]] = (-1.)*(a_j[k]/2./a_j[N]) + (1./2.)*delta(k as i32, N as i32 - 2);
        }

        for k in 0..N {
            for j in 1..N - 1 {
                A_jk[[j, k]] = (delta(j as i32, k as i32 + 1) + delta(j as i32, k as i32 - 1))/2.;
            }
        }
        A_jk
    }

    pub fn newton_polish(f: &dyn Fn(f64) -> f64, df: &dyn Fn(f64) -> f64, x0: f64, iter_max: usize, epsilon: f64) -> Result<f64, anyhow::Error> {

        if x0.is_nan() {
            return Err(anyhow!("Newton iteration guess is NaN. Check preceding calculation."))
        }

        let mut x = x0;

        for _ in 1..=iter_max {
            let xn = x - f(x)/df(x);
            let err = (xn - x)*(xn - x);
            x = xn;
            if err.sqrt() < epsilon {
                return Ok(x);
            }
        }
        Err(anyhow!("Newton failed to converge after {} iterations.", iter_max))
    }

    pub fn secant_polish(f: &dyn Fn(f64) -> f64, x0: f64, iter_max: usize, epsilon: f64) -> Result<f64, anyhow::Error> {

        if x0.is_nan() {
            return Err(anyhow!("Newton iteration guess is NaN. Check preceding calculation."))
        }

        let mut x1 = x0;
        let mut x2 = x0*1.5;

        for _ in 1..=iter_max {

            let x3 = x2 - f(x2)*(x2 - x1)/(f(x2) - f(x1));

            let err = (x3 - x2)*(x3 - x2);

            if err.sqrt() < epsilon {
                return Ok(x3)
            }
            x1 = x2;
            x2 = x3;
        }
        Err(anyhow!("Newton failed to converge after {} iterations.", iter_max))
    }

    pub fn bisection_polish(f: &dyn Fn(f64) -> f64, a0: f64, b0: f64, iter_max: usize, epsilon: f64) -> Result<f64, anyhow::Error> {
        let mut a = a0;
        let mut b = b0;
        let c = (a + b)/2.;
        let fc = f(c);
        assert!(f(a)*f(b) < 0., "There is an even number of roots of f(x) on the interval [{}, {}]. Cannot use bisection.", a, b);
        assert!(a > b, "[{}, {}] is not a valid interval.");

        for _ in 1..iter_max {
            let c = (a + b)/2.;
            let fc = f(c);
            if fc.abs() < epsilon {
                return Ok(c)
            }
            if f(a)*fc < 0. {
                let b = c;
            } else if fc*f(b) < 0. {
                let a = c;
            } else {
                return Err(anyhow!("Bisection failed to find root in interval [{}, {}]", a, b))
            }
        }
        Err(anyhow!("Bisection failed to converge."))
    }

    pub fn newton_iteration(f: &dyn Fn(f64) -> f64, df: &dyn Fn(f64) -> f64, x0: f64) -> f64 {
        x0 - f(x0)/df(x0)
    }

    pub fn newton_correction(f: &dyn Fn(f64) -> f64, df: &dyn Fn(f64) -> f64, x0: f64) -> f64 {
        f(x0)/df(x0)
    }

    pub fn newton_armijo_iteration(f: &dyn Fn(f64) -> f64, df: &dyn Fn(f64) -> f64, alpha: f64, x0: f64) -> f64 {
        x0 - alpha*f(x0)/df(x0)
    }

    pub fn newton_armijo_line_search_iteration(f: &dyn Fn(f64) -> f64, df: &dyn Fn(f64) -> f64, N: usize, x0: f64) -> f64 {

        let alphas: Vec<f64> = (1..=N).map(|j| 1./(2.).powf((j as f64 - 1.)/2.)).collect();

        let x1: Vec<f64> = alphas.iter().map(|a| newton_armijo_iteration(f, df, *a, x0)).collect();

        let fx1: Vec<f64> = x1.iter().map(|x| f(*x)).collect();

        let mut index_min = 0;
        let mut fx_min = fx1[index_min];

        for (index, fx) in fx1.iter().enumerate() {
            if *fx < fx_min {
                index_min = index;
                fx_min = *fx;
            }
        }
        x1[index_min]
    }

    fn truncate_coefficients(a_j: Array1<f64>, epsilon: f64) -> Array1<f64> {

        for (index, &a) in a_j.to_vec().iter().rev().enumerate() {
            if a.abs() > epsilon {
                return a_j.slice(s![..a_j.len() - index]).to_owned()
            }
        }
        a_j
    }

    pub fn find_roots(f: &dyn Fn(f64) -> f64, intervals: Vec<(f64, f64)>, N0: usize, epsilon: f64, N_max: usize, complex_threshold: f64, truncation_threshold: f64, interval_limit: f64, far_from_zero: f64) -> Result<Vec<f64>, anyhow::Error> {

        assert!(N0 > 0, "N0 cannot be zero.");

        assert!(N_max >= N0, "N_max cannot be smaller than N0.");

        assert!(complex_threshold >= 0.);
        assert!(truncation_threshold >= 0.);
        assert!(interval_limit >= 0.);
        assert!(far_from_zero >= 0.);

        let a = intervals[0].0;
        let b = intervals[intervals.len() - 1].1;

        if let Ok((intervals, coefficients)) = chebyshev_subdivide(&f, intervals, N0, epsilon, N_max, interval_limit) {
            let mut roots: Vec<f64> = Vec::new();

            for (i, c) in intervals.iter().zip(coefficients).filter(|(_, c)| !c.is_empty() ) {

                let xk = lobatto_grid(i.0, i.1, c.len() - 1);
                let fxk: Vec<f64> = xk.iter().map(|&x| f(x)).collect();

                //Test if all chebyshev interpolants are far from zero
                if fxk.iter().all(|fx| fx.abs() > far_from_zero) {
                    break
                }

                //Truncare chebyshev coefficients if below threshold
                let a_j = truncate_coefficients(c, truncation_threshold);

                //If a_j is 1, then its eigenvalue is simply itself.
                if a_j.len() == 1 {
                    roots.push(a_j[0])
                }

                let A = chebyshev_frobenius_matrix(a_j);

                if let Ok((eigenvalues, _)) = A.clone().eig() {
                    for eigenvalue in eigenvalues.iter() {
                        if (eigenvalue.re.abs() < 1.) && (eigenvalue.im.abs() < complex_threshold){
                            roots.push(eigenvalue.re*(i.1 - i.0)/2. + (i.1 + i.0)/2.)
                        }
                    }
                } else {
                    return Err(anyhow!("Eigenvalue calculation failed. Consider scaling the function to prevent singularities and high dynamic range."))
                }
            }
            Ok(roots)
        } else {
            Err(anyhow!("Subdivision reached interval limit without converging. Consider relaxing epsilon or increasing N_max. F(a) = {} F(b) = {}", f(a), f(b)))
        }
    }

    pub fn find_roots_with_newton_polishing(g: &dyn Fn(f64) -> f64, f: &dyn Fn(f64) -> f64, df: &dyn Fn(f64) -> f64, a: f64, b: f64, N0: usize, epsilon: f64, N_max: usize, complex_threshold: f64, truncation_threshold: f64, interval_limit: f64, far_from_zero: f64) -> Result<Vec<f64>, anyhow::Error> {

        if let Ok(roots) = find_roots(g, vec![(a, b)], N0, epsilon, N_max, complex_threshold, truncation_threshold, interval_limit, far_from_zero) {
            let mut polished_roots: Vec<f64> = Vec::new();

            for root in roots.iter() {

                if let Ok(root_refined) = newton_polish(&f, &df, *root, 100, epsilon){
                    let correction = root_refined - *root;

                    if ((correction/root_refined).abs() < 1.) & (root_refined >= a) & (root_refined <= b) {
                        polished_roots.push(root_refined);
                    }
                };
            }
            Ok(polished_roots)
        } else {
            Err(anyhow!("Subdivision reached interval limit without converging. Consider relaxing epsilon or increasing N_max. F(a) = {} F(b) = {}", g(a), g(b)))
        }
    }

    pub fn find_roots_piecewise_with_secant_polishing(g: &dyn Fn(f64) -> f64, f: &dyn Fn(f64) -> f64, intervals: Vec<(f64, f64)>, N0: usize, epsilon: f64, N_max: usize, complex_threshold: f64, truncation_threshold: f64, interval_limit: f64, far_from_zero: f64) -> Result<Vec<f64>, anyhow::Error> {

        let a = intervals[0].0;
        let b = intervals[intervals.len() - 1].1;

        if let Ok(roots) = find_roots(g, intervals, N0, epsilon, N_max, complex_threshold, truncation_threshold, interval_limit, far_from_zero) {
            let mut polished_roots: Vec<f64> = Vec::new();

            for root in roots.iter() {

                if let Ok(root_refined) = secant_polish(&f, *root, 100, epsilon){
                    let correction = root_refined - *root;

                    if ((correction/root_refined).abs() < 1.) & (root_refined >= a) & (root_refined <= b) {
                        polished_roots.push(root_refined);
                    }
                };
            }
            Ok(polished_roots)
        } else {
            Err(anyhow!("Subdivision reached interval limit without converging. Consider relaxing epsilon or increasing N_max. F(a) = {} F(b) = {}", g(a), g(b)))
        }
    }

    pub fn find_roots_with_secant_polishing(g: &dyn Fn(f64) -> f64, f: &dyn Fn(f64) -> f64, a: f64, b: f64, N0: usize, epsilon: f64, N_max: usize, complex_threshold: f64, truncation_threshold: f64, interval_limit: f64, far_from_zero: f64) -> Result<Vec<f64>, anyhow::Error> {

        if let Ok(roots) = find_roots(g, vec![(a, b)], N0, epsilon, N_max, complex_threshold, truncation_threshold, interval_limit, far_from_zero) {
            let mut polished_roots: Vec<f64> = Vec::new();

            for root in roots.iter() {

                if let Ok(root_refined) = secant_polish(&f, *root, 100, epsilon){
                    let correction = root_refined - *root;

                    if ((correction/root_refined).abs() < 1.) & (root_refined >= a) & (root_refined <= b) {
                        polished_roots.push(root_refined);
                    }
                };
            }
            Ok(polished_roots)
        } else {
            Err(anyhow!("Subdivision reached interval limit without converging. Consider relaxing epsilon or increasing N_max. F(a) = {} F(b) = {}", g(a), g(b)))
        }
    }

    fn chebyshev_coefficients(f: &dyn Fn(f64) -> f64, a: f64, b: f64, N: usize) -> Array1<f64> {

        let xk = lobatto_grid(a, b, N);
        let I_jk = interpolation_matrix(N);
        let f_xk: Array1<f64> = xk.iter().map(|&x| f(x)).collect();

        I_jk.dot(&f_xk)
    }

    fn lobatto_grid(a: f64, b: f64, N: usize) -> Vec<f64> {
        (0..=N).map(|k| (b - a)/2.*(PI*k as f64/N as f64).cos() + (b + a)/2.).collect::<Vec<f64>>()
    }

    pub fn chebyshev_subdivide(f: &dyn Fn(f64) -> f64, intervals: Vec<(f64, f64)>, N0: usize, epsilon: f64, N_max: usize, interval_limit: f64) -> Result<(Vec<(f64, f64)>, Vec<Array1<f64>>), anyhow::Error> {
        let mut coefficients: Vec<Array1<f64>> = Vec::new();
        let mut intervals_out: Vec<(f64, f64)> = Vec::new();

        for interval in intervals {

            if (interval.1 - interval.0) < interval_limit {
                return Err(anyhow!("Reached minimum interval limit. Failed to converge. [a, b] = [{}, {}], f(a) = {}, f(b) = {}",
                    interval.0, interval.1, f(interval.0), f(interval.1)));
            }

            let a = interval.0;
            let b = interval.1;

            let (a_0, error) = chebyshev_adaptive(f, a, b, N0, epsilon, N_max);

            if error < epsilon {
                intervals_out.push(interval);
                coefficients.push(a_0);

            } else {
                let a1 = a;
                let b1 = a + (b - a)/2.;

                let a2 = a + (b - a)/2.;
                let b2 = b;

                let result = chebyshev_subdivide(f, vec![(a1, b1), (a2, b2)], N0, epsilon, N_max, interval_limit);
                if let Ok((intervals_new, coefficients_new)) = result {
                    for (i, c) in intervals_new.iter().zip(coefficients_new) {
                        intervals_out.push(*i);
                        coefficients.push(c.clone());
                    }
                } else {
                    return result
                };
            }
        }
        Ok((intervals_out, coefficients))
    }

    pub fn chebyshev_approximate(a_j: Array1<f64>, a: f64, b: f64, x: f64) -> f64 {

        let N = a_j.len() - 1;

        let xi = (2.*x - (b + a))/(b - a);
        let mut b0 = 0.;
        let mut b1 = 0.;
        let mut b2 = 0.;
        let mut b3 = 0.;

        for i in 1..=N {
            b0 = 2.*xi*b1 - b2 + a_j[N - i];
            b3 = b2;
            b2 = b1;
            b1 = b0;
        }

        (b0 - b3 + a_j[0])/2.
    }

    fn chebyshev_adaptive(f: &dyn Fn(f64) -> f64, a: f64, b: f64, N0: usize, epsilon: f64, N_max: usize) -> (Array1<f64>, f64) {

        let mut a_0 = chebyshev_coefficients(f, a, b, N0);
        let mut N0 = N0;

        loop {

            let N1 = 2*N0;
            let a_1 = chebyshev_coefficients(f, a, b, N1);

            //Error is defined as sum(delta) where delta_2N = fN(x) - f2N(x)
            //Since the N0..2N0 terms of fN are zero, this sum can be split into two pieces
            let error = a_0.iter().enumerate().map(|(i, a)| (a - a_1[i]).abs()).sum::<f64>() + a_1.slice(s![N0..]).iter().map(|a| a.abs()).sum::<f64>();

            if (error < epsilon) || (2*N1 >= N_max/2) {
                return (a_1, error)
            }

            a_0 = a_1;
            N0 = N1;
        }
    }
}
