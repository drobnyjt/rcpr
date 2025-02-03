#![allow(non_snake_case)]

#[cfg(test)]
mod tests {
    use cached::proc_macro::cached;
    pub use crate::chebyshev::*;

    fn g(x: f64) -> f64 {
        f(x)/(10. + x.powf(6.))
    }

    //This is an adversarial function; it has 7 roots, 1 of which is on an end of the interval
    //and two which are both very near zero and very near each other.
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
        let epsilon = 1E-3;
        let truncation_threshold = 1E-9;
        let N_max = 10000;
        let complex_threshold = 1e-6;
        let interval_limit = 1E-12;
        let far_from_zero = 1E9;

        let roots = find_roots_piecewise_with_newton_polishing(&g, &f, &df, vec![(a, -2E-4), (-2E-4, 0.0), (0.0, 2E-5), (2E-5, b)], N0, epsilon, N_max, complex_threshold, truncation_threshold, interval_limit, far_from_zero).unwrap();
        let num_roots = roots.len();

        println!("Identified {} roots.", num_roots);
        for root in roots.iter() {
            println!("Root: {}", root);
        }
        println!("Sum of roots: {}; Calculated value: {}", roots.iter().sum::<f64>(), -4.00009);
        assert_eq!(7, num_roots, "Rootfinder should find 7 roots. It found {}", num_roots);
        assert!((roots.iter().sum::<f64>() - -4.00009).powf(2.) < 0.01, "Sum of all roots should be -4.00009. Rootfinder found {}", roots.iter().sum::<f64>());
    }

    #[test]
    fn test_rootfinding_with_secant() {
        let a = -10.;
        let b = 10.;
        let N0 = 2;
        let epsilon = 1E-3;
        let truncation_threshold = 1E-9;
        let N_max = 10000;
        let complex_threshold = 1e-6;
        let interval_limit = 1E-12;
        let far_from_zero = 1E9;

        let roots = find_roots_piecewise_with_secant_polishing(&g, &f, vec![(a, -2E-4), (-2E-4, 0.0), (0.0, 2E-5), (2E-5, b)], N0, epsilon, N_max, complex_threshold, truncation_threshold, interval_limit, far_from_zero).unwrap();
        let num_roots = roots.len();

        println!("Identified {} roots.", num_roots);
        for root in roots.iter() {
            println!("Root: {}", root);
        }
        println!("Sum of roots: {}; Calculated value: {}", roots.iter().sum::<f64>(), -4.00009);
        assert_eq!(7, num_roots, "Rootfinder should find 7 roots. It found {}", num_roots);
        assert!((roots.iter().sum::<f64>() - -4.00009).powf(2.) < 0.01, "Sum of all roots should be -4.00009. Rootfinder found {}", roots.iter().sum::<f64>());
    }

    #[test]
    fn test_polynom() {

        //let g = |x: f64| x.powf(4.) + 4.2*x.powf(3.) - 1.8*x.powf(2.) - 13.*x + 9.6;

        let c_j: Vec<f64> = vec![1., 5.2, 3.4, -9.6];

        let roots = real_polynomial_roots(c_j.clone(), 1E-20).unwrap();

        println!("Roots are: 1, -3, -3.2");

        for root in roots.iter() {
            println!("Found root: {}", root);
        }
    }

    fn evaluate_polynom(coefficients: &Vec<f64>, root: f64) -> f64 {
        let mut sum = 0.;

        for (i, c) in coefficients.iter().rev().enumerate() {
            sum += c*root.powi(i as i32);
        }
        sum
    }
}

pub mod chebyshev {

    const NEWTON_MAX_ITERATIONS: usize = 1000;
    const SECANT_MAX_ITERATIONS: usize = 1000;
    const SCHUR_DECOMPOSITION_MAX_ITERATIONS: usize = 1000;
    const SCHUR_DECOMPOSITION_EPSILON: f64 = 1e-15;

    use nalgebra::{DMatrix, DVector, Schur};
    use nalgebra::linalg::balancing::balance_parlett_reinsch;
    use std::f64::consts::PI;
    use cached::proc_macro::cached;
    use anyhow::{Result, Context, anyhow};

    pub fn real_polynomial_roots(c_j: Vec<f64>, complex_threshold: f64) -> Result<Vec<f64>, anyhow::Error> {

        let mut B_jk = monomial_fiedler_matrix(c_j.into());

        balance_parlett_reinsch(&mut B_jk);

        let roots = B_jk.complex_eigenvalues();

        Ok(roots.iter().filter(|x| (x.im).abs() <= complex_threshold).map(|x| x.re).collect::<Vec<f64>>())

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
    fn interpolation_matrix(N: usize) -> DMatrix<f64> {

        let mut I_jk: DMatrix<f64> = DMatrix::zeros(N + 1, N + 1);

        for j in 0..=N {
            for k in 0..=N {
                I_jk[(j, k)] = 2./p(j, N)/p(k, N)/N as f64*(j as f64*PI*k as f64/N as f64).cos();
            }
        }
        I_jk
    }

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

    pub fn chebyshev_frobenius_matrix(a_j: DVector<f64>) -> DMatrix<f64> {
        let N: usize = a_j.len() - 1;
        let mut A_jk: DMatrix<f64> = DMatrix::zeros(N, N);

        for k in 0..N {
            A_jk[(0, k)] = delta(1, k as i32);
            A_jk[(N - 1, k)] = (-1.)*(a_j[k]/2./a_j[N]) + (1./2.)*delta(k as i32, N as i32 - 2);
        }

        for k in 0..N {
            for j in 1..N - 1 {
                A_jk[(j, k)] = (delta(j as i32, k as i32 + 1) + delta(j as i32, k as i32 - 1))/2.;
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
            return Err(anyhow!("Secant iteration guess is NaN. Check preceding calculation."))
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
        Err(anyhow!("Secant failed to converge after {} iterations.", iter_max))
    }

    pub fn bisection_polish(f: &dyn Fn(f64) -> f64, a0: f64, b0: f64, iter_max: usize, epsilon: f64) -> Result<f64, anyhow::Error> {
        let mut a = a0;
        let mut b = b0;
        let c = (a + b)/2.;
        let fc = f(c);
        assert!(f(a)*f(b) < 0., "There is an even number of roots of f(x) on the interval [{}, {}]. Cannot use bisection.", a, b);
        assert!(a > b, "[{}, {}] is not a valid interval.", a, b);

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

        let alphas: Vec<f64> = (1..=N).map(|j| 1./(2_f64).powf((j as f64 - 1.)/2.)).collect();

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

    fn truncate_coefficients(a_j: DVector<f64>, epsilon: f64) -> DVector<f64> {

        for (index, &a) in a_j.iter().rev().enumerate() {
            if a.abs() > epsilon {

                let stop: usize = a_j.len() - index - 1;

                return DVector::from(
                    a_j.iter()
                    .enumerate()
                    .filter(|(i, _)| i <= &stop)
                    .map(|(_, &a)| a)
                    .collect::<Vec<f64>>()
                )
            }
        }
        a_j
    }

    pub fn find_roots(f: &dyn Fn(f64) -> f64, intervals: Vec<(f64, f64)>, N0: usize, epsilon: f64, N_max: usize, complex_threshold: f64, truncation_threshold: f64, interval_limit: f64, far_from_zero: f64) -> Result<Vec<f64>, anyhow::Error> {

        assert!(N0 > 0, "N0 cannot be zero.");
        assert!(N_max >= N0, "N_max cannot be smaller than N0.");

        assert!(complex_threshold >= 0., "Complex threshold cannot be less than zero.");
        assert!(truncation_threshold >= 0., "Truncation threshold cannot be less than zero.");
        assert!(interval_limit >= 0., "Interval limit cannot be less than zero.");
        assert!(far_from_zero >= 0., "Far-from-zero threshold cannot be less than zero.");

        let a = intervals[0].0;
        let b = intervals[intervals.len() - 1].1;

        if let Ok((intervals, coefficients)) = chebyshev_subdivide(&f, intervals, N0, epsilon, N_max, interval_limit) {
            let mut roots: Vec<f64> = Vec::new();

            for (i, c) in intervals.iter().zip(coefficients).filter(|(_, c)| !c.is_empty() ) {

                let xk = lobatto_grid(i.0, i.1, c.len() - 1);
                let fxk: Vec<f64> = xk.iter().map(|&x| f(x)).collect();

                //Test if all chebyshev interpolants in this interval are far from zero
                //If yes, skip this interval
                if fxk.iter().all(|fx| fx.abs() > far_from_zero) {
                    break
                }

                //Truncate chebyshev coefficients if below threshold
                let a_j = truncate_coefficients(c, truncation_threshold);

                //If len(a_j) is 1, then its eigenvalue is simply itself, and the interval can be skipped.
                if a_j.len() == 1 {
                    roots.push(a_j[0]*(i.1 - i.0)/2. + (i.1 + i.0)/2.);
                    break
                }

                let mut A = chebyshev_frobenius_matrix(a_j);

                //Parlett-Reinsch balancing conditions the values of the matrix to avoid floating point errors
                //https://doi.org/10.1007/BF02165404
                balance_parlett_reinsch(&mut A);

                //nalgebra eigenvalue solver can fail for certain matrices in the Schur decomposition step
                //See issue #611 (https://github.com/dimforge/nalgebra/issues/611)
                //So Schur decomposition is attempted, if it fails, the interval is split in two and rootfinding is attempted on the split interval
                if let Some(schur_matrix) = Schur::try_new(
                    A,
                    SCHUR_DECOMPOSITION_EPSILON,
                    SCHUR_DECOMPOSITION_MAX_ITERATIONS
                ) {

                    let eigenvalues = schur_matrix.complex_eigenvalues();
                    for eigenvalue in eigenvalues.iter() {

                        if (eigenvalue.re.abs() < 1.) && (eigenvalue.im.abs() < complex_threshold){
                            roots.push(eigenvalue.re*(i.1 - i.0)/2. + (i.1 + i.0)/2.)
                        }
                    }
                } else {
                    if let Ok(subroots) = find_roots(&f, vec![(i.0, (i.1 - i.0)/2.), ((i.1 - i.0)/2., i.1)], N0, epsilon, N_max, complex_threshold, truncation_threshold, interval_limit, far_from_zero) {
                        for root in subroots {
                            roots.push(root)
                        }
                    }
                }
            }
            Ok(roots)
        } else {
            Err(anyhow!("Subdivision reached interval limit without converging. Consider relaxing epsilon or increasing N_max. a = {} b = {} F(a) = {} F(b) = {}", a, b, f(a), f(b)))
        }
    }

    pub fn find_roots_with_newton_polishing(g: &dyn Fn(f64) -> f64, f: &dyn Fn(f64) -> f64, df: &dyn Fn(f64) -> f64, a: f64, b: f64, N0: usize, epsilon: f64, N_max: usize, complex_threshold: f64, truncation_threshold: f64, interval_limit: f64, far_from_zero: f64) -> Result<Vec<f64>, anyhow::Error> {

        if let Ok(roots) = find_roots(g, vec![(a, b)], N0, epsilon, N_max, complex_threshold, truncation_threshold, interval_limit, far_from_zero) {
            let mut polished_roots: Vec<f64> = Vec::new();

            for root in roots.iter() {

                if let Ok(root_refined) = newton_polish(&f, &df, *root, NEWTON_MAX_ITERATIONS, epsilon){

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

                if let Ok(root_refined) = secant_polish(&f, *root, SECANT_MAX_ITERATIONS, epsilon){
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

    pub fn find_roots_piecewise_with_newton_polishing(g: &dyn Fn(f64) -> f64, f: &dyn Fn(f64) -> f64, df: &dyn Fn(f64) -> f64, intervals: Vec<(f64, f64)>, N0: usize, epsilon: f64, N_max: usize, complex_threshold: f64, truncation_threshold: f64, interval_limit: f64, far_from_zero: f64) -> Result<Vec<f64>, anyhow::Error> {

        let a = intervals[0].0;
        let b = intervals[intervals.len() - 1].1;

        if let Ok(roots) = find_roots(g, intervals, N0, epsilon, N_max, complex_threshold, truncation_threshold, interval_limit, far_from_zero) {
            let mut polished_roots: Vec<f64> = Vec::new();

            for root in roots.iter() {

                if let Ok(root_refined) = newton_polish(&f, &df, *root, NEWTON_MAX_ITERATIONS, epsilon){
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

                if let Ok(root_refined) = secant_polish(&f, *root, SECANT_MAX_ITERATIONS, epsilon){
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

    fn chebyshev_coefficients(f: &dyn Fn(f64) -> f64, a: f64, b: f64, N: usize) -> DVector<f64> {
        //Given a function f and an interval [a, b], returns a vector of the Chebyshev interpolation
        //coefficients on that interval of order N.
        let xk = lobatto_grid(a, b, N);
        let I_jk = interpolation_matrix(N);
        let f_xk: DVector<f64> = DVector::<f64>::from(xk.iter().map(|&x| f(x)).collect::<Vec<f64>>());

        I_jk*f_xk
    }

    fn lobatto_grid(a: f64, b: f64, N: usize) -> Vec<f64> {
        //Returns a Lobatto Grid on the interval [a, b] of order N.
        (0..=N).map(|k| (b - a)/2.*(PI*k as f64/N as f64).cos() + (b + a)/2.).collect::<Vec<f64>>()
    }

    pub fn chebyshev_subdivide(f: &dyn Fn(f64) -> f64, intervals: Vec<(f64, f64)>, N0: usize, epsilon: f64, N_max: usize, interval_limit: f64) -> Result<(Vec<(f64, f64)>, Vec<DVector<f64>>), anyhow::Error> {
        //Adaptive Chebyshev Series interpolation with automatic subdivision.
        //
        //This function automatically divides the domain by halves into subintervals
        //such that the function F on each subinterval is well approximated (within
        //epsilon) by a Chebyshev series of degree N_max or less.

        //For each (sub)interval, the adaptive Chebyshev interpolation algorithm,
        //which uses degree-doubling, is used to find a Chebyshev series of degree
        //N0*2^(N_iterations) < N_max on the interval that is within epsilon of F.

        let mut coefficients: Vec<DVector<f64>> = Vec::new();
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

    pub fn chebyshev_approximate(a_j: DVector<f64>, a: f64, b: f64, x: f64) -> f64 {
        //
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

    fn chebyshev_adaptive(f: &dyn Fn(f64) -> f64, a: f64, b: f64, N0: usize, epsilon: f64, N_max: usize) -> (DVector<f64>, f64) {
        //Adaptive Chebyshev approximation of the function f on the interval [a, b], which starts from degree N0 and doubles
        //the degree each iteration until the error is less than epsilon, starting with order N0 returning the Chebyshev coefficients a if
        //convergence is reached before the degree exceeds N_max.
        //
        let mut a_0 = chebyshev_coefficients(f, a, b, N0);
        let mut N0 = N0;

        loop {

            let N1 = 2*N0;
            let a_1 = chebyshev_coefficients(f, a, b, N1);

            //Error is defined as sum(delta) where delta_2N = fN(x) - f2N(x)
            //Since the N0..2N0 terms of fN are zero, this sum can be split into two pieces
            let error = a_0.iter().enumerate().map(|(i, a)| (a - a_1[i]).abs()).sum::<f64>() + a_1.iter().enumerate().filter(|(i, _)| i >= &N0).map(|(_, a)| a.abs()).sum::<f64>();

            if (error < epsilon) || (2*N1 >= N_max/2) {
                return (a_1, error)
            }

            a_0 = a_1;
            N0 = N1;
        }
    }
}
