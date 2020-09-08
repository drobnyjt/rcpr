#![allow(non_snake_case)]
#[macro_use(s)]
extern crate ndarray;
extern crate ndarray_linalg;
extern crate blas;
extern crate openblas_src;
extern crate time;

#[cfg(test)]
mod tests {

    use time::Instant;
    pub use crate::chebyshev::*;
    use ndarray_linalg::*;

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    fn g(x: f64) -> f64 {
        f(x)*(-x.abs()).exp()
    }

    fn f(x: f64) -> f64 {
        (x - 2.)*(x + 3.)*(x - 8.)*(x + 1E-4)*(x - 1E-5)*(x + 1.)*(x + 10.)*(x*x).sin()
    }

    fn df(x: f64) -> f64 {
        (6000000000.*x*x*x*x*x - 34999550000.*x*x*x*x - 92002520004.*x*x*x + 116993790021.*x*x + 108007020046.*x + 4859961.)/1000000000.
    }

    #[test]
    fn test_chebyshev() {
        let a = -10.;
        let b = 10.;
        let N0 = 2;
        let epsilon = 1E-9;
        let truncation_threshold = 1E-13;
        let N_max = 100;
        let complex_threshold = 1E-13;
        let interval_limit = 1E-9;

        let start = Instant::now();
        let roots = find_roots(&g, a, b, N0, epsilon, N_max, complex_threshold, truncation_threshold, interval_limit);
        let stop = Instant::now();

        let mut num_roots: usize = 0;
        for &root in roots.iter() {

            let root_refined = newton_polish(&f, &df, root, 100, epsilon);

            let correction = newton_correction(&f, &df, root_refined);

            println!("CPR root: {} Newton Correction: {}%", root, correction*100.);

            if (correction).abs() < 1E-3 {
                num_roots += 1;
                println!("Root Identified: {}", root_refined);
            }
        }

        println!("Chebyshev adaptive interpolation with subdivision took {} s", start.to(stop).as_seconds_f32());
        println!("Identified {} roots.", num_roots)
    }
}

pub mod roots {
    pub use crate::chebyshev;
}

pub mod chebyshev {
    use ndarray::{Array2, Array1};
    use std::f64::consts::PI;
    use ndarray_linalg::*;
    //use itertools::{zip_eq, chain};

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

    fn interpolation_matrix(N: usize) -> Array2<f64> {

        let mut I_jk: Array2<f64> = Array2::zeros((N + 1, N + 1));

        for j in 0..=N {
            for k in 0..=N {
                I_jk[[j, k]] = 2./p(j, N)/p(k, N)/N as f64*(j as f64*PI*k as f64/N as f64).cos();
            }
        }

        return I_jk
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

    pub fn newton_polish(f: &dyn Fn(f64) -> f64, df: &dyn Fn(f64) -> f64, x0: f64, iter_max: usize, epsilon: f64) -> f64 {

        let mut x = x0;
        for _ in 0..=iter_max {
            let x1 = newton_iteration(f, df, x);
            if (x1 - x)/x1 < epsilon {
                return x1
            }
            x = x1;
        }
        return x0
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
        return a_j
    }

    pub fn find_roots(f: &dyn Fn(f64) -> f64, a: f64, b: f64, N0: usize, epsilon: f64, N_max: usize, complex_threshold: f64, truncation_threshold: f64, interval_limit: f64) -> Vec<f64> {
        let mut roots: Vec<f64> = Vec::new();

        assert!(b > a);

        let (intervals, coefficients) = chebyshev_subdivide(&f, vec![(a, b)], N0, epsilon, N_max, interval_limit);

        for (i, c) in intervals.iter().zip(coefficients) {

            let a_j = truncate_coefficients(c, truncation_threshold);

            if a_j.len() <= 1 {
                break
            }

            let A = chebyshev_frobenius_matrix(a_j);

            if let Ok((eigenvalues, _)) = A.clone().eig() {
                for eigenvalue in eigenvalues.iter() {
                    if (eigenvalue.re.abs() < 1.) && (eigenvalue.im.abs() < complex_threshold){
                        roots.push(eigenvalue.re*(i.1 - i.0)/2. + (i.1 + i.0)/2.)
                    }
                }
            }
        }
        roots
    }

    pub fn find_roots_with_newton_polishing(g: &dyn Fn(f64) -> f64, f: &dyn Fn(f64) -> f64, df: &dyn Fn(f64) -> f64, a: f64, b: f64, N0: usize, epsilon: f64, N_max: usize, complex_threshold: f64, truncation_threshold: f64, interval_limit: f64) -> Vec<f64> {
        let roots = find_roots(g, a, b, N0, epsilon, N_max, complex_threshold, truncation_threshold, interval_limit);

        let mut polished_roots: Vec<f64> = Vec::new();

        for &root in roots.iter() {

            let root_refined = newton_polish(&f, &df, root, 100, epsilon);
            let correction = newton_correction(&f, &df, root_refined);

            if (correction) < 1E-3 {
                polished_roots.push(root);
            }
        }
        polished_roots
    }

    fn chebyshev_coefficients(f: &dyn Fn(f64) -> f64, a: f64, b: f64, N: usize) -> Array1<f64> {

        let xk = lobatto_grid(a, b, N);
        let I_jk = interpolation_matrix(N);
        let f_xk: Array1<f64> = xk.iter().map(|&x| f(x)).collect();

        I_jk.dot(&f_xk)
    }

    fn lobatto_grid(a: f64, b: f64, N: usize) -> Vec<f64> {

        let mut xk: Vec<f64> = vec![0.; N + 1];

        for k in 0..=N {
            xk[k] = (b - a)/2.*(PI*k as f64/N as f64).cos() + (b + a)/2.;
        }

        xk
    }

    pub fn chebyshev_subdivide(f: &dyn Fn(f64) -> f64, intervals: Vec<(f64, f64)>, N0: usize, epsilon: f64, N_max: usize, interval_limit: f64) -> (Vec<(f64, f64)>, Vec<Array1<f64>>) {
        let mut coefficients: Vec<Array1<f64>> = Vec::new();
        let mut intervals_out: Vec<(f64, f64)> = Vec::new();

        for interval in intervals {

            if (interval.1 - interval.0) < interval_limit {
                panic!("Reached minimum interval limit. Failed to converge. [a, b] = [{}, {}], f(a) = {}, f(b) = {}", interval.0, interval.1, f(interval.0), f(interval.1));
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

                let (intervals_new, coefficients_new) = chebyshev_subdivide(f, vec![(a1, b1), (a2, b2)], N0, epsilon, N_max, interval_limit);

                for (i, c) in intervals_new.iter().zip(coefficients_new) {
                    intervals_out.push(i.clone());
                    coefficients.push(c.clone());
                }
            }
        }
        return (intervals_out, coefficients)
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
