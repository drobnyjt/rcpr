pub use super::*;

use crate::chebyshev::{chebyshev_subdivide, chebyshev_frobenius_matrix, truncate_chebyshev_coefficients};
use crate::polish::*;

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
                continue
            }

            //Truncate chebyshev coefficients if below threshold
            let a_j = truncate_chebyshev_coefficients(c, truncation_threshold);

            //If len(a_j) is 1, then its eigenvalue is simply itself, and the interval can be skipped.
            if a_j.len() == 1 {
                roots.push(a_j[0]*(i.1 - i.0)/2. + (i.1 + i.0)/2.);
                continue
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

                    if (eigenvalue.re.abs() <= 1.) && (eigenvalue.im.abs() <= complex_threshold){
                        roots.push(eigenvalue.re*(i.1 - i.0)/2. + (i.1 + i.0)/2.)
                    }
                }
            } else {
                let subroots = find_roots(&f, vec![(i.0, i.0 + (i.1 - i.0)/2.), (i.0 + (i.1 - i.0)/2., i.1)], N0, epsilon, N_max, complex_threshold, truncation_threshold, interval_limit, far_from_zero)?;
                for root in subroots {
                    roots.push(root)
                }
            }
        }
        Ok(roots)
    } else {
        Err(anyhow!("Subdivision reached interval limit without converging. Consider relaxing epsilon or increasing N_max. a = {} b = {} F(a) = {} F(b) = {}", a, b, f(a), f(b)))
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

pub fn real_polynomial_roots(c_j: Vec<f64>, complex_threshold: f64) -> Result<Vec<f64>, anyhow::Error> {

    let mut B_jk = monomial_fiedler_matrix(c_j.into());

    balance_parlett_reinsch(&mut B_jk);

    let roots = B_jk.complex_eigenvalues();

    Ok(roots.iter().filter(|x| (x.im).abs() <= complex_threshold).map(|x| x.re).collect::<Vec<f64>>())
}