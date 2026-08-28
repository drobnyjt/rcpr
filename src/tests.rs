pub use crate::rootfinders::*;
pub use crate::polish::*;
pub use crate::chebyshev::*;

fn g(x: f64) -> Result<f64, std::convert::Infallible> {
    Ok(f(x).unwrap()/(10. + x.powf(6.)))
}

//This is an adversarial function; it has 7 roots, 1 of which is on an end of the interval
//and two which are both very near zero and very near each other.
fn f(x: f64) -> Result<f64, std::convert::Infallible> {
    Ok((x - 2.)*(x + 3.)*(x - 8.)*(x + 1E-4)*(x - 1E-5)*(x + 1.)*(x + 10.0))
}

fn df(x: f64) -> Result<f64, std::convert::Infallible> {
    // From Wolfram Alpha
    Ok(7.*(0.00617137 + 137.153*x + 166.279*x.powi(2) - 100.576*x.powi(3) - 57.8569*x.powi(4) + 3.42865*x.powi(5) + x.powi(6)))
}

fn h(x: f64) -> Result<f64, std::convert::Infallible> {
    Ok((x - 0.5)*(x - 1.0)*(x + 1.0)*(x +  0.5))
}

fn q(x: f64) -> Result<f64, std::convert::Infallible> {
    Ok((x.powf(x) - x.powi(2))*(x - 3.).powi(3))
}

fn failing(_: f64) -> Result<f64, ChebError> {
    Err(ChebError::Function(format!("Failing function.")))
}

#[test]
fn test_newton_polish() {
    let f = |x: f64| -> Result<f64, std::convert::Infallible> { Ok(4.*(x - 3.)*(x - 2.)) };
    let df = |x: f64| -> Result<f64, std::convert::Infallible> { Ok(8.*x - 20.0) };

    let x0 = 3.1;
    let result = newton_polish(&f, &df, x0, 1000, 1e-12);
    assert!(result.is_ok());
    assert!((result.unwrap() - 3.0) < 1e-12);
    assert!(newton_polish(&failing, &failing, x0, 1000, 1e-12).is_err());
}

#[test]
fn test_secant_polish() {
    let f = |x: f64| -> Result<f64, std::convert::Infallible> { Ok(4.*(x - 3.)*(x - 2.)) };

    let x0 = 3.1;
    let result = secant_polish(&f, x0, 1000, 1e-12);

    // Ensure doesn't fail and is correct
    assert!(result.is_ok());
    assert!((result.unwrap() - 3.0) < 1e-12);
    // Ensure failing f(x) propagates
    assert!(secant_polish(&failing, x0, 1000, 1e-12).is_err());
    // Ensure bad iter_max errors
    assert!(secant_polish(&f, x0, 0, 1e-12).is_err());
    // Ensure bad delta errors
    assert!(secant_polish(&f, x0, 1000, -1.0).is_err());

}

#[test]
fn test_frobenius_matrix_inputs() {
    // empty coefficients can't be used to make a matrix
    let a = DVector::from(vec![]);
    assert!(chebyshev_frobenius_matrix(a).is_err());

    // constant functions don't have a companion matrix - no roots
    let b = DVector::from(vec![1.0]);
    assert!(chebyshev_frobenius_matrix(b).is_err());

    // trailing zeros should cause divide by zero error
    let c = DVector::from(vec![1.0, 2.0, 0.0]);
    assert!(chebyshev_frobenius_matrix(c).is_err());

    // minimum Ok example is two nonzero elements
    let d = DVector::from(vec![1.0, 2.0]);
    assert!(chebyshev_frobenius_matrix(d).is_ok());

    let e = DVector::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    assert!(chebyshev_frobenius_matrix(e).is_ok());
}

#[test]
fn test_monomial_frobenius_matrix_inputs() {
    // empty coefficients can't be used to make a matrix
    let a = DVector::from(vec![]);
    assert!(monomial_frobenius_matrix(a).is_err());

    // constant functions don't have a companion matrix - no roots
    let b = DVector::from(vec![1.0]);
    assert!(monomial_frobenius_matrix(b).is_err());

    // minimum Ok example is two nonzero elements
    let d = DVector::from(vec![1.0, 2.0]);
    assert!(monomial_frobenius_matrix(d).is_ok());

    let e = DVector::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    assert!(monomial_frobenius_matrix(e).is_ok());
}

#[test]
fn test_approximate() {
    let a = -5.0;
    let b = 5.0;
    let f = |x: f64| -> Result<f64, std::convert::Infallible> { Ok(x.powi(4) + 4.2*x.powi(3) - 1.8*x.powi(2) - 13.*x + 9.6) };
    let N0 = 2;
    let epsilon = 1e-6;
    let N_max = 512;
    let error_calc = ErrorCalc::Absolute;
    let (a_1, error, _) = chebyshev_adaptive(&f, a, b, N0, epsilon, N_max, error_calc).unwrap();    

    for x_test in vec![-5.0, -4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0, 5.0].into_iter() {
        assert!((chebyshev_approximate(a_1.clone(), a, b, x_test) - f(x_test).unwrap() ).abs() < epsilon);
    }
    assert!(error < epsilon);
}

#[test]
fn test_chebyshev_adaptive_and_term_truncation() {
    let a = -5.0;
    let b = 5.0;
    let f = |x: f64| -> Result<f64, std::convert::Infallible> { Ok(x.powi(4) + 4.2*x.powi(3) - 1.8*x.powi(2) - 13.*x + 9.6) };
    let N0 = 2;
    let epsilon = 1e-6;
    let N_max = 512;
    let error_calc = ErrorCalc::Absolute;
    let (a_1, error, _) = chebyshev_adaptive(&f, a, b, N0, epsilon, N_max, error_calc).unwrap();

    assert!(chebyshev_adaptive(&failing, a, b, N0, epsilon, N_max, error_calc).is_err());

    let x1 = 0.25;
    let fx_approx = chebyshev_approximate(a_1.clone(), a, b, x1);
    let fx = f(x1).unwrap();
    // Error between f(x) and ~f(x) must be < epsilon 
    assert!((fx_approx - fx).abs() < epsilon);
    // Output error must be less than input error
    assert!(error < epsilon);

    for a in a_1.clone().iter() {
        println!("{}", a);
    }
    let a_2 = truncate_chebyshev_coefficients(a_1.clone()).unwrap();
        for a in a_2.clone().iter() {
        println!("{}", a);
    }
    // truncation should remove terms in this case - doubling degree goes higher than degree of f
    assert!(a_2.clone().len() < a_1.len());
    // a_2 after truncation should be as long as degree of f + 1 if f is polynomial
    assert!(a_2.len() == 5);
}

#[test]
fn test_chebyshev_subdivision() {

    let f = |x: f64| -> Result<f64, std::convert::Infallible> { Ok(((x - 0.5)/1e-6).tanh())};
    let intervals = vec![(0.0, 1.0)];
    let N0 = 1;
    let N_max = 2048;
    let epsilon = 1e-5;
    let interval_limit = 1e-10;
    let error_calc = ErrorCalc::Relative;

    // Ensure Chebyshev correctly subdivides for step function
    let (output_intervals, _, _) = chebyshev_subdivide(&f, intervals, N0, epsilon, N_max, interval_limit, error_calc).unwrap();
    assert!(output_intervals.len() > 1);

    // Ensure it returns error for zero-width interval
    assert!(chebyshev_subdivide(&f, vec![(0.0, 0.0)], N0, epsilon, N_max, interval_limit, error_calc).is_err());

    // Ensure it returns error for N0 == 0
    assert!(chebyshev_subdivide(&f, vec![(0.0, 1.0)], 0, epsilon, N_max, interval_limit, error_calc).is_err());

    // Ensure it returns error for N_max < N0
    assert!(chebyshev_subdivide(&f, vec![(0.0, 1.0)], 2, epsilon, 1, interval_limit, error_calc).is_err());

    // Ensure it returns error for bad interval limit
    assert!(chebyshev_subdivide(&f, vec![(0.0, 1.0)], 2, epsilon, 4, -1.0, error_calc).is_err());

    // Ensure it fails for failing function
    assert!(chebyshev_subdivide(&failing, vec![(0.0, 1.0)], N0, epsilon, N_max, interval_limit, error_calc).is_err());
}

#[test]
fn test_lobatto_grid() {
    let a = 0.0;
    let b = 1.0;
    let N = 5;
    let grid = lobatto_grid(a, b, N).unwrap();

    // Lobatto grid has N + 1 points
    assert!(grid.len() == N + 1);

    // Lobatto grid cannot be constructed with an invalid interval
    assert!(lobatto_grid(0.0, 0.0, N).is_err());
    assert!(lobatto_grid(0.0, -1.0, N).is_err());

    // Lobatto grid cannot be degree 0 or lower
    assert!(lobatto_grid(a, b, 0).is_err());

    // Lobatto grid endpoints should be interval endpoints
    assert!(grid.clone()[0] == b);
    assert!(*grid.last().unwrap() == a);
}

#[test]
fn dynamic_range() {
    let a = 0.0;
    let b = 8.5;
    let mut config = Config::default();
    config.epsilon = 1e-9;
    config.error_calc = ErrorCalc::Relative;
    let mut roots = find_roots(&q, vec![(a, b)], config).unwrap();
    roots.sort_by(|a, b| a.total_cmp(b));

    let mut roots_polished = vec![];
    for &root in roots.iter() {
        let new_root = secant_polish(&q, root, 1000, f64::EPSILON).unwrap();
        roots_polished.push(new_root);
        println!("{} -> {}", root, new_root);
    }
    assert!((roots_polished[0] - 1.0).abs() < config.epsilon);
    assert!((roots_polished.last().unwrap() - 3.0).abs() < config.epsilon);
}

#[test]
fn test_roots_near_boundaries() {
    let a = -1.0;
    let b = 1.0;
    let config = Config::default();

    let mut roots = find_roots(&h, vec![(a, -0.5), (-0.5, 0.5), (0.5, b)], config).unwrap();
    roots.sort_by(|a, b| a.total_cmp(b));
    
    for root in roots.iter() {
        println!("Boundary Root: {}", root);
    }
    
    assert!((roots.last().unwrap() - 1.0).abs() < config.epsilon);
    assert!((roots[0] + 1.0).abs() < config.epsilon, "Root {} should be ~-1 with epsilon={}", roots[0], config.epsilon);
}

#[test]
fn test_rootfinding_with_newton() {

    let a = -10.;
    let b = 10.;
    let config = Config::default();

    let roots = find_roots_piecewise_with_newton_polishing(&g, &f, &df, vec![(a, -2E-4), (-2E-4, 0.0), (0.0, 2E-5), (2E-5, b)], config).unwrap();
    let num_roots = roots.len();

    println!("Identified {} roots.", num_roots);
    for root in roots.iter() {
        println!("Root: {}", root);
    }
    println!("Sum of roots found: {}; Expected value: {}", roots.iter().sum::<f64>(), -4.00009);
    assert_eq!(7, num_roots, "Rootfinder should find 7 roots. It found {}", num_roots);
    assert!((roots.iter().sum::<f64>() - -4.00009).powf(2.) < 0.01, "Sum of all roots should be -4.00009. Rootfinder found {}", roots.iter().sum::<f64>());
}

#[test]
fn test_rootfinding_with_secant() {
    let a = -10.;
    let b = 10.;
    let config = Config::default();

    let roots = find_roots_piecewise_with_secant_polishing(&g, &f, vec![(a, -2E-4), (-2E-4, 0.0), (0.0, 2E-5), (2E-5, b)], config).unwrap();
    let num_roots = roots.len();

    println!("Identified {} roots.", num_roots);
    for root in roots.iter() {
        println!("Root: {}", root);
    }
    println!("Sum of roots found: {}; Expected value: {}", roots.iter().sum::<f64>(), -4.00009);
    assert_eq!(7, num_roots, "Rootfinder should find 7 roots. It found {}", num_roots);
    assert!((roots.iter().sum::<f64>() - -4.00009).powf(2.) < 0.01, "Sum of all roots should be -4.00009. Rootfinder found {}", roots.iter().sum::<f64>());
}

#[test]
fn test_polynom() {

    let q = |x: f64| -> Result<f64, std::convert::Infallible> { Ok(x.powi(4) + 4.2*x.powi(3) - 1.8*x.powi(2) - 13.*x + 9.6)};
    let dq = |x: f64| -> Result<f64, std::convert::Infallible> { Ok(4.*x*(x*(x + 3.15) - 0.9) - 13.)};
    let c_j: Vec<f64> = vec![1., 4.2, -1.8, -13., 9.6];

    let mut roots = real_polynomial_roots(c_j.clone(), 1e-8).unwrap();

    for root in roots.iter() {
        println!("{}", root)
    }

    let true_roots = vec![-3.2, -3., 1., 1.];

    assert!(roots.len() == true_roots.len());

    roots.sort_by(|a, b| a.total_cmp(b));

    for (root, true_root) in roots.iter().zip(&true_roots) {
        let polished_root_newton = newton_polish(&q, &dq, *root, 10000, 10.*f64::EPSILON).unwrap();
        let polished_root_secant = secant_polish(&q, *root, 10000, 10.*f64::EPSILON).unwrap();
        assert!((polished_root_newton - true_root).abs() < 1e-14);
        assert!((polished_root_secant - true_root).abs() < 1e-14);
    }
}

#[test]
fn test_rootfinders() {
    let q = |x: f64| -> Result<f64, std::convert::Infallible> { Ok(x.powi(4) + 4.2*x.powi(3) - 1.8*x.powi(2) - 13.*x + 9.6)};
    let dq = |x: f64| -> Result<f64, std::convert::Infallible> { Ok(4.*x*(x*(x + 3.15) - 0.9) - 13.)};
    let c_j: Vec<f64> = vec![1., 4.2, -1.8, -13., 9.6];

    let a = -4.0;
    let b = 4.0;
    let config = Config::default();
    let intervals = vec![(-4.0, 0.0), (0.0, 4.0)];
    let roots_1 = real_polynomial_roots(c_j, config.complex_threshold).unwrap();
    let roots_2 = find_roots(&q, intervals.clone(), config).unwrap();
    let roots_3 = find_roots_with_newton_polishing(&q, &q, &dq, a, b, config).unwrap();
    let roots_4 = find_roots_with_secant_polishing(&q, &q, a, b, config).unwrap();
    let roots_5 = find_roots_piecewise_with_newton_polishing(&q, &q, &dq, intervals.clone(), config).unwrap();
    let roots_6 = find_roots_piecewise_with_secant_polishing(&q, &q, intervals.clone(), config).unwrap();

    assert!((roots_1[0] + 3.2).abs() < config.epsilon);
    assert!((roots_2[0] + 3.2).abs() < config.epsilon);
    assert!((roots_3[0] + 3.2).abs() < config.epsilon);
    assert!((roots_4[0] + 3.2).abs() < config.epsilon);
    assert!((roots_5[0] + 3.2).abs() < config.epsilon);
    assert!((roots_6[0] + 3.2).abs() < config.epsilon);

    assert!(find_roots(&failing, intervals.clone(), config).is_err());
    assert!(find_roots_with_newton_polishing(&failing, &failing, &failing, a, b, config).is_err());
    assert!(find_roots_with_secant_polishing(&failing, &failing, a, b, config).is_err());
    assert!(find_roots_piecewise_with_newton_polishing(&failing, &failing, &failing, intervals.clone(), config).is_err());
    assert!(find_roots_piecewise_with_secant_polishing(&failing, &failing, intervals.clone(), config).is_err());
}