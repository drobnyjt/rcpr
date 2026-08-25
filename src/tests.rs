pub use crate::rootfinders::*;
pub use crate::polish::*;

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

#[test]
fn dynamic_range() {
    let a = 0.0;
    let b = 8.5;
    let mut config = Config::default();
    config.epsilon = 1e-4;
    let mut roots = find_roots(&q, vec![(a, b)], config).unwrap();
    roots.sort_by(|a, b| a.total_cmp(b));
        for root in roots.iter() {
        println!("Dynamic Range Root: {}", root);
    }
    assert!((roots[0] - 1.0).abs() < config.epsilon);
    assert!((roots.last().unwrap() - 3.0).abs() < config.epsilon);
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
    let dq = |x: f64| -> Result<f64, std::convert::Infallible> { Ok(4.*x.powi(3) + 3.*4.2*x.powi(2) - 2.*1.8*x - 13.)};

    let c_j: Vec<f64> = vec![1., 5.2, 3.4, -9.6];

    let mut roots = real_polynomial_roots(c_j.clone(), f64::EPSILON).unwrap();

    let true_roots = vec![-3.2, -3., 1.];

    roots.sort_by(|a, b| a.total_cmp(b));

    for (root, true_root) in roots.iter().zip(&true_roots) {
        let polished_root = newton_polish(&q, &dq, *root, 10000, 10.*f64::EPSILON).unwrap();
        assert!((polished_root - true_root).abs() < 1e-14)
    }
}
