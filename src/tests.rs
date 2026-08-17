pub use crate::chebyshev::*;
pub use crate::rootfinders::*;

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
