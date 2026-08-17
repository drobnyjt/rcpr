use super::*;

pub fn newton_polish<F: Fn(f64) -> f64, D: Fn(f64) -> f64>(f: &F, df: &D, x0: f64, iter_max: usize, epsilon: f64) -> Result<f64, anyhow::Error> {

    if x0.is_nan() {
        return Err(anyhow!("Newton iteration guess is NaN. Check preceding calculation."))
    }

    let mut x = x0;

    for _ in 1..=iter_max {
        let df_x = df(x);
        let xn = x - f(x)/df_x;
        if xn.is_nan() || (df_x == 0.0) {
            return Err(anyhow!("NaN in Newton iteration."))
        }
        let err = (xn - x).abs();
        x = xn;
        if err < epsilon {
            return Ok(x);
        }
    }
    Err(anyhow!("Newton failed to converge after {} iterations.", iter_max))
}

pub fn secant_polish<F: Fn(f64) -> f64>(f: &F, x0: f64, iter_max: usize, epsilon: f64) -> Result<f64, anyhow::Error> {

    if x0.is_nan() {
        return Err(anyhow!("Secant iteration guess is NaN. Check preceding calculation."))
    }

    let mut x1 = x0;
    let mut x2 = if x0.abs() > 0.0 {
        x0*1.25
    } else {
        1e-12
    };
    for _ in 1..=iter_max {

        let f2 = f(x2);
        let x3 = x2 - f2*(x2 - x1)/(f2 - f(x1));

        //let err = (x3 - x2)*(x3 - x2);
        // This error is the absolute residual instead of the difference.
        let err = f(x3).abs();

        if err < epsilon {
            return Ok(x3)
        }
        x1 = x2;
        x2 = x3;
    }
    Err(anyhow!("Secant failed to converge after {} iterations.", iter_max))
}

pub fn bisection_polish<F: Fn(f64) -> f64>(f: &F, a0: f64, b0: f64, iter_max: usize, epsilon: f64) -> Result<f64, anyhow::Error> {
    let mut a = a0;
    let mut b = b0;

    assert!(f(a)*f(b) < 0., "There is an even number of roots of f(x) on the interval [{}, {}]. Cannot use bisection.", a, b);
    assert!(a < b, "[{}, {}] is not a valid interval.", a, b);

    for _ in 0..iter_max {
        let c = (a + b)/2.;
        let fc = f(c);
        if fc.abs() < epsilon {
            return Ok(c)
        }
        if f(a)*fc < 0. {
            b = c;
        } else if fc*f(b) < 0. {
            a = c;
        } else {
            return Err(anyhow!("Bisection failed to find root in interval [{}, {}]", a, b))
        }
    }
    Err(anyhow!("Bisection failed to converge."))
}

pub fn newton_iteration<F: Fn(f64) -> f64, D: Fn(f64) -> f64>(f: &F, df: &D, x0: f64) -> f64 {
    x0 - f(x0)/df(x0)
}

pub fn newton_correction<F: Fn(f64) -> f64, D: Fn(f64) -> f64>(f: &F, df: &D, x0: f64) -> f64 {
    f(x0)/df(x0)
}