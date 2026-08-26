use super::*;

pub fn newton_polish<F, D, E>(f: &F, df: &D, x0: f64, iter_max: usize, delta: f64) -> Result<f64, anyhow::Error>
where F: Fn(f64) -> Result<f64, E>, D: Fn(f64) -> Result<f64, E>, E: std::error::Error + Send + Sync + 'static, {

    if x0.is_nan() {
        return Err(anyhow!("Newton iteration guess is NaN. Check preceding calculation."))
    }

    let mut x = x0;

    for _ in 1..=iter_max {
        let df_x = df(x)?;
        let xn = x - f(x)?/df_x;
        if xn.is_nan() || (df_x.abs() < f64::EPSILON) {
            return Err(anyhow!("NaN in Newton iteration."))
        }
        let err = hyberr(xn, x);
        x = xn;
        if err < delta {
            return Ok(x);
        }
    }
    Err(anyhow!("Newton failed to converge after {} iterations.", iter_max))
}

/// Hybrid Error: https://arxiv.org/html/2403.07492v2
fn hyberr(x: f64, y: f64) -> f64 {
    (x - y).abs()/(1. + y.abs())
}

pub fn secant_polish<F, E>(f: &F, x0: f64, iter_max: usize, delta: f64) -> Result<f64, anyhow::Error> where F: Fn(f64) -> Result<f64, E>, E: std::error::Error + Send + Sync + 'static, {

    if x0.is_nan() {
        return Err(anyhow!("Secant iteration guess is NaN. Check preceding calculation."))
    }

    let mut x1 = x0;
    let dx = x0.abs().max(1.0)*f64::EPSILON.sqrt();
    let mut x2 = x1 + dx;

    if hyberr(x2, x1) < delta {
        return Ok(x1)
    }
    
    for _ in 1..=iter_max {

        let f2 = f(x2)?;
        let f1 = f(x1)?;
        let df = f2 - f1;

        let x3 = x2 - f2*(x2 - x1)/df;

        //let err = (x3 - x2)*(x3 - x2);
        // This error is the absolute residual instead of the difference.
        let err = hyberr(x3, x2);

        if err < delta {
            return Ok(x3)
        }
        x1 = x2;
        x2 = x3;
    }
    Err(anyhow!("Secant failed to converge after {} iterations.", iter_max))
}