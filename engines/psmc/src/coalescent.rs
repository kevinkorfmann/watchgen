/// Estimate initial theta from the observed fraction of heterozygous sites.
///
/// Uses exact inversion: if P(het) = 1 - exp(-theta), then theta = -ln(1 - P(het)).
/// Only counts non-missing sites (values 0 and 1).
pub fn estimate_theta_initial(seq: &[u8]) -> f64 {
    let mut n_valid = 0u64;
    let mut n_het = 0u64;
    for &v in seq {
        if v < 2 {
            n_valid += 1;
            if v == 1 {
                n_het += 1;
            }
        }
    }
    if n_valid == 0 {
        return 0.0;
    }
    let frac_het = n_het as f64 / n_valid as f64;
    if frac_het >= 1.0 {
        return f64::INFINITY;
    }
    -(1.0 - frac_het).ln()
}

/// Compute cumulative hazard Lambda(t) for piecewise-constant lambda.
///
/// Lambda(t) = integral_0^t 1/lambda(u) du
///
/// No quadrature needed -- just a running sum over intervals.
pub fn cumulative_hazard_piecewise(t: f64, t_boundaries: &[f64], lambdas: &[f64]) -> f64 {
    let mut result = 0.0;
    for k in 0..lambdas.len() {
        let t_lo = t_boundaries[k];
        let t_hi = t_boundaries[k + 1];
        if t <= t_lo {
            break;
        }
        let dt = t.min(t_hi) - t_lo;
        result += dt / lambdas[k];
        if t <= t_hi {
            break;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_theta_recovery() {
        // P(het) = 1 - exp(-theta) => theta = -ln(1 - P(het))
        // For theta=0.001, P(het) ~ 0.001
        let theta_true = 0.001;
        let p_het = 1.0 - (-theta_true as f64).exp();
        let n = 1_000_000;
        let n_het = (p_het * n as f64).round() as usize;
        let mut seq = vec![0u8; n];
        for i in 0..n_het {
            seq[i] = 1;
        }
        let theta_est = estimate_theta_initial(&seq);
        assert!((theta_est - theta_true).abs() / theta_true < 0.01);
    }

    #[test]
    fn test_estimate_theta_zero() {
        let seq = vec![0u8; 1000];
        assert_eq!(estimate_theta_initial(&seq), 0.0);
    }

    #[test]
    fn test_estimate_theta_skips_missing() {
        // All missing
        let seq = vec![2u8; 100];
        assert_eq!(estimate_theta_initial(&seq), 0.0);
    }

    #[test]
    fn test_hazard_constant() {
        let t_bounds = vec![0.0, 5.0, 10.0, 1000.0];
        let lambdas = vec![1.0, 1.0, 1.0];
        for &t in &[0.5, 3.0, 7.5] {
            let h = cumulative_hazard_piecewise(t, &t_bounds, &lambdas);
            assert!((h - t).abs() < 1e-10);
        }
    }

    #[test]
    fn test_hazard_two_intervals() {
        let t_bounds = vec![0.0, 1.0, 1000.0];
        let lambdas = vec![2.0, 1.0];
        assert!((cumulative_hazard_piecewise(0.5, &t_bounds, &lambdas) - 0.25).abs() < 1e-10);
        assert!((cumulative_hazard_piecewise(1.5, &t_bounds, &lambdas) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hazard_at_boundary() {
        let t_bounds = vec![0.0, 1.0, 1000.0];
        let lambdas = vec![2.0, 1.0];
        assert!((cumulative_hazard_piecewise(1.0, &t_bounds, &lambdas) - 0.5).abs() < 1e-10);
    }
}
