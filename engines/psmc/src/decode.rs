use serde::Serialize;

use crate::discretize::compute_time_intervals;

/// Scaled PSMC output in real units.
#[derive(Debug, Clone, Serialize)]
pub struct ScaledOutput {
    pub n0: f64,
    pub times_years: Vec<f64>,
    pub pop_sizes: Vec<f64>,
}

/// Scale PSMC output to real units.
///
/// - `N_0 = theta_0 / (4 * mu * s)`
/// - `times = 2 * N_0 * t * generation_time`
/// - `pop_sizes = N_0 * lambdas`
pub fn scale_psmc_output(
    theta_0: f64,
    lambdas: &[f64],
    t_boundaries: &[f64],
    mu: f64,
    s: f64,
    generation_time: f64,
) -> ScaledOutput {
    let n0 = theta_0 / (4.0 * mu * s);
    let times_years: Vec<f64> = t_boundaries.iter().map(|&t| 2.0 * n0 * t * generation_time).collect();
    let pop_sizes: Vec<f64> = lambdas.iter().map(|&lam| n0 * lam).collect();

    ScaledOutput {
        n0,
        times_years,
        pop_sizes,
    }
}

/// Generate step-function coordinates for PSMC plot.
///
/// Returns `(x_years, y_pop_sizes)`.
pub fn plot_psmc_history(
    theta_0: f64,
    lambdas: &[f64],
    t_boundaries: &[f64],
    mu: f64,
    s: f64,
    generation_time: f64,
) -> (Vec<f64>, Vec<f64>) {
    let scaled = scale_psmc_output(theta_0, lambdas, t_boundaries, mu, s, generation_time);
    let mut x = Vec::with_capacity(lambdas.len() * 2);
    let mut y = Vec::with_capacity(lambdas.len() * 2);

    for k in 0..lambdas.len() {
        x.push(scaled.times_years[k]);
        y.push(scaled.pop_sizes[k]);
        x.push(scaled.times_years[k + 1]);
        y.push(scaled.pop_sizes[k]);
    }

    (x, y)
}

/// Get the time boundaries for a given n, t_max, alpha_param.
pub fn get_t_boundaries(n: usize, t_max: f64, alpha_param: f64) -> Vec<f64> {
    compute_time_intervals(n, t_max, alpha_param)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_n0_calculation() {
        let theta_0 = 0.00069;
        let mu = 1.25e-8;
        let s = 100.0;
        let expected_n0 = theta_0 / (4.0 * mu * s);
        let scaled = scale_psmc_output(theta_0, &[1.0], &[0.0, 1.0], mu, s, 25.0);
        assert!((scaled.n0 - expected_n0).abs() < 1e-6);
    }

    #[test]
    fn test_time_scaling() {
        let theta_0 = 0.00069;
        let mu = 1.25e-8;
        let s = 100.0;
        let scaled = scale_psmc_output(theta_0, &[1.0, 1.0], &[0.0, 1.0, 2.0], mu, s, 25.0);
        assert!(scaled.times_years[0].abs() < 1e-10);
        let expected = 2.0 * scaled.n0 * 1.0 * 25.0;
        assert!((scaled.times_years[1] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_pop_size_scaling() {
        let theta_0 = 0.00069;
        let mu = 1.25e-8;
        let s = 100.0;
        let scaled = scale_psmc_output(theta_0, &[2.0, 0.5], &[0.0, 1.0, 2.0], mu, s, 25.0);
        assert!((scaled.pop_sizes[0] - scaled.n0 * 2.0).abs() < 1e-6);
        assert!((scaled.pop_sizes[1] - scaled.n0 * 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_step_function_structure() {
        let theta_0 = 0.001;
        let lambdas = vec![1.0, 2.0, 1.5];
        let t = vec![0.0, 1.0, 2.0, 3.0, 100.0];
        let (x, y) = plot_psmc_history(theta_0, &lambdas, &t, 1.25e-8, 100.0, 25.0);
        assert_eq!(x.len(), 6);
        assert_eq!(y.len(), 6);
        // Constant height within each interval
        assert_eq!(y[0], y[1]);
        assert_eq!(y[2], y[3]);
        assert_eq!(y[4], y[5]);
    }
}
