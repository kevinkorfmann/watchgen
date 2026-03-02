/// SMC++ parameter inference via coordinate descent in log-space.

use crate::hmm::composite_log_likelihood;

/// Configuration for SMC++ inference.
pub struct SmcppConfig {
    pub time_breaks: Vec<f64>,
    pub theta: f64,
    pub rho: f64,
    pub max_iter: usize,
}

/// Result of SMC++ inference.
pub struct SmcppResult {
    pub lambdas: Vec<f64>,
    pub log_likelihood: f64,
    pub iterations: usize,
}

/// Roughness penalty: sum of squared differences in log-lambda.
///
/// Encourages smooth population size trajectories.
fn roughness_penalty(log_lambdas: &[f64], weight: f64) -> f64 {
    let mut penalty = 0.0;
    for i in 1..log_lambdas.len() {
        let diff = log_lambdas[i] - log_lambdas[i - 1];
        penalty += diff * diff;
    }
    -weight * penalty
}

/// Fit SMC++ model using coordinate descent in log-space.
///
/// Optimizes piecewise-constant population sizes lambda(t) with a
/// roughness penalty to encourage smooth trajectories.
pub fn fit_smcpp(data: &[Vec<u8>], config: &SmcppConfig) -> SmcppResult {
    let k = config.time_breaks.len() - 1;
    let mut log_lambdas = vec![0.0; k]; // start at lambda=1

    // Roughness penalty: strong enough to prevent extreme swings,
    // weak enough to allow the data to drive the fit
    let n_obs: usize = data.iter().map(|d| d.len()).sum();
    let penalty_weight = (n_obs as f64).sqrt();

    let penalized_ll = |log_lam: &[f64]| -> f64 {
        let lambdas: Vec<f64> = log_lam.iter().map(|x| x.exp()).collect();
        composite_log_likelihood(data, &config.time_breaks, &lambdas, config.theta, config.rho)
            + roughness_penalty(log_lam, penalty_weight)
    };

    let mut best_ll = penalized_ll(&log_lambdas);

    for iter in 0..config.max_iter {
        let mut improved = false;
        for i in 0..k {
            let current = log_lambdas[i];
            let mut best_val = current;
            let mut best_q = best_ll;

            // Try candidate values at multiple scales
            for &delta in &[-1.0, -0.5, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.5, 1.0] {
                log_lambdas[i] = current + delta;
                let ll = penalized_ll(&log_lambdas);
                if ll > best_q {
                    best_q = ll;
                    best_val = current + delta;
                }
            }
            log_lambdas[i] = best_val;

            if best_val != current {
                best_ll = best_q;
                improved = true;
            }
        }

        if !improved && iter > 0 {
            let lambdas: Vec<f64> = log_lambdas.iter().map(|x| x.exp()).collect();
            let raw_ll = composite_log_likelihood(
                data, &config.time_breaks, &lambdas, config.theta, config.rho,
            );
            return SmcppResult {
                lambdas,
                log_likelihood: raw_ll,
                iterations: iter + 1,
            };
        }
    }

    let lambdas: Vec<f64> = log_lambdas.iter().map(|x| x.exp()).collect();
    let raw_ll = composite_log_likelihood(
        data, &config.time_breaks, &lambdas, config.theta, config.rho,
    );
    SmcppResult {
        lambdas,
        log_likelihood: raw_ll,
        iterations: config.max_iter,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fit_smcpp_improves_ll() {
        let time_breaks: Vec<f64> = (0..=5).map(|i| i as f64).collect();
        let data = vec![vec![0u8, 0, 1, 0, 0, 1, 0, 0, 0, 1]];

        let config = SmcppConfig {
            time_breaks: time_breaks.clone(),
            theta: 0.001,
            rho: 0.0002,
            max_iter: 5,
        };

        let initial_ll = composite_log_likelihood(
            &data,
            &time_breaks,
            &vec![1.0; 5],
            config.theta,
            config.rho,
        );

        let result = fit_smcpp(&data, &config);
        assert!(
            result.log_likelihood >= initial_ll - 1e-10,
            "fit should not decrease LL"
        );
    }
}
