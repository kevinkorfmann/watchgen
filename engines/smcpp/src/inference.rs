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

/// Fit SMC++ model using coordinate descent in log-space.
///
/// Optimizes piecewise-constant population sizes lambda(t).
pub fn fit_smcpp(data: &[Vec<u8>], config: &SmcppConfig) -> SmcppResult {
    let k = config.time_breaks.len() - 1;
    let mut log_lambdas = vec![0.0; k]; // start at lambda=1

    let mut best_ll =
        composite_log_likelihood(data, &config.time_breaks, &vec![1.0; k], config.theta, config.rho);

    for iter in 0..config.max_iter {
        let mut improved = false;
        for i in 0..k {
            let current = log_lambdas[i];
            let mut best_val = current;
            let mut best_q = best_ll;

            // Try a few candidate values
            for &delta in &[-0.5, -0.2, -0.1, 0.1, 0.2, 0.5] {
                let trial = current + delta;
                let mut trial_lambdas: Vec<f64> = log_lambdas.iter().map(|x| (*x as f64).exp()).collect();
                trial_lambdas[i] = (trial as f64).exp();

                let ll = composite_log_likelihood(
                    data,
                    &config.time_breaks,
                    &trial_lambdas,
                    config.theta,
                    config.rho,
                );
                if ll > best_q {
                    best_q = ll;
                    best_val = trial;
                }
            }

            if best_val != current {
                log_lambdas[i] = best_val;
                best_ll = best_q;
                improved = true;
            }
        }

        if !improved && iter > 0 {
            return SmcppResult {
                lambdas: log_lambdas.iter().map(|x| (*x as f64).exp()).collect(),
                log_likelihood: best_ll,
                iterations: iter + 1,
            };
        }
    }

    SmcppResult {
        lambdas: log_lambdas.iter().map(|x| (*x as f64).exp()).collect(),
        log_likelihood: best_ll,
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
