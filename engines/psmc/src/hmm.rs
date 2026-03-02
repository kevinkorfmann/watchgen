use ndarray::Array2;

use crate::discretize::build_psmc_hmm_parts;

/// Complete PSMC Hidden Markov Model.
pub struct PsmcHmm {
    pub n: usize,
    pub n_states: usize,
    pub theta: f64,
    pub rho: f64,
    pub t_max: f64,
    pub alpha_param: f64,
    pub lambdas: Vec<f64>,
    pub transitions: Array2<f64>,
    pub emissions: Array2<f64>,
    pub initial: Vec<f64>,
    pub t_boundaries: Vec<f64>,
    pub avg_times: Vec<f64>,
}

/// Expected counts from the E-step.
pub struct ExpectedCounts {
    /// Sum of posterior state probabilities across all positions.
    pub gamma_sum: Vec<f64>,
    /// Sum of expected transition counts.
    pub xi_sum: Array2<f64>,
    /// Emission counts: `emission_counts[b][k]` for observation b, state k.
    pub emission_counts: Array2<f64>,
    /// Log-likelihood of the observation sequence.
    pub log_likelihood: f64,
}

impl PsmcHmm {
    pub fn new(
        n: usize,
        theta: f64,
        rho: f64,
        lambdas: &[f64],
        t_max: f64,
        alpha_param: f64,
    ) -> Self {
        let (transitions, emissions, initial, t_boundaries, avg_times) =
            build_psmc_hmm_parts(n, t_max, theta, rho, lambdas, alpha_param);

        Self {
            n,
            n_states: n + 1,
            theta,
            rho,
            t_max,
            alpha_param,
            lambdas: lambdas.to_vec(),
            transitions,
            emissions,
            initial,
            t_boundaries,
            avg_times,
        }
    }

    /// Compute log-likelihood of an observation sequence.
    pub fn log_likelihood(&self, seq: &[u8]) -> f64 {
        let (_, ll) = self.forward_scaled(seq);
        ll
    }

    /// Scaled forward algorithm.
    ///
    /// Returns `(alpha_hat, log_likelihood)` where `alpha_hat` has shape `(L, N)`.
    pub fn forward_scaled(&self, seq: &[u8]) -> (Array2<f64>, f64) {
        let len = seq.len();
        let n_states = self.n_states;
        let mut alpha_hat = Array2::<f64>::zeros((len, n_states));
        let mut log_likelihood = 0.0;

        // Initialize
        let obs0 = seq[0];
        for k in 0..n_states {
            let e = if obs0 >= 2 {
                1.0
            } else {
                self.emissions[[obs0 as usize, k]]
            };
            alpha_hat[[0, k]] = self.initial[k] * e;
        }

        let c0: f64 = alpha_hat.row(0).sum();
        if c0 > 0.0 {
            for k in 0..n_states {
                alpha_hat[[0, k]] /= c0;
            }
            log_likelihood += c0.ln();
        }

        // Recurse
        for a in 1..len {
            let obs = seq[a];
            for l in 0..n_states {
                let e = if obs >= 2 {
                    1.0
                } else {
                    self.emissions[[obs as usize, l]]
                };
                let mut s = 0.0;
                for k in 0..n_states {
                    s += alpha_hat[[a - 1, k]] * self.transitions[[k, l]];
                }
                alpha_hat[[a, l]] = s * e;
            }

            let c: f64 = alpha_hat.row(a).sum();
            if c > 0.0 {
                for l in 0..n_states {
                    alpha_hat[[a, l]] /= c;
                }
                log_likelihood += c.ln();
            }
        }

        (alpha_hat, log_likelihood)
    }

    /// Scaled backward algorithm.
    pub fn backward_scaled(&self, seq: &[u8], _alpha_hat: &Array2<f64>) -> Array2<f64> {
        let len = seq.len();
        let n_states = self.n_states;
        let mut beta_hat = Array2::<f64>::zeros((len, n_states));

        // Initialize last row
        for k in 0..n_states {
            beta_hat[[len - 1, k]] = 1.0;
        }

        // Recurse backwards
        for a in (0..len - 1).rev() {
            let obs_next = seq[a + 1];
            for k in 0..n_states {
                let mut s = 0.0;
                for l in 0..n_states {
                    let e = if obs_next >= 2 {
                        1.0
                    } else {
                        self.emissions[[obs_next as usize, l]]
                    };
                    s += self.transitions[[k, l]] * e * beta_hat[[a + 1, l]];
                }
                beta_hat[[a, k]] = s;
            }

            let c: f64 = beta_hat.row(a).sum();
            if c > 0.0 {
                for k in 0..n_states {
                    beta_hat[[a, k]] /= c;
                }
            }
        }

        beta_hat
    }
}

/// Compute expected counts for the E-step of EM.
pub fn compute_expected_counts(hmm: &PsmcHmm, seq: &[u8]) -> ExpectedCounts {
    let len = seq.len();
    let n_states = hmm.n_states;

    let (alpha_hat, ll) = hmm.forward_scaled(seq);
    let beta_hat = hmm.backward_scaled(seq, &alpha_hat);

    let mut gamma_sum = vec![0.0; n_states];
    let mut emission_counts = Array2::<f64>::zeros((2, n_states));

    for pos in 0..len {
        let mut gamma = vec![0.0; n_states];
        let mut total = 0.0;
        for k in 0..n_states {
            gamma[k] = alpha_hat[[pos, k]] * beta_hat[[pos, k]];
            total += gamma[k];
        }
        if total > 0.0 {
            for g in gamma.iter_mut() {
                *g /= total;
            }
        }

        for k in 0..n_states {
            gamma_sum[k] += gamma[k];
        }

        let obs = seq[pos];
        if obs < 2 {
            for k in 0..n_states {
                emission_counts[[obs as usize, k]] += gamma[k];
            }
        }
    }

    let mut xi_sum = Array2::<f64>::zeros((n_states, n_states));
    for pos in 0..(len - 1) {
        let obs_next = seq[pos + 1];

        // Compute normalization factor for this position first
        let mut pos_total = 0.0;
        for k in 0..n_states {
            for l in 0..n_states {
                let e = if obs_next >= 2 {
                    1.0
                } else {
                    hmm.emissions[[obs_next as usize, l]]
                };
                pos_total +=
                    alpha_hat[[pos, k]] * hmm.transitions[[k, l]] * e * beta_hat[[pos + 1, l]];
            }
        }

        // Add per-position normalized xi values
        if pos_total > 0.0 {
            for k in 0..n_states {
                for l in 0..n_states {
                    let e = if obs_next >= 2 {
                        1.0
                    } else {
                        hmm.emissions[[obs_next as usize, l]]
                    };
                    xi_sum[[k, l]] += alpha_hat[[pos, k]]
                        * hmm.transitions[[k, l]]
                        * e
                        * beta_hat[[pos + 1, l]]
                        / pos_total;
                }
            }
        }
    }

    ExpectedCounts {
        gamma_sum,
        xi_sum,
        emission_counts,
        log_likelihood: ll,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_hmm() -> PsmcHmm {
        let n = 10;
        let theta = 0.001;
        let rho = theta / 5.0;
        let lambdas = vec![1.0; n + 1];
        PsmcHmm::new(n, theta, rho, &lambdas, 15.0, 0.1)
    }

    fn make_test_seq() -> Vec<u8> {
        // Deterministic test sequence
        let mut seq = vec![0u8; 1000];
        for i in (0..1000).step_by(10) {
            seq[i] = 1;
        }
        seq
    }

    #[test]
    fn test_forward_backward_ll_consistency() {
        let hmm = make_test_hmm();
        let seq = make_test_seq();
        let (alpha_hat, ll_fwd) = hmm.forward_scaled(&seq);
        // Forward LL should be finite and negative
        assert!(ll_fwd.is_finite());
        assert!(ll_fwd < 0.0);
        // Backward should produce valid matrix
        let beta_hat = hmm.backward_scaled(&seq, &alpha_hat);
        assert_eq!(beta_hat.shape(), &[1000, 11]);
    }

    #[test]
    fn test_gamma_sum_equals_length() {
        let hmm = make_test_hmm();
        let seq = make_test_seq();
        let counts = compute_expected_counts(&hmm, &seq);
        let total: f64 = counts.gamma_sum.iter().sum();
        assert!(
            (total - seq.len() as f64).abs() < 1.0,
            "gamma_sum total = {total}, expected {}", seq.len()
        );
    }

    #[test]
    fn test_missing_data_handling() {
        let hmm = make_test_hmm();
        let seq = vec![2u8; 100]; // all missing
        let (_, ll) = hmm.forward_scaled(&seq);
        assert!(ll.is_finite());
    }

    #[test]
    fn test_expected_counts_shapes() {
        let hmm = make_test_hmm();
        let seq = make_test_seq();
        let counts = compute_expected_counts(&hmm, &seq);
        assert_eq!(counts.gamma_sum.len(), 11);
        assert_eq!(counts.xi_sum.shape(), &[11, 11]);
        assert_eq!(counts.emission_counts.shape(), &[2, 11]);
    }
}
