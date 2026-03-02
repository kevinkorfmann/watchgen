use ndarray::Array2;

/// Helper quantities from `compute_helpers`.
pub struct Discretization {
    pub n: usize,
    pub n_states: usize,
    pub t_boundaries: Vec<f64>,
    pub tau: Vec<f64>,
    pub alpha: Vec<f64>,
    pub beta: Vec<f64>,
    pub q_aux: Vec<f64>,
    pub c_pi: f64,
}

/// Compute PSMC time interval boundaries with log-spacing.
///
/// Returns `n + 2` boundaries: `[t_0, t_1, ..., t_n, t_{n+1}]`.
/// The last boundary is set to 1000.0 (effectively infinity).
pub fn compute_time_intervals(n: usize, t_max: f64, alpha: f64) -> Vec<f64> {
    let beta = (1.0 + t_max / alpha).ln() / n as f64;
    let mut t = vec![0.0; n + 2];
    for (k, val) in t.iter_mut().enumerate().take(n) {
        *val = alpha * ((beta * k as f64).exp() - 1.0);
    }
    t[n] = t_max;
    t[n + 1] = 1000.0;
    t
}

/// Compute helper quantities for the discrete PSMC.
///
/// Returns `(tau, alpha, beta, q_aux, C_pi)`.
pub fn compute_helpers(n: usize, t: &[f64], lambdas: &[f64]) -> Discretization {
    let n_states = n + 1;
    let mut tau = vec![0.0; n_states];
    let mut alpha = vec![0.0; n + 2];
    let mut beta = vec![0.0; n_states];
    let mut q_aux = vec![0.0; n];

    for k in 0..n_states {
        tau[k] = t[k + 1] - t[k];
    }

    alpha[0] = 1.0;
    for k in 1..n_states {
        alpha[k] = alpha[k - 1] * (-tau[k - 1] / lambdas[k - 1]).exp();
    }
    alpha[n + 1] = 0.0;

    beta[0] = 0.0;
    for k in 1..n_states {
        beta[k] = beta[k - 1] + lambdas[k - 1] * (1.0 / alpha[k] - 1.0 / alpha[k - 1]);
    }

    for k in 0..n {
        let ak1 = alpha[k] - alpha[k + 1];
        q_aux[k] = ak1 * (beta[k] - lambdas[k] / alpha[k]) + tau[k];
    }

    let mut c_pi = 0.0;
    for k in 0..n_states {
        c_pi += lambdas[k] * (alpha[k] - alpha[k + 1]);
    }

    Discretization {
        n,
        n_states,
        t_boundaries: t.to_vec(),
        tau,
        alpha,
        beta,
        q_aux,
        c_pi,
    }
}

/// Compute discrete stationary distributions pi_k and sigma_k.
///
/// Returns `(pi_k, sigma_k, C_sigma)`.
pub fn compute_stationary(
    disc: &Discretization,
    lambdas: &[f64],
    rho: f64,
) -> (Vec<f64>, Vec<f64>, f64) {
    let n_states = disc.n_states;
    let mut pi_k = vec![0.0; n_states];
    let mut sum_tau = 0.0;

    for k in 0..n_states {
        let ak1 = disc.alpha[k] - disc.alpha[k + 1];
        pi_k[k] = (ak1 * (sum_tau + lambdas[k]) - disc.alpha[k + 1] * disc.tau[k]) / disc.c_pi;
        sum_tau += disc.tau[k];
    }

    let c_sigma = 1.0 / (disc.c_pi * rho) + 0.5;

    let mut sigma_k = vec![0.0; n_states];
    for k in 0..n_states {
        let ak1 = disc.alpha[k] - disc.alpha[k + 1];
        sigma_k[k] = (ak1 / (disc.c_pi * rho) + pi_k[k] / 2.0) / c_sigma;
    }

    // Clamp to avoid tiny negatives from floating-point
    for k in 0..n_states {
        if pi_k[k] < 0.0 {
            pi_k[k] = 0.0;
        }
        if sigma_k[k] < 0.0 {
            sigma_k[k] = 0.0;
        }
    }

    // Renormalize sigma_k
    let sigma_sum: f64 = sigma_k.iter().sum();
    if sigma_sum > 0.0 {
        for v in &mut sigma_k {
            *v /= sigma_sum;
        }
    }

    (pi_k, sigma_k, c_sigma)
}

/// Compute the full discrete PSMC transition matrix.
///
/// Returns `(p, q)` where `p` is the full transition matrix and `q` is
/// the transition-given-recombination matrix.
pub fn compute_transition_matrix(
    disc: &Discretization,
    lambdas: &[f64],
    c_sigma: f64,
    pi_k: &[f64],
    sigma_k: &[f64],
) -> (Array2<f64>, Array2<f64>) {
    let n = disc.n;
    let n_states = disc.n_states;
    let mut q = Array2::<f64>::zeros((n_states, n_states));

    for k in 0..n_states {
        let ak1 = disc.alpha[k] - disc.alpha[k + 1];
        let sum_tau_k: f64 = disc.tau[..k].iter().sum();
        let cpik = ak1 * (sum_tau_k + lambdas[k]) - disc.alpha[k + 1] * disc.tau[k];

        if cpik < 1e-30 {
            for l in 0..n_states {
                q[[k, l]] = 1.0 / n_states as f64;
            }
            continue;
        }

        for l in 0..k {
            q[[k, l]] = ak1 / cpik * disc.q_aux[l];
        }

        q[[k, k]] = (ak1 * ak1 * (disc.beta[k] - lambdas[k] / disc.alpha[k])
            + 2.0 * lambdas[k] * ak1
            - 2.0 * disc.alpha[k + 1] * disc.tau[k])
            / cpik;

        if k < n {
            for l in (k + 1)..n_states {
                q[[k, l]] = (disc.alpha[l] - disc.alpha[l + 1]) / cpik * disc.q_aux[k];
            }
        }
    }

    let mut p = Array2::<f64>::zeros((n_states, n_states));
    for k in 0..n_states {
        let recomb_prob = if sigma_k[k] > 0.0 {
            pi_k[k] / (c_sigma * sigma_k[k])
        } else {
            0.0
        };
        for l in 0..n_states {
            p[[k, l]] = recomb_prob * q[[k, l]];
            if k == l {
                p[[k, l]] += 1.0 - recomb_prob;
            }
        }
    }

    (p, q)
}

/// Compute the effective coalescence time for each interval.
pub fn compute_avg_times(
    disc: &Discretization,
    lambdas: &[f64],
    pi_k: &[f64],
    sigma_k: &[f64],
    c_sigma: f64,
    rho: f64,
) -> Vec<f64> {
    let n_states = disc.n_states;
    let mut avg_t = vec![0.0; n_states];
    let mut sum_tau = 0.0;

    for k in 0..n_states {
        let ak1 = disc.alpha[k] - disc.alpha[k + 1];
        let recomb_prob = if sigma_k[k] > 0.0 {
            pi_k[k] / (c_sigma * sigma_k[k])
        } else {
            0.0
        };

        if recomb_prob < 1.0 {
            avg_t[k] = -(1.0 - recomb_prob).ln() / rho;
        } else {
            let lak = lambdas[k];
            avg_t[k] = sum_tau
                + if ak1 > 0.0 {
                    lak - disc.tau[k] * disc.alpha[k + 1] / ak1
                } else {
                    disc.tau[k] / 2.0
                };
        }

        if avg_t[k].is_nan() || avg_t[k] < sum_tau || avg_t[k] > sum_tau + disc.tau[k] {
            let lak = lambdas[k];
            avg_t[k] = sum_tau
                + if ak1 > 0.0 {
                    lak - disc.tau[k] * disc.alpha[k + 1] / ak1
                } else {
                    disc.tau[k] / 2.0
                };
        }

        sum_tau += disc.tau[k];
    }

    avg_t
}

/// Build complete PSMC HMM parts from parameters.
///
/// Returns `(transitions, emissions, initial, t_boundaries, avg_times)`.
#[allow(clippy::type_complexity)]
pub fn build_psmc_hmm_parts(
    n: usize,
    t_max: f64,
    theta: f64,
    rho: f64,
    lambdas: &[f64],
    alpha_param: f64,
) -> (Array2<f64>, Array2<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n_states = n + 1;
    let t = compute_time_intervals(n, t_max, alpha_param);
    let disc = compute_helpers(n, &t, lambdas);
    let (pi_k, sigma_k, c_sigma) = compute_stationary(&disc, lambdas, rho);
    let (transitions, _) = compute_transition_matrix(&disc, lambdas, c_sigma, &pi_k, &sigma_k);
    let avg_t = compute_avg_times(&disc, lambdas, &pi_k, &sigma_k, c_sigma, rho);

    let mut emissions = Array2::<f64>::zeros((2, n_states));
    for k in 0..n_states {
        emissions[[0, k]] = (-theta * avg_t[k]).exp();
        emissions[[1, k]] = 1.0 - emissions[[0, k]];
    }

    let initial = sigma_k;

    (transitions, emissions, initial, t, avg_t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_intervals_first_is_zero() {
        let t = compute_time_intervals(10, 15.0, 0.1);
        assert_eq!(t[0], 0.0);
    }

    #[test]
    fn test_time_intervals_nth_is_tmax() {
        let t = compute_time_intervals(10, 15.0, 0.1);
        assert!((t[10] - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_time_intervals_last_is_large() {
        let t = compute_time_intervals(10, 15.0, 0.1);
        assert_eq!(t[11], 1000.0);
    }

    #[test]
    fn test_time_intervals_increasing() {
        let t = compute_time_intervals(20, 15.0, 0.1);
        for i in 0..t.len() - 1 {
            assert!(t[i] < t[i + 1]);
        }
    }

    #[test]
    fn test_time_intervals_length() {
        let t = compute_time_intervals(63, 15.0, 0.1);
        assert_eq!(t.len(), 65);
    }

    #[test]
    fn test_time_intervals_log_spacing() {
        let t = compute_time_intervals(20, 15.0, 0.1);
        let w_first = t[1] - t[0];
        let w_later = t[10] - t[9];
        assert!(w_later > w_first);
    }

    #[test]
    fn test_alpha_boundary_conditions() {
        let n = 10;
        let t = compute_time_intervals(n, 15.0, 0.1);
        let lambdas = vec![1.0; n + 1];
        let disc = compute_helpers(n, &t, &lambdas);
        assert_eq!(disc.alpha[0], 1.0);
        assert_eq!(disc.alpha[n + 1], 0.0);
    }

    #[test]
    fn test_alpha_decreasing() {
        let n = 10;
        let t = compute_time_intervals(n, 15.0, 0.1);
        let lambdas = vec![1.0; n + 1];
        let disc = compute_helpers(n, &t, &lambdas);
        for k in 0..=n {
            assert!(disc.alpha[k] >= disc.alpha[k + 1]);
        }
    }

    #[test]
    fn test_c_pi_constant_pop() {
        let n = 10;
        let t = compute_time_intervals(n, 15.0, 0.1);
        let lambdas = vec![1.0; n + 1];
        let disc = compute_helpers(n, &t, &lambdas);
        assert!((disc.c_pi - 1.0).abs() < 0.05);
    }

    #[test]
    fn test_tau_positive() {
        let n = 10;
        let t = compute_time_intervals(n, 15.0, 0.1);
        let lambdas = vec![1.0; n + 1];
        let disc = compute_helpers(n, &t, &lambdas);
        for k in 0..disc.n_states {
            assert!(disc.tau[k] > 0.0);
        }
    }

    #[test]
    fn test_pi_sums_to_one() {
        let n = 10;
        let t = compute_time_intervals(n, 15.0, 0.1);
        let lambdas = vec![1.0; n + 1];
        let disc = compute_helpers(n, &t, &lambdas);
        let (pi_k, _, _) = compute_stationary(&disc, &lambdas, 0.001);
        let sum: f64 = pi_k.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sigma_sums_to_one() {
        let n = 10;
        let t = compute_time_intervals(n, 15.0, 0.1);
        let lambdas = vec![1.0; n + 1];
        let disc = compute_helpers(n, &t, &lambdas);
        let (_, sigma_k, _) = compute_stationary(&disc, &lambdas, 0.001);
        let sum: f64 = sigma_k.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_transition_rows_sum_to_one() {
        let n = 10;
        let theta = 0.001;
        let rho = theta / 5.0;
        let lambdas = vec![1.0; n + 1];
        let (transitions, _, _, _, _) =
            build_psmc_hmm_parts(n, 15.0, theta, rho, &lambdas, 0.1);
        for k in 0..n + 1 {
            let row_sum: f64 = transitions.row(k).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-6,
                "row {k} sums to {row_sum}"
            );
        }
    }

    #[test]
    fn test_initial_sums_to_one() {
        let n = 10;
        let theta = 0.001;
        let rho = theta / 5.0;
        let lambdas = vec![1.0; n + 1];
        let (_, _, initial, _, _) = build_psmc_hmm_parts(n, 15.0, theta, rho, &lambdas, 0.1);
        let sum: f64 = initial.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_emissions_sum_to_one() {
        let n = 10;
        let theta = 0.001;
        let rho = theta / 5.0;
        let lambdas = vec![1.0; n + 1];
        let (_, emissions, _, _, _) = build_psmc_hmm_parts(n, 15.0, theta, rho, &lambdas, 0.1);
        for k in 0..n + 1 {
            let sum = emissions[[0, k]] + emissions[[1, k]];
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_avg_times_positive_and_increasing() {
        let n = 10;
        let t = compute_time_intervals(n, 15.0, 0.1);
        let lambdas = vec![1.0; n + 1];
        let disc = compute_helpers(n, &t, &lambdas);
        let (pi_k, sigma_k, c_sigma) = compute_stationary(&disc, &lambdas, 0.001);
        let avg_t = compute_avg_times(&disc, &lambdas, &pi_k, &sigma_k, c_sigma, 0.001);
        for k in 0..n + 1 {
            assert!(avg_t[k] >= 0.0);
        }
        for k in 0..n {
            assert!(avg_t[k] <= avg_t[k + 1] + 1e-6);
        }
    }
}
