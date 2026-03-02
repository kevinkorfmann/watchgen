/// Continuous HMM for SMC++: transition matrix, emission probabilities, forward algorithm.

use crate::ode;
use crate::rates;
use ndarray::Array2;

/// Emission probability for the distinguished lineage.
///
/// `genotype`: 0 or 1 (het/hom at binary site).
/// `t`: coalescence time.
/// `theta`: scaled mutation rate per bin.
pub fn emission_probability(genotype: usize, t: f64, theta: f64) -> f64 {
    match genotype {
        0 => (-theta * t).exp(),
        1 => 1.0 - (-theta * t).exp(),
        _ => (1.0 - (-theta * t).exp()).powi(2),
    }
}

/// Build the SMC++ transition matrix.
///
/// Uses ODE solutions to compute h(t) at interval midpoints,
/// then constructs transition probabilities incorporating recombination.
pub fn compute_transition_matrix(
    time_breaks: &[f64],
    lambdas: &[f64],
    rho: f64,
    n_undist: usize,
) -> Array2<f64> {
    let k = time_breaks.len() - 1;
    let (eig, v, v_inv) = ode::eigendecompose_rate_matrix(n_undist);

    // Compute h(t) at midpoints
    let mut h = vec![0.0; k];
    let mut p_current = vec![0.0; n_undist];
    p_current[n_undist - 1] = 1.0;

    for i in 0..k {
        let dt = time_breaks[i + 1] - time_breaks[i];
        let lam = lambdas[i];

        // Midpoint ODE solution
        let m_mid = ode::fast_matrix_exp(&eig, &v, &v_inv, dt / 2.0, lam);
        let mut p_mid = vec![0.0; n_undist];
        for r in 0..n_undist {
            for c in 0..n_undist {
                p_mid[r] += m_mid[[r, c]] * p_current[c];
            }
        }
        h[i] = rates::compute_h(&p_mid, lam);

        // Full step
        let m_full = ode::fast_matrix_exp(&eig, &v, &v_inv, dt, lam);
        let mut p_new = vec![0.0; n_undist];
        for r in 0..n_undist {
            for c in 0..n_undist {
                p_new[r] += m_full[[r, c]] * p_current[c];
            }
        }
        p_current = p_new;
    }

    // Build transition matrix
    let mut p_mat = Array2::<f64>::zeros((k, k));
    for src in 0..k {
        let t_mid = (time_breaks[src] + time_breaks[src + 1]) / 2.0;
        let r_k = 1.0 - (-rho * t_mid).exp();

        // No recombination: stay
        p_mat[[src, src]] += 1.0 - r_k;

        // With recombination: re-coalescence density
        for dst in 0..k {
            let dt_l = time_breaks[dst + 1] - time_breaks[dst];
            let cum_h: f64 = (0..dst)
                .map(|m| h[m] * (time_breaks[m + 1] - time_breaks[m]))
                .sum();
            let q_kl = h[dst] * (-cum_h).exp() * dt_l;
            p_mat[[src, dst]] += r_k * q_kl;
        }
    }

    // Normalize rows
    for src in 0..k {
        let row_sum: f64 = p_mat.row(src).sum();
        if row_sum > 0.0 {
            for dst in 0..k {
                p_mat[[src, dst]] /= row_sum;
            }
        }
    }

    p_mat
}

/// Compute composite log-likelihood for SMC++ via forward algorithm.
///
/// `data[i]` is the observation sequence for the i-th distinguished sample.
pub fn composite_log_likelihood(
    data: &[Vec<u8>],
    time_breaks: &[f64],
    lambdas: &[f64],
    theta: f64,
    rho: f64,
) -> f64 {
    let n_samples = data.len();
    let n_undist = 2 * n_samples - 1;
    let k = time_breaks.len() - 1;
    let mut total_ll = 0.0;

    let p_mat = compute_transition_matrix(time_breaks, lambdas, rho, n_undist);
    let pi = vec![1.0 / k as f64; k]; // uniform initial

    for obs in data {
        let len = obs.len();
        let mut alpha = vec![0.0; k];

        // Initialize
        for j in 0..k {
            let t_mid = (time_breaks[j] + time_breaks[j + 1]) / 2.0;
            alpha[j] = pi[j] * emission_probability(obs[0] as usize, t_mid, theta);
        }

        let mut scale = vec![0.0_f64; len];
        scale[0] = alpha.iter().sum();
        if scale[0] > 0.0 {
            for j in 0..k {
                alpha[j] /= scale[0];
            }
        }

        // Forward recursion
        for a in 1..len {
            let mut alpha_new = vec![0.0; k];
            for l in 0..k {
                let mut s = 0.0;
                for j in 0..k {
                    s += alpha[j] * p_mat[[j, l]];
                }
                let t_mid = (time_breaks[l] + time_breaks[l + 1]) / 2.0;
                alpha_new[l] = s * emission_probability(obs[a] as usize, t_mid, theta);
            }
            scale[a] = alpha_new.iter().sum();
            if scale[a] > 0.0 {
                for l in 0..k {
                    alpha_new[l] /= scale[a];
                }
            }
            alpha = alpha_new;
        }

        total_ll += scale
            .iter()
            .filter(|&&s| s > 0.0)
            .map(|s| s.ln())
            .sum::<f64>();
    }

    total_ll
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emission_sums_binary() {
        let t = 0.5;
        let theta = 0.001;
        let sum = emission_probability(0, t, theta) + emission_probability(1, t, theta);
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_transition_matrix_rows_sum_to_one() {
        let time_breaks: Vec<f64> = (0..=10).map(|i| i as f64 * 0.5).collect();
        let lambdas = vec![1.0; 10];
        let p = compute_transition_matrix(&time_breaks, &lambdas, 0.001, 5);
        for i in 0..10 {
            let row_sum: f64 = p.row(i).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-6,
                "row {i} sum = {row_sum}"
            );
        }
    }

    #[test]
    fn test_composite_ll_finite() {
        let time_breaks: Vec<f64> = (0..=5).map(|i| i as f64).collect();
        let lambdas = vec![1.0; 5];
        let data = vec![vec![0u8, 0, 1, 0, 0, 1, 0, 0, 0, 1]];
        let ll =
            composite_log_likelihood(&data, &time_breaks, &lambdas, 0.001, 0.0002);
        assert!(ll.is_finite());
        assert!(ll < 0.0);
    }

    #[test]
    fn test_ll_changes_with_lambdas() {
        let time_breaks: Vec<f64> = (0..=5).map(|i| i as f64).collect();
        let data = vec![vec![0u8, 0, 1, 0, 0, 1, 0, 0, 0, 1]];

        let ll1 = composite_log_likelihood(
            &data,
            &time_breaks,
            &[1.0, 1.0, 1.0, 1.0, 1.0],
            0.001,
            0.0002,
        );
        let ll2 = composite_log_likelihood(
            &data,
            &time_breaks,
            &[0.5, 0.5, 0.5, 0.5, 0.5],
            0.001,
            0.0002,
        );
        assert!((ll1 - ll2).abs() > 1e-10, "LL should differ for different lambdas");
    }
}
