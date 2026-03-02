use crate::discretize::build_psmc_hmm_parts;
use crate::hmm::{compute_expected_counts, PsmcHmm};

/// One EM iteration for PSMC.
///
/// M-step uses coordinate descent: each parameter (theta, rho, lambda_0, ...,
/// lambda_{n_free-1}) is optimized one-at-a-time via 1D golden section search
/// on the full Q-function. Multiple sweeps ensure convergence.
///
/// - t_max held fixed (matching the original Li & Durbin PSMC)
/// - No initial distribution term in Q (matching the original)
/// - All parameters optimized in log-space for smoothness
///
/// Returns `(new_hmm, log_likelihood)`.
pub fn psmc_em_step(hmm: &PsmcHmm, seq: &[u8], par_map: &[usize]) -> (PsmcHmm, f64) {
    let n_states = hmm.n_states;
    let n = hmm.n;

    let counts = compute_expected_counts(hmm, seq);
    let ll = counts.log_likelihood;

    let n_free = par_map.iter().copied().max().unwrap_or(0) + 1;
    let t_max = hmm.t_max;
    let alpha_param = hmm.alpha_param;

    let xi_sum = counts.xi_sum;
    let emission_counts = counts.emission_counts;

    // Parameter vector: [log_theta, log_rho, log_lambda_0, ..., log_lambda_{n_free-1}]
    let n_params = 2 + n_free;
    let mut params = vec![0.0f64; n_params];
    params[0] = hmm.theta.ln();
    params[1] = hmm.rho.ln();
    for k in 0..n_free {
        let idx = par_map.iter().position(|&p| p == k).unwrap_or(0);
        params[2 + k] = hmm.lambdas[idx].ln();
    }

    // Evaluate Q-function for a given parameter vector
    let eval_q = |p: &[f64]| -> f64 {
        let theta = p[0].exp();
        let rho = p[1].exp();
        let free_lambdas: Vec<f64> = p[2..].iter().map(|x| x.exp()).collect();

        if !theta.is_finite()
            || !rho.is_finite()
            || theta > 10.0
            || rho > 10.0
            || theta < 1e-10
            || rho < 1e-10
        {
            return -1e30;
        }
        if free_lambdas
            .iter()
            .any(|v| !v.is_finite() || *v > 1e4 || *v < 1e-4)
        {
            return -1e30;
        }

        let full_lambdas: Vec<f64> = (0..n_states).map(|k| free_lambdas[par_map[k]]).collect();

        let parts = std::panic::catch_unwind(|| {
            build_psmc_hmm_parts(n, t_max, theta, rho, &full_lambdas, alpha_param)
        });
        let (transitions, emissions, _, _, _) = match parts {
            Ok(p) => p,
            Err(_) => return -1e30,
        };
        if transitions.iter().any(|v| v.is_nan()) || emissions.iter().any(|v| v.is_nan()) {
            return -1e30;
        }

        let mut q_val = 0.0;

        // Transition term
        for k in 0..n_states {
            for l in 0..n_states {
                if transitions[[k, l]] > 0.0 && xi_sum[[k, l]] > 0.0 {
                    q_val += xi_sum[[k, l]] * (transitions[[k, l]] + 1e-300).ln();
                }
            }
        }

        // Emission term
        for b in 0..2usize {
            for k in 0..n_states {
                if emissions[[b, k]] > 0.0 && emission_counts[[b, k]] > 0.0 {
                    q_val += emission_counts[[b, k]] * (emissions[[b, k]] + 1e-300).ln();
                }
            }
        }

        q_val
    };

    // Coordinate descent: optimize each parameter via 1D golden section.
    // Safe accept: only update if Q actually improves (handles non-unimodal Q).
    let n_sweeps = 3;
    let mut q_current = eval_q(&params);

    for _ in 0..n_sweeps {
        for i in 0..n_params {
            let current = params[i];
            // Search range in log-space: ±1.5 for theta/rho, ±2.0 for lambdas
            let radius = if i < 2 { 1.5 } else { 2.0 };
            let candidate = golden_section_max(
                |v| {
                    let mut trial = params.clone();
                    trial[i] = v;
                    eval_q(&trial)
                },
                current - radius,
                current + radius,
                1e-6,
                100,
            );
            let mut trial = params.clone();
            trial[i] = candidate;
            let q_candidate = eval_q(&trial);
            if q_candidate > q_current {
                params[i] = candidate;
                q_current = q_candidate;
            }
            // Otherwise keep current value — never degrade Q
        }
    }

    let new_theta = params[0].exp();
    let new_rho = params[1].exp();
    let new_free_lambdas: Vec<f64> = params[2..].iter().map(|x| x.exp()).collect();
    let new_full_lambdas: Vec<f64> = (0..n_states)
        .map(|k| new_free_lambdas[par_map[k]])
        .collect();

    let new_hmm = PsmcHmm::new(n, new_theta, new_rho, &new_full_lambdas, t_max, alpha_param);

    (new_hmm, ll)
}

/// Golden section search to MAXIMIZE f on [a, b].
fn golden_section_max<F>(f: F, mut a: f64, mut b: f64, tol: f64, max_iter: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    let phi = (5.0_f64.sqrt() - 1.0) / 2.0;

    let mut c = b - phi * (b - a);
    let mut d = a + phi * (b - a);
    let mut fc = f(c);
    let mut fd = f(d);

    for _ in 0..max_iter {
        if (b - a).abs() < tol {
            break;
        }
        if fc < fd {
            a = c;
            c = d;
            fc = fd;
            d = a + phi * (b - a);
            fd = f(d);
        } else {
            b = d;
            d = c;
            fd = fc;
            c = b - phi * (b - a);
            fc = f(c);
        }
    }

    (a + b) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_golden_section_finds_maximum() {
        let x = golden_section_max(|x| -(x - 3.0) * (x - 3.0) + 10.0, 0.0, 6.0, 1e-8, 100);
        assert!((x - 3.0).abs() < 1e-6, "x={x}, expected 3.0");
    }

    #[test]
    fn test_golden_section_at_boundary() {
        // Maximum is at x=0 (left boundary), f(x) = -x^2
        let x = golden_section_max(|x| -x * x, 0.0, 5.0, 1e-8, 100);
        assert!(x.abs() < 1e-4, "x={x}, expected near 0.0");
    }
}
