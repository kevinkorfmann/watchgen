use serde::Serialize;

use crate::coalescent::estimate_theta_initial;
use crate::em::psmc_em_step;
use crate::error::PsmcError;
use crate::hmm::{compute_expected_counts, PsmcHmm};
use crate::pattern::parse_pattern;

/// Configuration for PSMC inference.
#[derive(Debug, Clone, Serialize)]
pub struct PsmcConfig {
    pub n: usize,
    pub t_max: f64,
    pub theta_rho_ratio: f64,
    pub pattern: String,
    pub n_iters: usize,
    pub alpha_param: f64,
}

impl Default for PsmcConfig {
    fn default() -> Self {
        Self {
            n: 63,
            t_max: 15.0,
            theta_rho_ratio: 5.0,
            pattern: "4+25*2+4+6".to_string(),
            n_iters: 25,
            alpha_param: 0.1,
        }
    }
}

/// Result from a single EM iteration.
#[derive(Debug, Clone, Serialize)]
pub struct IterationResult {
    pub iteration: usize,
    pub log_likelihood: f64,
    pub theta: f64,
    pub rho: f64,
    pub lambdas: Vec<f64>,
}

/// Run the full PSMC inference pipeline.
pub fn psmc_inference(
    seq: &[u8],
    config: &PsmcConfig,
) -> Result<Vec<IterationResult>, PsmcError> {
    if seq.is_empty() {
        return Err(PsmcError::EmptySequence);
    }

    let n = config.n;
    let n_states = n + 1;

    let (par_map, n_free, n_intervals) = parse_pattern(&config.pattern)?;
    if n_intervals != n_states {
        return Err(PsmcError::PatternMismatch {
            expected: n_states,
            got: n_intervals,
        });
    }

    let theta = estimate_theta_initial(seq);
    if theta <= 0.0 || !theta.is_finite() {
        return Err(PsmcError::Numerical(
            "could not estimate initial theta".into(),
        ));
    }
    let rho = theta / config.theta_rho_ratio;

    let free_lambdas = vec![1.0; n_free];
    let full_lambdas: Vec<f64> = (0..n_states).map(|k| free_lambdas[par_map[k]]).collect();

    let mut hmm = PsmcHmm::new(n, theta, rho, &full_lambdas, config.t_max, config.alpha_param);

    let mut results = Vec::with_capacity(config.n_iters);

    eprintln!(
        "PSMC inference: {} bins, {} intervals, {} free params",
        seq.len(),
        n_states,
        n_free
    );
    eprintln!("Initial theta={:.6}, rho={:.6}", theta, rho);

    for iteration in 0..config.n_iters {
        let counts = compute_expected_counts(&hmm, seq);
        let ll = counts.log_likelihood;

        results.push(IterationResult {
            iteration,
            log_likelihood: ll,
            theta: hmm.theta,
            rho: hmm.rho,
            lambdas: hmm.lambdas.clone(),
        });

        eprintln!(
            "  Iteration {}: LL = {:.2}, theta = {:.6}, rho = {:.6}",
            iteration, ll, hmm.theta, hmm.rho
        );

        let (new_hmm, _) = psmc_em_step(&hmm, seq, &par_map);
        hmm = new_hmm;
    }

    Ok(results)
}
