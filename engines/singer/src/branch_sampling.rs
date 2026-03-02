/// Branch sampling for SINGER: determines which branch a new lineage joins.
///
/// Uses the coalescent prior to compute joining probabilities for each branch
/// in a marginal tree, then models emission probabilities given the topology.

/// Deterministic approximation for number of lineages at time t.
///
/// Replaces the stochastic step-function with a smooth curve:
/// λ(t) = n / (n + (1 - n)·exp(-t/2))
pub fn lambda_approx(t: f64, n: usize) -> f64 {
    let nf = n as f64;
    nf / (nf + (1.0 - nf) * (-t / 2.0).exp())
}

/// Exceedance probability P(T > t) under the smooth approximation.
///
/// F̄(t) = exp(-t) / (n + (1-n)·exp(-t/2))²
pub fn f_bar_approx(t: f64, n: usize) -> f64 {
    let nf = n as f64;
    let denom = nf + (1.0 - nf) * (-t / 2.0).exp();
    (-t).exp() / (denom * denom)
}

/// Density of joining time: f(t) = n·exp(-t) / (n + (1-n)·exp(-t/2))³
pub fn f_density_approx(t: f64, n: usize) -> f64 {
    let nf = n as f64;
    let denom = nf + (1.0 - nf) * (-t / 2.0).exp();
    nf * (-t).exp() / (denom * denom * denom)
}

/// Fast joining probability for a branch spanning [x, y].
///
/// P(join branch) = F̄(x) - F̄(y)
pub fn joining_prob_approx(x: f64, y: f64, n: usize) -> f64 {
    f_bar_approx(x, n) - f_bar_approx(y, n)
}

/// Inverse of lambda_approx: find time t given lineage count ℓ.
///
/// Solves λ(t) = ℓ for t.
pub fn lambda_inverse(ell: f64, n: usize) -> f64 {
    let nf = n as f64;
    -2.0 * (nf * (ell - 1.0) / (ell * (nf - 1.0))).ln()
}

/// Compute representative joining time for a branch [x, y].
///
/// Uses the geometric mean of lambda at x and y.
pub fn representative_time(x: f64, y: f64, n: usize) -> f64 {
    let lam_x = lambda_approx(x, n);
    let lam_y = lambda_approx(y, n);
    let lam_mid = (lam_x * lam_y).sqrt();
    lambda_inverse(lam_mid, n)
}

/// A branch in the HMM state space.
#[derive(Debug, Clone)]
pub struct BranchState {
    pub child: u32,
    pub parent: u32,
    pub lower_time: f64,
    pub upper_time: f64,
    pub is_partial: bool,
}

impl BranchState {
    pub fn length(&self) -> f64 {
        self.upper_time - self.lower_time
    }
}

/// Emission probability for a single site given joining topology.
///
/// Computes P(allele_new | allele_at_join, tau, branch bounds, theta)
/// by multiplying mutation probabilities across sub-branches.
pub fn emission_probability(
    allele_new: u8,
    allele_at_join: u8,
    tau: f64,
    branch_lower: f64,
    branch_upper: f64,
    theta: f64,
) -> f64 {
    let p_no_mut_below = (-theta * (tau - branch_lower)).exp();
    let p_no_mut_above = (-theta * (branch_upper - tau)).exp();
    let p_no_mut_new = (-theta * tau).exp();

    // P(surviving allele = allele_at_join) from existing branch
    let p_existing_survives = p_no_mut_below * p_no_mut_above;
    // P(new lineage allele = allele_new) from new branch
    let p_new_match = if allele_new == allele_at_join {
        p_no_mut_new
    } else {
        1.0 - p_no_mut_new
    };

    p_existing_survives * p_new_match + (1.0 - p_existing_survives) * (1.0 - p_new_match)
}

/// Branch transition probability accounting for recombination.
///
/// `tau_i`, `tau_j`: representative times for source/dest branches.
/// `p_j`: joining probability for destination branch.
/// `rho`: recombination rate per site.
/// `same_branch`: true if source and destination are the same branch.
pub fn branch_transition_prob(
    _tau_i: f64,
    _tau_j: f64,
    p_j: f64,
    rho: f64,
    same_branch: bool,
) -> f64 {
    let p_recomb = 1.0 - (-rho).exp();
    if same_branch {
        (1.0 - p_recomb) + p_recomb * p_j
    } else {
        p_recomb * p_j
    }
}

/// Build state space, pruning low-probability partial branches.
pub fn build_state_space(
    full_branches: &[BranchState],
    partial_branches: &[BranchState],
    forward_probs: Option<&[f64]>,
    epsilon: f64,
) -> Vec<BranchState> {
    let mut states: Vec<BranchState> = full_branches.to_vec();

    if let Some(probs) = forward_probs {
        for (i, branch) in partial_branches.iter().enumerate() {
            if i < probs.len() && probs[i] >= epsilon {
                states.push(branch.clone());
            }
        }
    } else {
        states.extend(partial_branches.iter().cloned());
    }

    states
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lambda_approx_at_zero() {
        // At t=0, lambda(0) = n (all lineages present)
        let n = 10;
        assert!((lambda_approx(0.0, n) - n as f64).abs() < 1e-10);
    }

    #[test]
    fn test_lambda_approx_decreasing() {
        let n = 20;
        let l0 = lambda_approx(0.0, n);
        let l1 = lambda_approx(1.0, n);
        let l2 = lambda_approx(2.0, n);
        assert!(l0 > l1);
        assert!(l1 > l2);
    }

    #[test]
    fn test_f_bar_at_zero() {
        // F_bar(0) = 1/(n^2) * n = 1/n... actually let's compute
        let n = 10;
        let fb = f_bar_approx(0.0, n);
        // F_bar(0) = exp(0)/(n + (1-n)*1)^2 = 1/1 = 1.0
        assert!((fb - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_f_bar_decreasing() {
        let n = 10;
        assert!(f_bar_approx(0.0, n) > f_bar_approx(1.0, n));
        assert!(f_bar_approx(1.0, n) > f_bar_approx(3.0, n));
    }

    #[test]
    fn test_joining_prob_positive() {
        let p = joining_prob_approx(0.5, 2.0, 10);
        assert!(p > 0.0);
        assert!(p < 1.0);
    }

    #[test]
    fn test_joining_probs_sum_approx_one() {
        // Total joining prob from 0 to infinity should be ~1
        let n = 10;
        let p = f_bar_approx(0.0, n) - f_bar_approx(100.0, n);
        assert!((p - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_lambda_inverse_roundtrip() {
        let n = 10;
        let t = 1.5;
        let ell = lambda_approx(t, n);
        let t_back = lambda_inverse(ell, n);
        assert!((t - t_back).abs() < 1e-8);
    }

    #[test]
    fn test_representative_time_in_interval() {
        let n = 10;
        let tau = representative_time(0.5, 2.0, n);
        assert!(tau >= 0.5);
        assert!(tau <= 2.0);
    }

    #[test]
    fn test_emission_probability_range() {
        let e = emission_probability(0, 0, 0.5, 0.0, 1.0, 0.001);
        assert!(e >= 0.0 && e <= 1.0);
    }

    #[test]
    fn test_branch_transition_same_vs_different() {
        let p_j = 0.1;
        let rho = 0.01;
        let same = branch_transition_prob(1.0, 1.0, p_j, rho, true);
        let diff = branch_transition_prob(1.0, 2.0, p_j, rho, false);
        assert!(same > diff); // staying is more likely than switching
    }
}
