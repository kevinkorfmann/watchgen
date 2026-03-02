/// Time sampling for SINGER: determines when a lineage joins a branch.
///
/// Uses PSMC transition density as a component and partitions branches
/// into sub-intervals for fine-grained time resolution.

/// Partition a branch [x, y] into d sub-intervals with equal joining probability.
///
/// Returns d+1 boundary times.
pub fn partition_branch(x: f64, y: f64, d: usize) -> Vec<f64> {
    use crate::branch_sampling::f_bar_approx;

    let n_large = 1000; // use large-n approximation
    let fb_x = f_bar_approx(x, n_large);
    let fb_y = f_bar_approx(y, n_large);
    let total_prob = fb_x - fb_y;

    if total_prob <= 0.0 || d == 0 {
        return vec![x, y];
    }

    let prob_per_bin = total_prob / d as f64;
    let mut boundaries = Vec::with_capacity(d + 1);
    boundaries.push(x);

    for i in 1..d {
        let target_fb = fb_x - i as f64 * prob_per_bin;
        // Binary search for t such that F_bar(t) = target
        let mut lo = x;
        let mut hi = y;
        for _ in 0..100 {
            let mid = (lo + hi) / 2.0;
            let fb_mid = f_bar_approx(mid, n_large);
            if fb_mid > target_fb {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        boundaries.push((lo + hi) / 2.0);
    }

    boundaries.push(y);
    boundaries
}

/// Compute representative time for each sub-interval.
///
/// Uses the center-of-mass in exponential time.
pub fn representative_times(boundaries: &[f64]) -> Vec<f64> {
    let d = boundaries.len() - 1;
    let mut taus = Vec::with_capacity(d);
    for i in 0..d {
        let lo = boundaries[i];
        let hi = boundaries[i + 1];
        // Center-of-mass under f(t) ∝ e^{-t} on [lo, hi]:
        // E[t] = ((lo+1)·e^{-lo} - (hi+1)·e^{-hi}) / (e^{-lo} - e^{-hi})
        let tau = if (hi - lo).abs() < 1e-15 {
            lo
        } else {
            let exp_lo = (-lo).exp();
            let exp_hi = (-hi).exp();
            let denom = exp_lo - exp_hi;
            if denom.abs() < 1e-30 {
                (lo + hi) / 2.0
            } else {
                ((lo + 1.0) * exp_lo - (hi + 1.0) * exp_hi) / denom
            }
        };
        taus.push(tau);
    }
    taus
}

/// PSMC transition density q_rho(t | s).
///
/// The probability density that the next coalescence time is t,
/// given that the previous was s, with recombination rate rho.
pub fn psmc_transition_density(t: f64, s: f64, rho: f64) -> f64 {
    if rho <= 0.0 {
        // No recombination: delta at t=s
        return 0.0;
    }

    let p_no_recomb = (-rho * s).exp();

    if (t - s).abs() < 1e-15 {
        // Point mass at t=s (no recombination)
        return p_no_recomb; // This is the weight, not density
    }

    // Continuous part: density of re-coalescence time
    let p_recomb = 1.0 - p_no_recomb;
    // Under standard coalescent, density of new coalescence at time t is exp(-t)
    p_recomb * (-t).exp()
}

/// PSMC transition CDF: P(T' <= t | S = s).
pub fn psmc_transition_cdf(t: f64, s: f64, rho: f64) -> f64 {
    if t < 0.0 {
        return 0.0;
    }

    let p_no_recomb = (-rho * s).exp();

    // Point mass at s contributes if t >= s
    let point_mass = if t >= s { p_no_recomb } else { 0.0 };

    // Continuous part: P(T' <= t | recomb) = 1 - exp(-t)
    let p_recomb = 1.0 - p_no_recomb;
    let continuous = p_recomb * (1.0 - (-t).exp());

    point_mass + continuous
}

/// Build time transition matrix Q[d_prev, d_next].
///
/// Uses PSMC transition CDF to compute transition probabilities
/// between time sub-intervals.
pub fn time_transition_matrix(
    _boundaries_prev: &[f64],
    taus_prev: &[f64],
    boundaries_next: &[f64],
    rho: f64,
) -> Vec<Vec<f64>> {
    let d_prev = taus_prev.len();
    let d_next = boundaries_next.len() - 1;
    let mut q = vec![vec![0.0; d_next]; d_prev];

    for i in 0..d_prev {
        let s = taus_prev[i];
        for j in 0..d_next {
            let cdf_hi = psmc_transition_cdf(boundaries_next[j + 1], s, rho);
            let cdf_lo = psmc_transition_cdf(boundaries_next[j], s, rho);
            q[i][j] = (cdf_hi - cdf_lo).max(0.0);
        }
        // Normalize row
        let row_sum: f64 = q[i].iter().sum();
        if row_sum > 0.0 {
            for j in 0..d_next {
                q[i][j] /= row_sum;
            }
        }
    }

    q
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_branch_boundaries_increasing() {
        let bounds = partition_branch(0.5, 3.0, 10);
        assert_eq!(bounds.len(), 11);
        assert!((bounds[0] - 0.5).abs() < 1e-10);
        assert!((bounds[10] - 3.0).abs() < 1e-10);
        for i in 0..10 {
            assert!(bounds[i] < bounds[i + 1]);
        }
    }

    #[test]
    fn test_representative_times_in_intervals() {
        let bounds = partition_branch(0.0, 5.0, 5);
        let taus = representative_times(&bounds);
        assert_eq!(taus.len(), 5);
        for i in 0..5 {
            assert!(
                taus[i] >= bounds[i] && taus[i] <= bounds[i + 1],
                "tau[{i}]={} not in [{}, {}]",
                taus[i],
                bounds[i],
                bounds[i + 1]
            );
        }
    }

    #[test]
    fn test_psmc_transition_cdf_monotone() {
        let s = 1.0;
        let rho = 0.01;
        let mut prev = 0.0;
        for t in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0] {
            let cdf = psmc_transition_cdf(t, s, rho);
            assert!(cdf >= prev - 1e-10);
            prev = cdf;
        }
    }

    #[test]
    fn test_psmc_transition_cdf_bounds() {
        let s = 1.0;
        let rho = 0.01;
        assert!(psmc_transition_cdf(0.0, s, rho) >= 0.0);
        assert!(psmc_transition_cdf(100.0, s, rho) <= 1.0 + 1e-10);
    }

    #[test]
    fn test_time_transition_matrix_rows_sum_to_one() {
        let bounds = vec![0.0, 1.0, 2.0, 3.0, 5.0];
        let taus = representative_times(&bounds);
        let q = time_transition_matrix(&bounds, &taus, &bounds, 0.01);
        for i in 0..taus.len() {
            let row_sum: f64 = q[i].iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 0.01,
                "row {i} sum = {row_sum}"
            );
        }
    }
}
