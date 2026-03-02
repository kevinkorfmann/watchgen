/// Copying-model primitives for the Li & Stephens HMM.
///
/// The Li & Stephens model (2003) treats a query haplotype as an imperfect
/// mosaic of n reference haplotypes. At each site the query "copies" from one
/// reference; between sites, recombination may switch the source; at each
/// site, mutation may change the copied allele.

/// Uniform initial distribution over n reference haplotypes.
pub fn initial_distribution(n: usize) -> Vec<f64> {
    vec![1.0 / n as f64; n]
}

/// Compute per-site recombination probabilities.
///
/// `rho[i]` is the population-scaled recombination rate at site i.
/// Returns `r` where `r[0] = 0` and `r[i] = 1 - exp(-rho[i] / n)`.
pub fn compute_recombination_probs(rho: &[f64], n: usize) -> Vec<f64> {
    let mut r = Vec::with_capacity(rho.len());
    r.push(0.0);
    for &rho_i in &rho[1..] {
        r.push(1.0 - (-rho_i / n as f64).exp());
    }
    r
}

/// Emission probability for one site.
///
/// Returns `1 - mu` if alleles match, `mu / (num_alleles - 1)` otherwise.
pub fn emission_prob(query: i8, reference: i8, mu: f64, num_alleles: usize) -> f64 {
    const NONCOPY: i8 = -2;
    const MISSING: i8 = -1;

    if reference == NONCOPY {
        return 0.0;
    }
    if query == MISSING {
        return 1.0;
    }
    if query == reference {
        1.0 - mu
    } else if num_alleles <= 1 {
        0.0
    } else {
        mu / (num_alleles - 1) as f64
    }
}

/// Estimate mutation probability from the number of haplotypes (Li & Stephens A2-A3).
pub fn estimate_mutation_probability(n: usize) -> f64 {
    assert!(n >= 3, "Need at least 3 haplotypes");
    let theta_tilde: f64 = 1.0 / (1..n).map(|k| 1.0 / k as f64).sum::<f64>();
    0.5 * theta_tilde / (n as f64 + theta_tilde)
}

/// O(n) forward step using Li-Stephens structure.
///
/// Instead of full O(n²) matrix-vector multiply, we exploit the
/// rank-1-plus-diagonal structure of the transition matrix.
///
/// For each state j: alpha_new[j] = (alpha_prev[j] * (1 - r) + S * r/n) * e[j]
/// where S = sum(alpha_prev).
pub fn forward_step_fast(
    alpha_prev: &[f64],
    r: f64,
    r_n: f64,
    emission: &[f64],
    n: usize,
) -> Vec<f64> {
    let sum_prev: f64 = alpha_prev.iter().sum();
    let mut alpha = Vec::with_capacity(n);
    for j in 0..n {
        let v = alpha_prev[j] * (1.0 - r) + sum_prev * r_n;
        alpha.push(v * emission[j]);
    }
    alpha
}

/// Build haploid emission matrix: (m, 2) where col 0 = mismatch, col 1 = match.
pub fn emission_matrix_haploid(mu: f64, num_sites: usize, num_alleles: &[usize]) -> Vec<[f64; 2]> {
    let mut e = Vec::with_capacity(num_sites);
    for i in 0..num_sites {
        if num_alleles[i] <= 1 {
            e.push([0.0, 1.0]);
        } else {
            e.push([mu / (num_alleles[i] - 1) as f64, 1.0 - mu]);
        }
    }
    e
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_distribution_sums_to_one() {
        let pi = initial_distribution(10);
        let sum: f64 = pi.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_recombination_probs_first_is_zero() {
        let rho = vec![0.04; 10];
        let r = compute_recombination_probs(&rho, 100);
        assert_eq!(r[0], 0.0);
        assert!(r[1] > 0.0);
    }

    #[test]
    fn test_emission_match_mismatch() {
        let mu = 0.01;
        assert!((emission_prob(0, 0, mu, 2) - 0.99).abs() < 1e-12);
        assert!((emission_prob(0, 1, mu, 2) - 0.01).abs() < 1e-12);
    }

    #[test]
    fn test_emission_noncopy_missing() {
        assert_eq!(emission_prob(0, -2, 0.01, 2), 0.0); // NONCOPY
        assert_eq!(emission_prob(-1, 0, 0.01, 2), 1.0); // MISSING
    }

    #[test]
    fn test_estimate_mutation_probability() {
        let mu = estimate_mutation_probability(100);
        assert!(mu > 0.0 && mu < 0.1);
    }

    #[test]
    fn test_forward_step_fast_preserves_mass() {
        let n = 5;
        let alpha = vec![0.2; n];
        let r = 0.1;
        let emission = vec![1.0; n]; // identity emission
        let result = forward_step_fast(&alpha, r, r / n as f64, &emission, n);
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_emission_matrix_haploid() {
        let num_alleles = vec![2, 2, 1, 2];
        let e = emission_matrix_haploid(0.01, 4, &num_alleles);
        assert!((e[0][0] - 0.01).abs() < 1e-12);
        assert!((e[0][1] - 0.99).abs() < 1e-12);
        assert_eq!(e[2][0], 0.0); // mono-allelic
        assert_eq!(e[2][1], 1.0);
    }
}
