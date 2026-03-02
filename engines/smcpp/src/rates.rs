/// Coalescence rates for the SMC++ distinguished lineage model.

/// Expected time to first coalescence among n lineages in population of size N.
pub fn expected_first_coalescence(n: usize, big_n: f64) -> f64 {
    let rate = (n * (n - 1)) as f64 / (2.0 * big_n);
    1.0 / rate
}

/// Rate at which j undistinguished lineages coalesce among themselves: C(j,2)/lambda.
pub fn undistinguished_coalescence_rate(j: usize, lam: f64) -> f64 {
    (j * (j - 1)) as f64 / (2.0 * lam)
}

/// Rate at which the distinguished lineage coalesces with an undistinguished one: j/lambda.
pub fn distinguished_coalescence_rate(j: usize, lam: f64) -> f64 {
    j as f64 / lam
}

/// Emission probability for unphased diploid genotype.
///
/// `genotype`: 0, 1, or 2 (count of derived alleles).
/// `t`: coalescence time of the distinguished lineage.
/// `theta`: scaled mutation rate per bin.
pub fn emission_unphased(genotype: usize, t: f64, theta: f64) -> f64 {
    let p = 1.0 - (-theta * t).exp();
    match genotype {
        0 => (1.0 - p) * (1.0 - p),
        1 => 2.0 * p * (1.0 - p),
        _ => p * p,
    }
}

/// Effective coalescence rate h(t) of the distinguished lineage.
///
/// `p_j[j-1]` = P(J(t) = j) for j = 1, ..., n_undist.
pub fn compute_h(p_j: &[f64], lam: f64) -> f64 {
    let mut h = 0.0;
    for (idx, &pj) in p_j.iter().enumerate() {
        let j = idx + 1;
        h += j as f64 / lam * pj;
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expected_first_coalescence() {
        let t = expected_first_coalescence(2, 10000.0);
        assert!((t - 10000.0).abs() < 1.0);
    }

    #[test]
    fn test_coalescence_rates() {
        assert!((undistinguished_coalescence_rate(9, 1.0) - 36.0).abs() < 1e-10);
        assert!((distinguished_coalescence_rate(9, 1.0) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_emission_sums_to_one() {
        let t = 0.5;
        let theta = 0.001;
        let sum = emission_unphased(0, t, theta)
            + emission_unphased(1, t, theta)
            + emission_unphased(2, t, theta);
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_h() {
        // If all mass on j=5, h = 5/lam
        let mut p = vec![0.0; 9];
        p[4] = 1.0; // j=5
        assert!((compute_h(&p, 1.0) - 5.0).abs() < 1e-10);
    }
}
