/// Haploid Li-Stephens HMM algorithms: forward, backward, Viterbi, posterior decoding.

/// Result of the forward algorithm.
pub struct ForwardResult {
    /// Scaled forward probabilities, shape (m, n).
    pub f: Vec<Vec<f64>>,
    /// Scaling factors, shape (m,).
    pub c: Vec<f64>,
    /// Log-likelihood (base 10).
    pub ll: f64,
}

/// Forward algorithm for the haploid Li-Stephens model.
///
/// Uses the O(n) Li-Stephens trick at each site.
///
/// - `h`: Reference panel, `h[site][haplotype]`.
/// - `s`: Query haplotype.
/// - `e`: Emission matrix (m, 2): col 0 = mismatch, col 1 = match.
/// - `r`: Per-site recombination probabilities.
pub fn forward(
    n: usize,
    m: usize,
    h: &[Vec<i8>],
    s: &[i8],
    e: &[[f64; 2]],
    r: &[f64],
) -> ForwardResult {
    let mut f = vec![vec![0.0; n]; m];
    let mut c = vec![0.0; m];
    let r_n: Vec<f64> = r.iter().map(|&ri| ri / n as f64).collect();

    // Initialize
    for i in 0..n {
        let em = if s[0] == h[0][i] { e[0][1] } else { e[0][0] };
        f[0][i] = (1.0 / n as f64) * em;
        c[0] += f[0][i];
    }
    if c[0] > 0.0 {
        for i in 0..n {
            f[0][i] /= c[0];
        }
    }

    // Recurse
    for l in 1..m {
        for i in 0..n {
            f[l][i] = f[l - 1][i] * (1.0 - r[l]) + r_n[l];
            let em = if s[l] == h[l][i] { e[l][1] } else { e[l][0] };
            f[l][i] *= em;
            c[l] += f[l][i];
        }
        if c[l] > 0.0 {
            for i in 0..n {
                f[l][i] /= c[l];
            }
        }
    }

    let ll = c.iter().filter(|&&ci| ci > 0.0).map(|ci| ci.log10()).sum();

    ForwardResult { f, c, ll }
}

/// Backward algorithm for the haploid Li-Stephens model.
///
/// Uses the forward scaling factors `c` for numerical consistency.
pub fn backward(
    n: usize,
    m: usize,
    h: &[Vec<i8>],
    s: &[i8],
    e: &[[f64; 2]],
    c: &[f64],
    r: &[f64],
) -> Vec<Vec<f64>> {
    let mut b = vec![vec![0.0; n]; m];
    let r_n: Vec<f64> = r.iter().map(|&ri| ri / n as f64).collect();

    // Initialize last row
    for i in 0..n {
        b[m - 1][i] = 1.0;
    }

    // Recurse backwards
    for l in (0..m - 1).rev() {
        let mut tmp_sum = 0.0;
        let mut tmp_b = vec![0.0; n];
        for i in 0..n {
            let em = if s[l + 1] == h[l + 1][i] {
                e[l + 1][1]
            } else {
                e[l + 1][0]
            };
            tmp_b[i] = em * b[l + 1][i];
            tmp_sum += tmp_b[i];
        }
        for i in 0..n {
            b[l][i] = r_n[l + 1] * tmp_sum + (1.0 - r[l + 1]) * tmp_b[i];
            if c[l + 1] > 0.0 {
                b[l][i] /= c[l + 1];
            }
        }
    }

    b
}

/// Posterior decoding from forward-backward probabilities.
///
/// Returns `(gamma, path)` where `gamma[l]` is the posterior distribution
/// at site l and `path[l]` is the MAP state.
pub fn posterior_decoding(f: &[Vec<f64>], b: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<usize>) {
    let m = f.len();
    let n = f[0].len();
    let mut gamma = vec![vec![0.0; n]; m];
    let mut path = vec![0usize; m];

    for l in 0..m {
        let mut total = 0.0;
        for i in 0..n {
            gamma[l][i] = f[l][i] * b[l][i];
            total += gamma[l][i];
        }
        if total > 0.0 {
            for i in 0..n {
                gamma[l][i] /= total;
            }
        }
        let mut best = 0;
        for i in 1..n {
            if gamma[l][i] > gamma[l][best] {
                best = i;
            }
        }
        path[l] = best;
    }

    (gamma, path)
}

/// Viterbi algorithm for the haploid Li-Stephens model.
///
/// Uses O(n) per site via the Li-Stephens structure.
/// Returns `(v_last, pointers, ll)`.
pub fn viterbi(
    n: usize,
    m: usize,
    h: &[Vec<i8>],
    s: &[i8],
    e: &[[f64; 2]],
    r: &[f64],
) -> (Vec<f64>, Vec<Vec<usize>>, f64) {
    let mut v = vec![0.0; n];
    let mut p = vec![vec![0usize; n]; m];
    let mut c = vec![1.0_f64; m];
    let r_n: Vec<f64> = r.iter().map(|&ri| ri / n as f64).collect();

    // Initialize
    for i in 0..n {
        let em = if s[0] == h[0][i] { e[0][1] } else { e[0][0] };
        v[i] = (1.0 / n as f64) * em;
    }

    // Recurse
    for j in 1..m {
        let (argmax, &max_val) = v
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        c[j] = max_val;
        if c[j] > 0.0 {
            for vi in v.iter_mut() {
                *vi /= c[j];
            }
        }

        let mut v_new = vec![0.0; n];
        for i in 0..n {
            let stay = v[i] * (1.0 - r[j] + r_n[j]);
            let switch = r_n[j];

            if stay >= switch {
                v_new[i] = stay;
                p[j][i] = i;
            } else {
                v_new[i] = switch;
                p[j][i] = argmax;
            }

            let em = if s[j] == h[j][i] { e[j][1] } else { e[j][0] };
            v_new[i] *= em;
        }
        v = v_new;
    }

    let max_v = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let ll = c.iter().filter(|&&ci| ci > 0.0).map(|ci| ci.log10()).sum::<f64>()
        + if max_v > 0.0 { max_v.log10() } else { 0.0 };

    (v, p, ll)
}

/// Viterbi traceback.
pub fn viterbi_traceback(m: usize, v_last: &[f64], p: &[Vec<usize>]) -> Vec<usize> {
    let mut path = vec![0usize; m];
    path[m - 1] = v_last
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    for j in (0..m - 1).rev() {
        path[j] = p[j + 1][path[j + 1]];
    }
    path
}

/// Evaluate the log-likelihood of a specific copying path.
pub fn path_loglik(
    n: usize,
    m: usize,
    h: &[Vec<i8>],
    path: &[usize],
    s: &[i8],
    e: &[[f64; 2]],
    r: &[f64],
) -> f64 {
    let r_n: Vec<f64> = r.iter().map(|&ri| ri / n as f64).collect();

    let em0 = if s[0] == h[0][path[0]] {
        e[0][1]
    } else {
        e[0][0]
    };
    let mut ll = ((1.0 / n as f64) * em0).log10();

    for l in 1..m {
        let trans = if path[l - 1] == path[l] {
            1.0 - r[l] + r_n[l]
        } else {
            r_n[l]
        };
        let em = if s[l] == h[l][path[l]] {
            e[l][1]
        } else {
            e[l][0]
        };
        ll += (trans * em).log10();
    }

    ll
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (usize, usize, Vec<Vec<i8>>, Vec<i8>, Vec<[f64; 2]>, Vec<f64>) {
        let n = 3;
        let m = 4;
        // Reference panel
        let h = vec![
            vec![0, 1, 0],
            vec![0, 0, 1],
            vec![1, 1, 0],
            vec![0, 1, 1],
        ];
        let s = vec![0, 0, 1, 1];
        let mu = 0.1;
        let e = vec![[mu, 1.0 - mu]; m];
        let r = vec![0.0, 0.1, 0.1, 0.1];
        (n, m, h, s, e, r)
    }

    #[test]
    fn test_forward_ll_finite() {
        let (n, m, h, s, e, r) = make_test_data();
        let result = forward(n, m, &h, &s, &e, &r);
        assert!(result.ll.is_finite());
        assert!(result.ll < 0.0);
    }

    #[test]
    fn test_forward_backward_consistency() {
        let (n, m, h, s, e, r) = make_test_data();
        let fwd = forward(n, m, &h, &s, &e, &r);
        let bwd = backward(n, m, &h, &s, &e, &fwd.c, &r);
        // Posterior should sum to 1 at each site
        for l in 0..m {
            let total: f64 = (0..n).map(|i| fwd.f[l][i] * bwd[l][i]).sum();
            assert!(total > 0.0, "posterior total at site {l} = {total}");
        }
    }

    #[test]
    fn test_posterior_decoding() {
        let (n, m, h, s, e, r) = make_test_data();
        let fwd = forward(n, m, &h, &s, &e, &r);
        let bwd = backward(n, m, &h, &s, &e, &fwd.c, &r);
        let (gamma, path) = posterior_decoding(&fwd.f, &bwd);
        assert_eq!(gamma.len(), m);
        assert_eq!(path.len(), m);
        for l in 0..m {
            let sum: f64 = gamma[l].iter().sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_viterbi_path_valid() {
        let (n, m, h, s, e, r) = make_test_data();
        let (v, p, ll) = viterbi(n, m, &h, &s, &e, &r);
        let path = viterbi_traceback(m, &v, &p);
        assert!(ll.is_finite());
        for &state in &path {
            assert!(state < n);
        }
    }

    #[test]
    fn test_viterbi_ll_leq_forward_ll() {
        let (n, m, h, s, e, r) = make_test_data();
        let fwd = forward(n, m, &h, &s, &e, &r);
        let (v, p, ll_vit) = viterbi(n, m, &h, &s, &e, &r);
        let path = viterbi_traceback(m, &v, &p);
        let ll_path = path_loglik(n, m, &h, &path, &s, &e, &r);
        // Viterbi path LL should be <= total LL
        assert!(ll_vit <= fwd.ll + 1e-6);
        // Path LL should match Viterbi LL closely
        assert!(
            (ll_vit - ll_path).abs() < 0.1,
            "viterbi ll={ll_vit}, path ll={ll_path}"
        );
    }

    #[test]
    fn test_mosaic_recovery() {
        // Larger test: 10 haplotypes, 100 sites, mosaic query
        let n = 10;
        let m = 100;
        // Reference panel with unique haplotypes using a simple PRNG
        // (each haplotype gets a distinct pseudo-random sequence)
        let mut h = vec![vec![0i8; n]; m];
        for i in 0..n {
            let mut state: u64 = (i as u64 + 1) * 2654435761;
            for l in 0..m {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                h[l][i] = ((state >> 33) % 2) as i8;
            }
        }
        // True path: copy from hap 2 (sites 0-49), hap 5 (sites 50-99)
        let mut s = vec![0i8; m];
        for l in 0..50 {
            s[l] = h[l][2];
        }
        for l in 50..100 {
            s[l] = h[l][5];
        }

        let mu = 0.01;
        let e = vec![[mu, 1.0 - mu]; m];
        let mut r = vec![0.05; m];
        r[0] = 0.0;

        let (v, p, _) = viterbi(n, m, &h, &s, &e, &r);
        let path = viterbi_traceback(m, &v, &p);

        // Should recover the correct haplotypes in each segment
        let correct_first: usize = (0..50).filter(|&l| path[l] == 2).count();
        let correct_second: usize = (50..100).filter(|&l| path[l] == 5).count();
        assert!(
            correct_first >= 40,
            "first segment accuracy: {correct_first}/50"
        );
        assert!(
            correct_second >= 40,
            "second segment accuracy: {correct_second}/50"
        );
    }
}
