/// Diploid Li-Stephens HMM: forward and Viterbi over (n×n) state space.

/// Map (ref_genotype, query_genotype) to emission matrix index.
///
/// Index = 4*is_match + 2*is_ref_het + is_query_het.
pub fn genotype_comparison_index(ref_gt: i8, query_gt: i8) -> usize {
    if query_gt < 0 {
        return 3; // MISSING
    }
    let is_match = (ref_gt == query_gt) as usize;
    let is_ref_het = (ref_gt == 1) as usize;
    let is_query_het = (query_gt == 1) as usize;
    4 * is_match + 2 * is_ref_het + is_query_het
}

/// Build diploid emission matrix: (m, 8).
///
/// Indices: 4=EQUAL_BOTH_HOM, 0=UNEQUAL_BOTH_HOM, 7=BOTH_HET,
///          1=REF_HOM_OBS_HET, 2=REF_HET_OBS_HOM, 3=MISSING.
pub fn emission_matrix_diploid(mu: f64, num_sites: usize, num_alleles: &[usize]) -> Vec<[f64; 8]> {
    let mut e = vec![[0.0f64; 8]; num_sites];
    for i in 0..num_sites {
        let (p_mut, p_no_mut) = if num_alleles[i] <= 1 {
            (0.0, 1.0)
        } else {
            (mu / (num_alleles[i] - 1) as f64, 1.0 - mu)
        };
        e[i][4] = p_no_mut * p_no_mut; // EQUAL_BOTH_HOM
        e[i][0] = p_mut * p_mut; // UNEQUAL_BOTH_HOM
        e[i][7] = p_no_mut * p_no_mut + p_mut * p_mut; // BOTH_HET
        e[i][1] = 2.0 * p_mut * p_no_mut; // REF_HOM_OBS_HET
        e[i][2] = p_mut * p_no_mut; // REF_HET_OBS_HOM
        e[i][3] = 1.0; // MISSING
    }
    e
}

/// Build reference genotype matrix from haplotype panel.
///
/// `g[l][j1][j2] = h[l][j1] + h[l][j2]`.
pub fn build_genotype_matrix(h: &[Vec<i8>], n: usize, m: usize) -> Vec<Vec<Vec<i8>>> {
    let mut g = vec![vec![vec![0i8; n]; n]; m];
    for l in 0..m {
        for j1 in 0..n {
            for j2 in 0..n {
                g[l][j1][j2] = h[l][j1] + h[l][j2];
            }
        }
    }
    g
}

/// Diploid forward algorithm.
///
/// State space is (n × n) — O(n²·m) per forward pass.
/// Returns `(f, c, ll)` where `f[l]` is a flattened n×n matrix.
pub fn forward_diploid(
    n: usize,
    m: usize,
    g: &[Vec<Vec<i8>>],
    s: &[i8],
    e: &[[f64; 8]],
    r: &[f64],
) -> (Vec<Vec<f64>>, Vec<f64>, f64) {
    let nn = n * n;
    let mut f = vec![vec![0.0; nn]; m];
    let mut c = vec![0.0_f64; m];
    let r_n: Vec<f64> = r.iter().map(|&ri| ri / n as f64).collect();

    // Initialize
    for j1 in 0..n {
        for j2 in 0..n {
            let idx = genotype_comparison_index(g[0][j1][j2], s[0]);
            f[0][j1 * n + j2] = (1.0 / nn as f64) * e[0][idx];
            c[0] += f[0][j1 * n + j2];
        }
    }
    if c[0] > 0.0 {
        for v in f[0].iter_mut() {
            *v /= c[0];
        }
    }

    // Recurse
    for l in 1..m {
        // Precompute summaries from previous step
        let mut f_j2_change = vec![0.0; n]; // j1 stays, j2 switches
        let mut f_j1_change = vec![0.0; n]; // j2 stays, j1 switches

        for j1 in 0..n {
            for j2 in 0..n {
                let prev = f[l - 1][j1 * n + j2];
                f_j2_change[j1] += (1.0 - r[l]) * r_n[l] * prev;
                f_j1_change[j2] += (1.0 - r[l]) * r_n[l] * prev;
            }
        }

        let both_switch = r_n[l] * r_n[l];

        for j1 in 0..n {
            for j2 in 0..n {
                let no_change = (1.0 - r[l]).powi(2) * f[l - 1][j1 * n + j2];
                f[l][j1 * n + j2] =
                    no_change + f_j2_change[j1] + f_j1_change[j2] + both_switch;

                let idx = genotype_comparison_index(g[l][j1][j2], s[l]);
                f[l][j1 * n + j2] *= e[l][idx];
                c[l] += f[l][j1 * n + j2];
            }
        }
        if c[l] > 0.0 {
            for v in f[l].iter_mut() {
                *v /= c[l];
            }
        }
    }

    let ll = c.iter().filter(|&&ci| ci > 0.0).map(|ci| ci.log10()).sum();
    (f, c, ll)
}

/// Diploid Viterbi algorithm.
///
/// Returns `(v_last, pointers, ll)` where pointers index into flattened n×n space.
pub fn viterbi_diploid(
    n: usize,
    m: usize,
    g: &[Vec<Vec<i8>>],
    s: &[i8],
    e: &[[f64; 8]],
    r: &[f64],
) -> (Vec<f64>, Vec<Vec<usize>>, f64) {
    let nn = n * n;
    let mut v_prev = vec![0.0; nn];
    let mut p = vec![vec![0usize; nn]; m];
    let mut c = vec![1.0_f64; m];
    let r_n: Vec<f64> = r.iter().map(|&ri| ri / n as f64).collect();

    // Initialize
    for j1 in 0..n {
        for j2 in 0..n {
            let idx = genotype_comparison_index(g[0][j1][j2], s[0]);
            v_prev[j1 * n + j2] = (1.0 / nn as f64) * e[0][idx];
        }
    }

    // Recurse
    for l in 1..m {
        let max_val = v_prev.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let argmax = v_prev
            .iter()
            .position(|&x| (x - max_val).abs() < f64::EPSILON)
            .unwrap_or(0);
        c[l] = max_val;
        if c[l] > 0.0 {
            for vi in v_prev.iter_mut() {
                *vi /= c[l];
            }
        }

        // Row/col max for single-switch
        let mut row_max = vec![f64::NEG_INFINITY; n];
        let mut row_argmax = vec![0usize; n];
        for j1 in 0..n {
            for j2 in 0..n {
                if v_prev[j1 * n + j2] > row_max[j1] {
                    row_max[j1] = v_prev[j1 * n + j2];
                    row_argmax[j1] = j2;
                }
            }
        }

        let no_switch = |_j1: usize, _j2: usize| {
            (1.0 - r[l]).powi(2) + 2.0 * r_n[l] * (1.0 - r[l]) + r_n[l] * r_n[l]
        };
        let single_switch_val = r_n[l] * (1.0 - r[l]) + r_n[l] * r_n[l];
        let double_switch = r_n[l] * r_n[l];

        let mut v_new = vec![0.0; nn];
        for j1 in 0..n {
            for j2 in 0..n {
                let flat = j1 * n + j2;

                // No switch
                v_new[flat] = v_prev[flat] * no_switch(j1, j2);
                p[l][flat] = flat;

                // Single switch (best of row j1 or row j2)
                let v_single = f64::max(row_max[j1], row_max[j2]);
                let single_val = single_switch_val * v_single;
                if single_val > v_new[flat] {
                    v_new[flat] = single_val;
                    if row_max[j1] >= row_max[j2] {
                        p[l][flat] = j1 * n + row_argmax[j1];
                    } else {
                        p[l][flat] = row_argmax[j2] * n + j2;
                    }
                }

                // Double switch
                if double_switch > v_new[flat] {
                    v_new[flat] = double_switch;
                    p[l][flat] = argmax;
                }

                let idx = genotype_comparison_index(g[l][j1][j2], s[l]);
                v_new[flat] *= e[l][idx];
            }
        }
        v_prev = v_new;
    }

    let max_v = v_prev.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let ll = c
        .iter()
        .filter(|&&ci| ci > 0.0)
        .map(|ci| ci.log10())
        .sum::<f64>()
        + if max_v > 0.0 { max_v.log10() } else { 0.0 };

    (v_prev, p, ll)
}

/// Diploid Viterbi traceback.
pub fn viterbi_traceback_diploid(m: usize, v_last: &[f64], p: &[Vec<usize>]) -> Vec<usize> {
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

/// Convert flat diploid path to two haploid paths.
pub fn get_phased_path(n: usize, flat_path: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let mut p1 = Vec::with_capacity(flat_path.len());
    let mut p2 = Vec::with_capacity(flat_path.len());
    for &idx in flat_path {
        p1.push(idx / n);
        p2.push(idx % n);
    }
    (p1, p2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genotype_comparison_index() {
        assert_eq!(genotype_comparison_index(0, 0), 4); // equal, both hom
        assert_eq!(genotype_comparison_index(0, 2), 0); // unequal, both hom
        assert_eq!(genotype_comparison_index(1, 1), 7); // both het
        assert_eq!(genotype_comparison_index(0, 1), 1); // ref hom, query het
        assert_eq!(genotype_comparison_index(1, 0), 2); // ref het, query hom
        assert_eq!(genotype_comparison_index(0, -1), 3); // missing
    }

    #[test]
    fn test_emission_matrix_diploid_values() {
        let e = emission_matrix_diploid(0.01, 1, &[2]);
        assert!((e[0][4] - 0.99 * 0.99).abs() < 1e-10); // match hom
        assert!((e[0][0] - 0.01 * 0.01).abs() < 1e-10); // mismatch hom
        assert_eq!(e[0][3], 1.0); // missing
    }

    #[test]
    fn test_diploid_forward_ll_finite() {
        let n = 4;
        let m = 10;
        let mut h = vec![vec![0i8; n]; m];
        for l in 0..m {
            for i in 0..n {
                h[l][i] = ((l + i) % 2) as i8;
            }
        }
        let g = build_genotype_matrix(&h, n, m);
        let s: Vec<i8> = (0..m).map(|l| h[l][0] + h[l][1]).collect();
        let e = emission_matrix_diploid(0.01, m, &vec![2; m]);
        let mut r = vec![0.05; m];
        r[0] = 0.0;

        let (_, _, ll) = forward_diploid(n, m, &g, &s, &e, &r);
        assert!(ll.is_finite());
        assert!(ll < 0.0);
    }

    #[test]
    fn test_diploid_viterbi_valid_path() {
        let n = 4;
        let m = 10;
        let mut h = vec![vec![0i8; n]; m];
        for l in 0..m {
            for i in 0..n {
                h[l][i] = ((l + i) % 2) as i8;
            }
        }
        let g = build_genotype_matrix(&h, n, m);
        let s: Vec<i8> = (0..m).map(|l| h[l][1] + h[l][2]).collect();
        let e = emission_matrix_diploid(0.01, m, &vec![2; m]);
        let mut r = vec![0.05; m];
        r[0] = 0.0;

        let (v, p, ll) = viterbi_diploid(n, m, &g, &s, &e, &r);
        let path = viterbi_traceback_diploid(m, &v, &p);
        assert!(ll.is_finite());
        let (p1, p2) = get_phased_path(n, &path);
        for &state in &p1 {
            assert!(state < n);
        }
        for &state in &p2 {
            assert!(state < n);
        }
    }
}
