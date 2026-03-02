/// ODE system for the SMC++ undistinguished lineage count process.
///
/// The rate matrix Q governs how undistinguished lineages coalesce.
/// Matrix exponential exp(dt/λ · Q) gives exact transition probabilities
/// for piecewise-constant population size.

use ndarray::Array2;

/// Build the rate matrix Q for the undistinguished lineage count process.
///
/// States are indexed j = 1, 2, ..., n_undist (stored 0-indexed).
/// Q[j-1, j-1] = -C(j,2) = -j(j-1)/2
/// Q[j-1, j]   = C(j+1, 2) = (j+1)j/2  (inflow from j+1 -> j)
pub fn build_rate_matrix(n_undist: usize) -> Array2<f64> {
    let mut q = Array2::<f64>::zeros((n_undist, n_undist));
    for j in 1..=n_undist {
        let idx = j - 1;
        q[[idx, idx]] = -((j * (j - 1)) as f64) / 2.0;
        if j < n_undist {
            q[[idx, idx + 1]] = ((j + 1) * j) as f64 / 2.0;
        }
    }
    q
}

/// Eigendecomposition of the rate matrix Q.
///
/// Since Q is upper triangular, eigenvalues are the diagonal entries.
/// Eigenvectors are computed by back-substitution.
pub fn eigendecompose_rate_matrix(
    n_undist: usize,
) -> (Vec<f64>, Array2<f64>, Array2<f64>) {
    let q = build_rate_matrix(n_undist);

    // Eigenvalues = diagonal
    let eigenvalues: Vec<f64> = (0..n_undist).map(|i| q[[i, i]]).collect();

    // Compute eigenvectors by back-substitution
    let mut v = Array2::<f64>::zeros((n_undist, n_undist));
    for j in 0..n_undist {
        v[[j, j]] = 1.0;
        for i in (0..j).rev() {
            let mut rhs = 0.0;
            for k in (i + 1)..=j {
                rhs += q[[i, k]] * v[[k, j]];
            }
            let denom = eigenvalues[j] - q[[i, i]];
            if denom.abs() > 1e-15 {
                v[[i, j]] = rhs / denom;
            }
        }
    }

    // Compute inverse by solving V * V_inv = I
    let v_inv = invert_upper_triangular(&v, n_undist);

    (eigenvalues, v, v_inv)
}

/// Invert an upper triangular matrix.
fn invert_upper_triangular(v: &Array2<f64>, n: usize) -> Array2<f64> {
    let mut inv = Array2::<f64>::zeros((n, n));

    for i in 0..n {
        inv[[i, i]] = 1.0 / v[[i, i]];
    }

    for col in (0..n).rev() {
        for row in (0..col).rev() {
            let mut sum = 0.0;
            for k in (row + 1)..=col {
                sum += v[[row, k]] * inv[[k, col]];
            }
            inv[[row, col]] = -sum / v[[row, row]];
        }
    }

    inv
}

/// Compute exp(t/lam * Q) using precomputed eigendecomposition.
pub fn fast_matrix_exp(
    eigenvalues: &[f64],
    v: &Array2<f64>,
    v_inv: &Array2<f64>,
    t: f64,
    lam: f64,
) -> Array2<f64> {
    let n = eigenvalues.len();
    let mut d = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        d[[i, i]] = (eigenvalues[i] * t / lam).exp();
    }
    v.dot(&d).dot(v_inv)
}

/// Solve the ODE system for piecewise-constant population size.
///
/// Returns `p_at_breaks[k]` = P(J(t_k) = j) for j = 1, ..., n_undist.
pub fn solve_ode_piecewise(
    n_undist: usize,
    time_breaks: &[f64],
    lambdas: &[f64],
) -> Vec<Vec<f64>> {
    let (eig, v, v_inv) = eigendecompose_rate_matrix(n_undist);

    // Initial condition: all n_undist lineages present
    let mut p = vec![0.0; n_undist];
    p[n_undist - 1] = 1.0;

    let mut p_at_breaks = Vec::with_capacity(time_breaks.len());
    p_at_breaks.push(p.clone());

    for k in 0..(time_breaks.len() - 1) {
        let dt = time_breaks[k + 1] - time_breaks[k];
        let lam = lambdas[k];
        let m = fast_matrix_exp(&eig, &v, &v_inv, dt, lam);

        // p_new = M @ p
        let mut p_new = vec![0.0; n_undist];
        for i in 0..n_undist {
            for j in 0..n_undist {
                p_new[i] += m[[i, j]] * p[j];
            }
        }
        p = p_new;
        p_at_breaks.push(p.clone());
    }

    p_at_breaks
}

/// Compute h(t) values at each time break.
pub fn compute_h_values(
    time_breaks: &[f64],
    p_history: &[Vec<f64>],
    lambdas: &[f64],
) -> Vec<f64> {
    let mut h = Vec::with_capacity(time_breaks.len());
    for k in 0..time_breaks.len() {
        let lam = lambdas[k.min(lambdas.len() - 1)];
        let p_j = &p_history[k];
        let expected_j: f64 = p_j
            .iter()
            .enumerate()
            .map(|(idx, &pj)| (idx + 1) as f64 * pj)
            .sum();
        h.push(expected_j / lam);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_matrix_structure() {
        let q = build_rate_matrix(4);
        // j=1: Q[0,0] = 0
        assert!((q[[0, 0]] - 0.0).abs() < 1e-10);
        // j=2: Q[1,1] = -1
        assert!((q[[1, 1]] - (-1.0)).abs() < 1e-10);
        // j=3: Q[2,2] = -3
        assert!((q[[2, 2]] - (-3.0)).abs() < 1e-10);
        // j=4: Q[3,3] = -6
        assert!((q[[3, 3]] - (-6.0)).abs() < 1e-10);
    }

    #[test]
    fn test_eigendecomposition_recovery() {
        let n = 4;
        let q = build_rate_matrix(n);
        let (eig, v, v_inv) = eigendecompose_rate_matrix(n);

        // V * diag(eig) * V_inv should equal Q
        let mut d = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            d[[i, i]] = eig[i];
        }
        let recovered = v.dot(&d).dot(&v_inv);
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (recovered[[i, j]] - q[[i, j]]).abs() < 1e-8,
                    "mismatch at [{i},{j}]: {} vs {}",
                    recovered[[i, j]],
                    q[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_matrix_exp_identity_at_zero() {
        let n = 4;
        let (eig, v, v_inv) = eigendecompose_rate_matrix(n);
        let m = fast_matrix_exp(&eig, &v, &v_inv, 0.0, 1.0);
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (m[[i, j]] - expected).abs() < 1e-10,
                    "identity fail at [{i},{j}]"
                );
            }
        }
    }

    #[test]
    fn test_ode_initial_condition() {
        let n_undist = 9;
        let time_breaks = vec![0.0, 1.0, 2.0];
        let lambdas = vec![1.0, 1.0];
        let p = solve_ode_piecewise(n_undist, &time_breaks, &lambdas);

        // At t=0, all mass on j=n_undist
        assert!((p[0][n_undist - 1] - 1.0).abs() < 1e-10);

        // At t>0, mass should spread to lower j values
        assert!(p[1][n_undist - 1] < 1.0);
        assert!(p[1].iter().sum::<f64>() > 0.99);
    }

    #[test]
    fn test_ode_probability_conservation() {
        let n_undist = 5;
        let time_breaks: Vec<f64> = (0..=50).map(|i| i as f64 * 0.1).collect();
        let lambdas = vec![1.0; 50];
        let p = solve_ode_piecewise(n_undist, &time_breaks, &lambdas);

        for k in 0..p.len() {
            let sum: f64 = p[k].iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "probability sum at k={k}: {sum}"
            );
        }
    }

    #[test]
    fn test_h_values_decrease() {
        let n_undist = 9;
        let time_breaks: Vec<f64> = (0..=20).map(|i| i as f64 * 0.25).collect();
        let lambdas = vec![1.0; 20];
        let p = solve_ode_piecewise(n_undist, &time_breaks, &lambdas);
        let h = compute_h_values(&time_breaks, &p, &lambdas);

        // h should decrease as lineages coalesce
        assert!(h[0] > h[h.len() - 1]);
    }
}
