/// Configuration for Nelder-Mead optimization.
pub struct NelderMeadConfig {
    pub max_iter: usize,
    pub x_tol: f64,
    pub f_tol: f64,
    pub initial_step: f64, // initial simplex perturbation size
    pub alpha: f64,        // reflection
    pub gamma: f64,        // expansion
    pub rho: f64,          // contraction
    pub sigma: f64,        // shrink
}

impl Default for NelderMeadConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            x_tol: 1e-6,
            f_tol: 1e-8,
            initial_step: 0.0, // 0 means use adaptive (5% of |x_i| or 0.00025)
            alpha: 1.0,
            gamma: 2.0,
            rho: 0.5,
            sigma: 0.5,
        }
    }
}

/// Result of Nelder-Mead optimization.
pub struct NelderMeadResult {
    pub x: Vec<f64>,
    pub f: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// Minimize `f` starting from `x0` using the Nelder-Mead simplex method.
pub fn minimize<F>(f: F, x0: &[f64], config: &NelderMeadConfig) -> NelderMeadResult
where
    F: Fn(&[f64]) -> f64,
{
    let n = x0.len();
    let n1 = n + 1;

    // Initialize simplex: x0 plus n perturbed vertices
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n1);
    simplex.push(x0.to_vec());
    for i in 0..n {
        let mut vertex = x0.to_vec();
        let delta = if config.initial_step > 0.0 {
            config.initial_step
        } else if x0[i].abs() > 1e-12 {
            x0[i] * 0.05
        } else {
            0.00025
        };
        vertex[i] += delta;
        simplex.push(vertex);
    }

    // Evaluate function at each vertex
    let mut fvals: Vec<f64> = simplex.iter().map(|v| f(v)).collect();
    let mut order: Vec<usize> = (0..n1).collect();

    let mut iterations = 0;
    let mut converged = false;

    for _iter in 0..config.max_iter {
        iterations = _iter + 1;

        // Sort by function value
        order.sort_by(|&a, &b| fvals[a].partial_cmp(&fvals[b]).unwrap_or(std::cmp::Ordering::Equal));

        // Check convergence: simplex diameter and function range
        let f_best = fvals[order[0]];
        let f_worst = fvals[order[n]];
        if (f_worst - f_best).abs() < config.f_tol {
            // Also check x tolerance
            let mut max_dist = 0.0f64;
            for i in 1..n1 {
                let mut dist = 0.0;
                for j in 0..n {
                    let d = simplex[order[i]][j] - simplex[order[0]][j];
                    dist += d * d;
                }
                max_dist = max_dist.max(dist.sqrt());
            }
            if max_dist < config.x_tol {
                converged = true;
                break;
            }
        }

        let i_best = order[0];
        let i_worst = order[n];
        let i_second_worst = order[n - 1];

        // Compute centroid (excluding worst)
        let mut centroid = vec![0.0; n];
        for &idx in order.iter().take(n) {
            for (j, c) in centroid.iter_mut().enumerate() {
                *c += simplex[idx][j];
            }
        }
        for c in centroid.iter_mut() {
            *c /= n as f64;
        }

        // Reflection
        let mut xr = vec![0.0; n];
        for j in 0..n {
            xr[j] = centroid[j] + config.alpha * (centroid[j] - simplex[i_worst][j]);
        }
        let fr = f(&xr);

        if fr < fvals[i_best] {
            // Try expansion
            let mut xe = vec![0.0; n];
            for j in 0..n {
                xe[j] = centroid[j] + config.gamma * (xr[j] - centroid[j]);
            }
            let fe = f(&xe);
            if fe < fr {
                simplex[i_worst] = xe;
                fvals[i_worst] = fe;
            } else {
                simplex[i_worst] = xr;
                fvals[i_worst] = fr;
            }
        } else if fr < fvals[i_second_worst] {
            // Accept reflection
            simplex[i_worst] = xr;
            fvals[i_worst] = fr;
        } else {
            // Contraction
            let (xc, fc) = if fr < fvals[i_worst] {
                // Outside contraction
                let mut xc = vec![0.0; n];
                for j in 0..n {
                    xc[j] = centroid[j] + config.rho * (xr[j] - centroid[j]);
                }
                let fc = f(&xc);
                (xc, fc)
            } else {
                // Inside contraction
                let mut xc = vec![0.0; n];
                for j in 0..n {
                    xc[j] = centroid[j] + config.rho * (simplex[i_worst][j] - centroid[j]);
                }
                let fc = f(&xc);
                (xc, fc)
            };

            if fc < fvals[i_worst] {
                simplex[i_worst] = xc;
                fvals[i_worst] = fc;
            } else {
                // Shrink
                let best = simplex[i_best].clone();
                for &idx in order.iter().take(n1) {
                    if idx != i_best {
                        for (j, &bj) in best.iter().enumerate().take(n) {
                            simplex[idx][j] = bj + config.sigma * (simplex[idx][j] - bj);
                        }
                        fvals[idx] = f(&simplex[idx]);
                    }
                }
            }
        }
    }

    // Final sort
    order.sort_by(|&a, &b| fvals[a].partial_cmp(&fvals[b]).unwrap_or(std::cmp::Ordering::Equal));
    let best_idx = order[0];

    NelderMeadResult {
        x: simplex[best_idx].clone(),
        f: fvals[best_idx],
        iterations,
        converged,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimize_quadratic() {
        // Minimize (x-3)^2 + (y-5)^2
        let result = minimize(
            |x| (x[0] - 3.0).powi(2) + (x[1] - 5.0).powi(2),
            &[0.0, 0.0],
            &NelderMeadConfig::default(),
        );
        assert!((result.x[0] - 3.0).abs() < 1e-4);
        assert!((result.x[1] - 5.0).abs() < 1e-4);
        assert!(result.f < 1e-8);
    }

    #[test]
    fn test_minimize_rosenbrock() {
        // Minimize Rosenbrock: (1-x)^2 + 100*(y - x^2)^2
        let config = NelderMeadConfig {
            max_iter: 5000,
            x_tol: 1e-8,
            f_tol: 1e-12,
            ..Default::default()
        };
        let result = minimize(
            |x| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2),
            &[0.0, 0.0],
            &config,
        );
        assert!((result.x[0] - 1.0).abs() < 1e-2);
        assert!((result.x[1] - 1.0).abs() < 1e-2);
    }

    #[test]
    fn test_minimize_1d() {
        let result = minimize(
            |x| (x[0] - 2.5).powi(2),
            &[0.0],
            &NelderMeadConfig::default(),
        );
        assert!((result.x[0] - 2.5).abs() < 1e-4);
    }
}
