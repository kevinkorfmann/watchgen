/// ARG rescaling: adjusts coalescence times to match the mutation clock.
///
/// After inferring an ARG, the raw coalescence times may not match the
/// mutation rate. Rescaling partitions the time axis into windows of equal
/// total branch length, counts mutations in each window, and applies a
/// piecewise linear transformation to bring observed and expected mutation
/// counts into alignment.

use std::collections::HashMap;

/// An ARG branch with genomic span and time bounds.
#[derive(Debug, Clone)]
pub struct ArgBranch {
    /// Genomic span in base pairs.
    pub span: f64,
    /// Lower time bound.
    pub lower_time: f64,
    /// Upper time bound.
    pub upper_time: f64,
}

/// A mutation with time and genomic position.
#[derive(Debug, Clone)]
pub struct Mutation {
    pub time: f64,
    pub position: f64,
}

/// Compute total ARG branch length (weighted by span) in a time window.
pub fn compute_arg_length_in_window(
    branches: &[ArgBranch],
    window_lower: f64,
    window_upper: f64,
) -> f64 {
    let mut total = 0.0;
    for b in branches {
        let overlap_lo = b.lower_time.max(window_lower);
        let overlap_hi = b.upper_time.min(window_upper);
        if overlap_hi > overlap_lo {
            total += b.span * (overlap_hi - overlap_lo);
        }
    }
    total
}

/// Partition the time axis into J windows of approximately equal total ARG length.
///
/// Returns J+1 boundary times.
pub fn partition_time_axis(branches: &[ArgBranch], j: usize) -> Vec<f64> {
    if branches.is_empty() || j == 0 {
        return vec![0.0, 1.0];
    }

    // Find time range
    let t_min = 0.0;
    let t_max = branches
        .iter()
        .map(|b| b.upper_time)
        .fold(f64::NEG_INFINITY, f64::max);

    // Compute total ARG length
    let total_length = compute_arg_length_in_window(branches, t_min, t_max);
    let target_per_window = total_length / j as f64;

    let mut boundaries = Vec::with_capacity(j + 1);
    boundaries.push(t_min);

    // Sweep through time to find boundaries
    let n_steps = 1000;
    let dt = (t_max - t_min) / n_steps as f64;
    let mut cumulative = 0.0;
    let mut window_idx = 1;

    for step in 0..n_steps {
        let t_lo = t_min + step as f64 * dt;
        let t_hi = t_lo + dt;
        cumulative += compute_arg_length_in_window(branches, t_lo, t_hi);

        if cumulative >= target_per_window * window_idx as f64 && window_idx < j {
            boundaries.push(t_hi);
            window_idx += 1;
        }
    }

    boundaries.push(t_max);

    // Ensure we have exactly j+1 boundaries
    while boundaries.len() < j + 1 {
        boundaries.push(t_max);
    }
    boundaries.truncate(j + 1);

    boundaries
}

/// Count mutations falling into each time window.
pub fn count_mutations_per_window(mutations: &[Mutation], boundaries: &[f64]) -> Vec<f64> {
    let j = boundaries.len() - 1;
    let mut counts = vec![0.0; j];
    for m in mutations {
        for w in 0..j {
            if m.time >= boundaries[w] && m.time < boundaries[w + 1] {
                counts[w] += 1.0;
                break;
            }
        }
    }
    counts
}

/// Compute per-window scaling factors.
///
/// `c[j] = observed_j / expected_j` where expected = theta * total_arg_length / J.
pub fn compute_scaling_factors(
    counts: &[f64],
    total_arg_length: f64,
    theta: f64,
    j: usize,
) -> Vec<f64> {
    let expected_per_window = theta * total_arg_length / j as f64;
    if expected_per_window <= 0.0 {
        return vec![1.0; j];
    }
    counts
        .iter()
        .map(|&c| {
            let ratio = c / expected_per_window;
            ratio.max(0.1).min(10.0) // Clamp to avoid extreme scaling
        })
        .collect()
}

/// Apply piecewise-linear rescaling to node times.
///
/// Returns a map from node_id to rescaled time.
pub fn rescale_times(
    node_times: &HashMap<u32, f64>,
    boundaries: &[f64],
    scaling_factors: &[f64],
) -> HashMap<u32, f64> {
    let j = scaling_factors.len();

    // Compute new boundaries after rescaling
    let mut new_boundaries = vec![0.0; j + 1];
    for w in 0..j {
        let width = boundaries[w + 1] - boundaries[w];
        new_boundaries[w + 1] = new_boundaries[w] + scaling_factors[w] * width;
    }

    let mut rescaled = HashMap::new();
    for (&node_id, &t) in node_times {
        // Find which window this time falls in
        let mut new_t = t;
        for w in 0..j {
            if t >= boundaries[w] && t < boundaries[w + 1] {
                let frac = if (boundaries[w + 1] - boundaries[w]).abs() > 1e-15 {
                    (t - boundaries[w]) / (boundaries[w + 1] - boundaries[w])
                } else {
                    0.0
                };
                new_t = new_boundaries[w] + frac * (new_boundaries[w + 1] - new_boundaries[w]);
                break;
            }
        }
        rescaled.insert(node_id, new_t);
    }

    rescaled
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_branches() -> Vec<ArgBranch> {
        vec![
            ArgBranch {
                span: 1000.0,
                lower_time: 0.0,
                upper_time: 2.0,
            },
            ArgBranch {
                span: 1000.0,
                lower_time: 0.0,
                upper_time: 5.0,
            },
            ArgBranch {
                span: 500.0,
                lower_time: 2.0,
                upper_time: 5.0,
            },
        ]
    }

    #[test]
    fn test_arg_length_in_window() {
        let branches = make_test_branches();
        let len = compute_arg_length_in_window(&branches, 0.0, 1.0);
        // Branch 1: 1000 * 1.0 = 1000
        // Branch 2: 1000 * 1.0 = 1000
        // Branch 3: 0 (starts at 2.0)
        assert!((len - 2000.0).abs() < 1e-6);
    }

    #[test]
    fn test_partition_time_axis_boundaries() {
        let branches = make_test_branches();
        let bounds = partition_time_axis(&branches, 5);
        assert_eq!(bounds.len(), 6);
        assert_eq!(bounds[0], 0.0);
        for i in 0..5 {
            assert!(bounds[i] < bounds[i + 1]);
        }
    }

    #[test]
    fn test_count_mutations() {
        let mutations = vec![
            Mutation {
                time: 0.5,
                position: 100.0,
            },
            Mutation {
                time: 1.5,
                position: 200.0,
            },
            Mutation {
                time: 2.5,
                position: 300.0,
            },
        ];
        let boundaries = vec![0.0, 1.0, 2.0, 3.0];
        let counts = count_mutations_per_window(&mutations, &boundaries);
        assert!((counts[0] - 1.0).abs() < 1e-10);
        assert!((counts[1] - 1.0).abs() < 1e-10);
        assert!((counts[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_scaling_factors() {
        let counts = vec![2.0, 4.0, 1.0];
        let factors = compute_scaling_factors(&counts, 300.0, 0.01, 3);
        // expected = 0.01 * 300 / 3 = 1.0
        assert!((factors[0] - 2.0).abs() < 1e-10);
        assert!((factors[1] - 4.0).abs() < 1e-10);
        assert!((factors[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rescale_preserves_order() {
        let mut node_times = HashMap::new();
        node_times.insert(0, 0.5);
        node_times.insert(1, 1.5);
        node_times.insert(2, 2.5);

        let boundaries = vec![0.0, 1.0, 2.0, 3.0];
        let factors = vec![1.0, 2.0, 0.5];

        let rescaled = rescale_times(&node_times, &boundaries, &factors);
        assert!(rescaled[&0] < rescaled[&1]);
        assert!(rescaled[&1] < rescaled[&2]);
    }

    #[test]
    fn test_rescale_identity() {
        let mut node_times = HashMap::new();
        node_times.insert(0, 0.5);
        node_times.insert(1, 1.5);

        let boundaries = vec![0.0, 1.0, 2.0];
        let factors = vec![1.0, 1.0]; // identity scaling

        let rescaled = rescale_times(&node_times, &boundaries, &factors);
        assert!((rescaled[&0] - 0.5).abs() < 1e-10);
        assert!((rescaled[&1] - 1.5).abs() < 1e-10);
    }
}
