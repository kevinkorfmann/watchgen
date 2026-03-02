/// SGPR (Sub-Graph Pruning and Re-grafting): MCMC updates for ARG exploration.
///
/// The MCMC mechanism removes a subtree and re-threads it, accepting or
/// rejecting the new ARG based on a simple tree height ratio.

use rand::Rng;
use std::collections::HashMap;

/// A simple tree representation using parent and time maps.
#[derive(Debug, Clone)]
pub struct SimpleTree {
    pub parent: HashMap<u32, u32>,
    pub time: HashMap<u32, f64>,
    pub children: HashMap<u32, Vec<u32>>,
}

impl SimpleTree {
    /// Build a tree from parent and time maps.
    pub fn new(parent: HashMap<u32, u32>, time: HashMap<u32, f64>) -> Self {
        let mut children: HashMap<u32, Vec<u32>> = HashMap::new();
        for (&child, &par) in &parent {
            children.entry(par).or_default().push(child);
        }
        Self {
            parent,
            time,
            children,
        }
    }

    /// List all branches as (child, parent, time_child).
    pub fn branches(&self) -> Vec<(u32, u32, f64)> {
        self.parent
            .iter()
            .map(|(&child, &par)| (child, par, self.time[&child]))
            .collect()
    }

    /// Tree height (maximum node time).
    pub fn height(&self) -> f64 {
        self.time
            .values()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Find the root (node with no parent).
    pub fn root(&self) -> Option<u32> {
        self.time
            .keys()
            .find(|&&node| !self.parent.contains_key(&node))
            .copied()
    }

    /// Get leaf nodes (nodes with no children).
    pub fn leaves(&self) -> Vec<u32> {
        self.time
            .keys()
            .filter(|&&node| !self.children.contains_key(&node) || self.children[&node].is_empty())
            .copied()
            .collect()
    }
}

/// Perform an SPR (Subtree Pruning and Regrafting) move.
///
/// Cuts the edge above `cut_node`, removes it from the tree,
/// and re-attaches it below `new_parent` at `new_time`.
pub fn spr_move(tree: &mut SimpleTree, cut_node: u32, new_parent: u32, new_time: f64) {
    // Remove cut_node from its current parent's children
    if let Some(&old_parent) = tree.parent.get(&cut_node) {
        if let Some(children) = tree.children.get_mut(&old_parent) {
            children.retain(|&c| c != cut_node);
        }
    }

    // Attach to new parent
    tree.parent.insert(cut_node, new_parent);
    tree.children
        .entry(new_parent)
        .or_default()
        .push(cut_node);

    // Update the coalescence time (the time of the join point)
    // In a real implementation this would create a new internal node
    tree.time.insert(cut_node, new_time);
}

/// Randomly select a node to cut and a new parent.
///
/// Returns `(cut_node, new_parent)`.
pub fn select_cut<R: Rng>(tree: &SimpleTree, rng: &mut R) -> Option<(u32, u32)> {
    let leaves = tree.leaves();
    if leaves.len() < 2 {
        return None;
    }

    // Pick a random non-root node to cut
    let non_root: Vec<u32> = tree
        .parent
        .keys()
        .copied()
        .collect();

    if non_root.is_empty() {
        return None;
    }

    let cut_idx = rng.gen_range(0..non_root.len());
    let cut_node = non_root[cut_idx];

    // Pick a random other branch to attach to
    let candidates: Vec<u32> = tree
        .time
        .keys()
        .filter(|&&n| n != cut_node)
        .copied()
        .collect();

    if candidates.is_empty() {
        return None;
    }

    let new_parent_idx = rng.gen_range(0..candidates.len());
    let new_parent = candidates[new_parent_idx];

    Some((cut_node, new_parent))
}

/// SGPR acceptance ratio: min(1, old_height / new_height).
pub fn sgpr_acceptance_ratio(old_height: f64, new_height: f64) -> f64 {
    if new_height <= 0.0 {
        return 1.0;
    }
    (old_height / new_height).min(1.0)
}

/// Simulate tree heights under neutral coalescent for n samples.
pub fn simulate_tree_heights<R: Rng>(n: usize, n_replicates: usize, rng: &mut R) -> Vec<f64> {
    let mut heights = Vec::with_capacity(n_replicates);

    for _ in 0..n_replicates {
        let mut k = n;
        let mut t = 0.0;
        while k > 1 {
            let rate = (k * (k - 1)) as f64 / 2.0;
            let wait: f64 = -rng.gen::<f64>().ln() / rate;
            t += wait;
            k -= 1;
        }
        heights.push(t);
    }

    heights
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_tree() -> SimpleTree {
        // Simple 3-tip tree:
        //     4 (root, t=3.0)
        //    / \
        //   3   2  (t=1.5 for node 3)
        //  / \
        // 0   1  (tips, t=0)
        let mut parent = HashMap::new();
        parent.insert(0, 3);
        parent.insert(1, 3);
        parent.insert(2, 4);
        parent.insert(3, 4);

        let mut time = HashMap::new();
        time.insert(0, 0.0);
        time.insert(1, 0.0);
        time.insert(2, 0.0);
        time.insert(3, 1.5);
        time.insert(4, 3.0);

        SimpleTree::new(parent, time)
    }

    #[test]
    fn test_tree_height() {
        let tree = make_test_tree();
        assert!((tree.height() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_tree_leaves() {
        let tree = make_test_tree();
        let mut leaves = tree.leaves();
        leaves.sort();
        assert_eq!(leaves, vec![0, 1, 2]);
    }

    #[test]
    fn test_tree_root() {
        let tree = make_test_tree();
        assert_eq!(tree.root(), Some(4));
    }

    #[test]
    fn test_tree_branches() {
        let tree = make_test_tree();
        let branches = tree.branches();
        assert_eq!(branches.len(), 4);
    }

    #[test]
    fn test_sgpr_acceptance_ratio() {
        assert!((sgpr_acceptance_ratio(3.0, 3.0) - 1.0).abs() < 1e-10);
        assert!((sgpr_acceptance_ratio(3.0, 6.0) - 0.5).abs() < 1e-10);
        assert!((sgpr_acceptance_ratio(6.0, 3.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_simulate_tree_heights() {
        let mut rng = rand::thread_rng();
        let heights = simulate_tree_heights(10, 1000, &mut rng);
        assert_eq!(heights.len(), 1000);
        let mean: f64 = heights.iter().sum::<f64>() / 1000.0;
        // Expected height for n=10 samples: sum_{k=2}^{10} 2/(k(k-1)) ≈ 1.8
        assert!(mean > 0.5 && mean < 5.0, "mean height = {mean}");
    }

    #[test]
    fn test_select_cut() {
        let tree = make_test_tree();
        let mut rng = rand::thread_rng();
        let result = select_cut(&tree, &mut rng);
        assert!(result.is_some());
        let (cut, new_parent) = result.unwrap();
        assert_ne!(cut, new_parent);
    }
}
