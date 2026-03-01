"""
Mini-Relate: A minimal implementation of the Relate algorithm.

Relate (Speidel, Forest, Shi & Myers, 2019) estimates genome-wide genealogies
-- local trees along the chromosome -- from phased haplotype data. Its defining
design decision is the **two-phase architecture**: infer tree topologies first
(fast, heuristic), then estimate branch lengths second (rigorous, MCMC).

The implementation covers five gears:

1. **Asymmetric Painting** -- A modified Li & Stephens HMM with directional
   emission probabilities that produces an asymmetric distance matrix.
2. **Tree Building** -- Agglomerative clustering on asymmetric distances to
   produce rooted binary trees.
3. **Mutation Mapping** -- Placing derived alleles on tree branches under the
   infinite-sites model.
4. **Branch Length MCMC** -- Metropolis-Hastings sampling of coalescence times
   using a Poisson mutation likelihood and a coalescent prior.
5. **Population Size Estimation** -- EM algorithm for piecewise-constant N_e(t).
"""

import numpy as np
from scipy.special import gammaln


# ---------------------------------------------------------------------------
# Gear 1: Asymmetric Painting
# ---------------------------------------------------------------------------

def directional_emission(h_target, h_ref, mu, w_d=1.0, w_a=0.5):
    """Compute directional emission probability.

    Parameters
    ----------
    h_target : int
        Allele of the target haplotype (0 = ancestral, 1 = derived).
    h_ref : int
        Allele of the reference haplotype.
    mu : float
        Base mismatch probability.
    w_d : float
        Weight for "target derived, reference ancestral" mismatch.
    w_a : float
        Weight for "target ancestral, reference derived" mismatch.

    Returns
    -------
    float
        Emission probability.
    """
    if h_target == h_ref:
        return 1.0 - mu  # match
    elif h_target == 1 and h_ref == 0:
        return mu * w_d  # target has derived, reference has ancestral
    else:  # h_target == 0 and h_ref == 1
        return mu * w_a  # target has ancestral, reference has derived


def forward_backward_relate(target, panel, rho, mu, w_d=1.0, w_a=0.5):
    """Forward-backward with directional emission for Relate.

    Parameters
    ----------
    target : ndarray of shape (L,)
        Target haplotype (0/1 at each site).
    panel : ndarray of shape (L, K)
        Reference panel (L sites, K haplotypes).
    rho : ndarray of shape (L,)
        Per-site recombination probabilities.
    mu : float
        Base mismatch probability.
    w_d, w_a : float
        Directional mismatch weights.

    Returns
    -------
    posterior : ndarray of shape (L, K)
        Posterior copying probabilities p_ij(ell).
    """
    L, K = panel.shape

    # --- Emission matrix ---
    E = np.zeros((L, K))
    for ell in range(L):
        for j in range(K):
            E[ell, j] = directional_emission(
                target[ell], panel[ell, j], mu, w_d, w_a)

    # --- Forward pass ---
    alpha = np.zeros((L, K))
    # Initialization
    alpha[0] = (1.0 / K) * E[0]
    # Rescale
    scale_f = np.zeros(L)
    scale_f[0] = alpha[0].sum()
    if scale_f[0] > 0:
        alpha[0] /= scale_f[0]

    for ell in range(1, L):
        total = alpha[ell - 1].sum()
        for j in range(K):
            # O(K) trick: stay + switch
            alpha[ell, j] = E[ell, j] * (
                (1 - rho[ell]) * alpha[ell - 1, j]
                + (rho[ell] / K) * total
            )
        scale_f[ell] = alpha[ell].sum()
        if scale_f[ell] > 0:
            alpha[ell] /= scale_f[ell]

    # --- Backward pass ---
    beta = np.zeros((L, K))
    beta[-1] = 1.0

    for ell in range(L - 2, -1, -1):
        # Compute sum_j (beta[ell+1,j] * E[ell+1,j] * rho/K)
        total_be = 0.0
        for j in range(K):
            total_be += beta[ell + 1, j] * E[ell + 1, j] * (rho[ell + 1] / K)
        for j in range(K):
            beta[ell, j] = (
                (1 - rho[ell + 1]) * beta[ell + 1, j] * E[ell + 1, j]
                + total_be
            )
        scale_b = beta[ell].sum()
        if scale_b > 0:
            beta[ell] /= scale_b

    # --- Posterior ---
    posterior = alpha * beta
    for ell in range(L):
        row_sum = posterior[ell].sum()
        if row_sum > 0:
            posterior[ell] /= row_sum

    return posterior


def compute_distance_matrix(haplotypes, positions, recomb_rate, mu,
                             focal_snp, w_d=1.0, w_a=0.5):
    """Compute the asymmetric distance matrix at a focal SNP.

    Parameters
    ----------
    haplotypes : ndarray of shape (N, L)
        Haplotype matrix (N haplotypes, L sites).
    positions : ndarray of float, shape (L,)
        Genomic positions of SNPs.
    recomb_rate : float
        Per-base recombination rate.
    mu : float
        Base mismatch probability.
    focal_snp : int
        Index of the focal SNP.
    w_d, w_a : float
        Directional mismatch weights.

    Returns
    -------
    D : ndarray of shape (N, N)
        Asymmetric distance matrix. D[i,j] = distance from i to j.
    """
    N, L = haplotypes.shape

    # Compute recombination probabilities
    rho = np.zeros(L)
    for ell in range(1, L):
        d = positions[ell] - positions[ell - 1]
        rho[ell] = 1 - np.exp(-d * recomb_rate / max(N - 1, 1))

    D = np.zeros((N, N))

    for i in range(N):
        # Build the panel: all haplotypes except i
        panel_idx = [j for j in range(N) if j != i]
        panel = haplotypes[panel_idx].T  # shape (L, N-1)
        target = haplotypes[i]

        # Run forward-backward
        posterior = forward_backward_relate(
            target, panel, rho, mu, w_d, w_a)

        # Extract posterior at focal SNP
        p_focal = posterior[focal_snp]  # shape (N-1,)

        # Fill in distances
        for idx, j in enumerate(panel_idx):
            D[i, j] = -np.log(max(p_focal[idx], 1e-300))

    return D


# ---------------------------------------------------------------------------
# Gear 2: Tree Building
# ---------------------------------------------------------------------------

class TreeNode:
    """A node in a binary tree."""

    def __init__(self, node_id, left=None, right=None, is_leaf=True):
        self.id = node_id
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.leaf_ids = {node_id} if is_leaf else set()

    def __repr__(self):
        if self.is_leaf:
            return f"Leaf({self.id})"
        return f"Node({self.id}, L={self.left.id}, R={self.right.id})"


def find_pair_to_merge(D, active):
    """Find the pair of active clusters to merge next.

    Parameters
    ----------
    D : ndarray of shape (N, N)
        Asymmetric distance matrix.
    active : set of int
        Indices of currently active (unmerged) clusters.

    Returns
    -------
    i, j : int
        Indices of the pair to merge.
    """
    best_d_min = np.inf
    best_d_sym = np.inf
    best_pair = None

    active_list = sorted(active)
    for idx_a, i in enumerate(active_list):
        for j in active_list[idx_a + 1:]:
            d_min = min(D[i, j], D[j, i])
            d_sym = D[i, j] + D[j, i]

            # Primary criterion: smallest min distance
            # Tiebreaker: smallest symmetrized distance
            if (d_min < best_d_min or
                (d_min == best_d_min and d_sym < best_d_sym)):
                best_d_min = d_min
                best_d_sym = d_sym
                best_pair = (i, j)

    return best_pair


def update_distances(D, i, j, c, active):
    """Update the distance matrix after merging i and j into c.

    Parameters
    ----------
    D : ndarray of shape (M, M)
        Distance matrix (will be modified in-place, M >= max index).
    i, j : int
        Indices of the merged pair.
    c : int
        Index of the new cluster.
    active : set of int
        Currently active clusters (should include c, not i or j).
    """
    for k in active:
        if k == c:
            continue
        # Distance from new cluster to k: minimum of the two children
        D[c, k] = min(D[i, k], D[j, k])
        # Distance from k to new cluster
        D[k, c] = min(D[k, i], D[k, j])

    # Self-distance
    D[c, c] = 0.0


def build_tree(D_orig, N):
    """Build a rooted binary tree from an asymmetric distance matrix.

    Parameters
    ----------
    D_orig : ndarray of shape (N, N)
        Asymmetric distance matrix for N haplotypes.
    N : int
        Number of haplotypes (leaves).

    Returns
    -------
    root : TreeNode
        Root of the binary tree.
    merge_order : list of (int, int, int)
        Sequence of (child1, child2, parent) merges.
    """
    # Allocate space for up to 2N-1 nodes (N leaves + N-1 internal)
    max_nodes = 2 * N - 1
    D = np.full((max_nodes, max_nodes), np.inf)
    D[:N, :N] = D_orig.copy()
    for i in range(N):
        D[i, i] = 0.0

    # Initialize: each haplotype is a leaf node
    nodes = {}
    for i in range(N):
        nodes[i] = TreeNode(i)

    active = set(range(N))
    next_id = N
    merge_order = []

    # Iteratively merge pairs
    for step in range(N - 1):
        # Find the best pair to merge
        i, j = find_pair_to_merge(D, active)

        # Create new internal node
        c = next_id
        parent_node = TreeNode(c, left=nodes[i], right=nodes[j],
                                is_leaf=False)
        parent_node.leaf_ids = nodes[i].leaf_ids | nodes[j].leaf_ids
        nodes[c] = parent_node

        # Update bookkeeping
        active.remove(i)
        active.remove(j)
        active.add(c)

        # Update distances
        update_distances(D, i, j, c, active)

        merge_order.append((i, j, c))
        next_id += 1

    # The last remaining node is the root
    root_id = active.pop()
    return nodes[root_id], merge_order


def to_newick(node):
    """Convert a TreeNode to Newick format string.

    Parameters
    ----------
    node : TreeNode

    Returns
    -------
    str
        Newick representation.
    """
    if node.is_leaf:
        return str(node.id)
    left_str = to_newick(node.left)
    right_str = to_newick(node.right)
    return f"({left_str},{right_str})"


def build_local_trees(haplotypes, positions, recomb_rate, mu):
    """Build a local tree at each focal SNP.

    Parameters
    ----------
    haplotypes : ndarray of shape (N, L)
    positions : ndarray of float, shape (L,)
    recomb_rate : float
    mu : float

    Returns
    -------
    trees : list of dict
        Local trees with genomic intervals.
    """
    N, L = haplotypes.shape
    trees = []
    prev_newick = None

    for s in range(L):
        # Compute asymmetric distance matrix at this focal SNP
        D = compute_distance_matrix(
            haplotypes, positions, recomb_rate, mu, focal_snp=s)

        # Build tree
        root, _ = build_tree(D, N)
        newick = to_newick(root)

        if newick != prev_newick:
            # New tree topology -- start a new interval
            start_pos = positions[s]
            trees.append({
                'start': start_pos,
                'root': root,
                'newick': newick,
                'focal_snp': s,
            })
            prev_newick = newick

    # Set end positions
    for i in range(len(trees) - 1):
        trees[i]['end'] = trees[i + 1]['start']
    if trees:
        trees[-1]['end'] = positions[-1] + 1

    return trees


# ---------------------------------------------------------------------------
# Gear 3: Branch Length Estimation (MCMC)
# ---------------------------------------------------------------------------

def get_descendants(node):
    """Get the set of leaf IDs descended from a node."""
    if node.is_leaf:
        return {node.id}
    return get_descendants(node.left) | get_descendants(node.right)


def map_mutations(root, haplotypes, site_indices):
    """Map mutations to branches of the tree.

    Parameters
    ----------
    root : TreeNode
        Root of the local tree.
    haplotypes : ndarray of shape (N, L)
        Haplotype matrix.
    site_indices : list of int
        Indices of sites that fall within this tree's genomic interval.

    Returns
    -------
    branch_mutations : dict
        {(parent_id, child_id): count} -- number of mutations on each branch.
    unmapped : int
        Number of mutations that don't map to any branch.
    """
    # Pre-compute descendant sets for each internal node
    def collect_branches(node):
        """Collect all branches as (parent_id, child_id, descendant_set)."""
        branches = []
        if not node.is_leaf:
            left_desc = get_descendants(node.left)
            right_desc = get_descendants(node.right)
            branches.append((node.id, node.left.id, left_desc))
            branches.append((node.id, node.right.id, right_desc))
            branches.extend(collect_branches(node.left))
            branches.extend(collect_branches(node.right))
        return branches

    branches = collect_branches(root)
    N = haplotypes.shape[0]

    branch_mutations = {}
    for parent_id, child_id, _ in branches:
        branch_mutations[(parent_id, child_id)] = 0

    unmapped = 0

    for site in site_indices:
        # Which haplotypes carry the derived allele?
        carriers = {i for i in range(N) if haplotypes[i, site] == 1}

        if len(carriers) == 0 or len(carriers) == N:
            continue  # monomorphic -- skip

        # Find the branch whose descendants exactly match the carriers
        matched = False
        for parent_id, child_id, desc_set in branches:
            if desc_set == carriers:
                branch_mutations[(parent_id, child_id)] += 1
                matched = True
                break

        if not matched:
            unmapped += 1

    return branch_mutations, unmapped


def log_mutation_likelihood(branch_mutations, node_times, mu, span):
    """Compute the log Poisson mutation likelihood.

    Parameters
    ----------
    branch_mutations : dict
        {(parent_id, child_id): mutation_count}.
    node_times : dict
        {node_id: coalescence_time}.
    mu : float
        Mutation rate per base per generation.
    span : float
        Genomic span of this tree (in base pairs).

    Returns
    -------
    float
        Log likelihood.
    """
    log_lik = 0.0
    for (parent, child), m_b in branch_mutations.items():
        dt = node_times[parent] - node_times[child]
        if dt <= 0:
            return -np.inf  # invalid: parent must be older than child

        rate = mu * span * dt
        # Poisson log-probability: m*log(rate) - rate - log(m!)
        log_lik += m_b * np.log(rate) - rate - gammaln(m_b + 1)

    return log_lik


def log_coalescent_prior(coalescence_times, N_e):
    """Compute the log coalescent prior for a set of coalescence times.

    Parameters
    ----------
    coalescence_times : list of float
        Coalescence times sorted in increasing order (t_N, t_{N-1}, ..., t_2).
        These are the times of the N-1 internal nodes, sorted youngest first.
    N_e : float
        Effective population size (constant).

    Returns
    -------
    float
        Log prior probability.
    """
    n_coal = len(coalescence_times)
    N = n_coal + 1  # number of leaves (lineages start at N)

    log_prior = 0.0
    prev_time = 0.0  # most recent time (present)

    for idx, t in enumerate(coalescence_times):
        # Number of lineages just before this coalescence
        k = N - idx
        if k < 2:
            break

        # Coalescence rate
        rate = k * (k - 1) / (2.0 * N_e)
        # Waiting time
        dt = t - prev_time
        if dt < 0:
            return -np.inf

        # Exponential log-density: log(rate) - rate * dt
        log_prior += np.log(rate) - rate * dt
        prev_time = t

    return log_prior


def log_posterior(node_times, branch_mutations, mu, span, N_e,
                  internal_ids, leaf_ids):
    """Compute the log posterior over coalescence times.

    Parameters
    ----------
    node_times : dict
        {node_id: time} for all nodes.
    branch_mutations : dict
        {(parent, child): count}.
    mu : float
        Mutation rate.
    span : float
        Genomic span.
    N_e : float
        Effective population size.
    internal_ids : list of int
        IDs of internal nodes, sorted by time (youngest first).
    leaf_ids : list of int
        IDs of leaf nodes.

    Returns
    -------
    float
        Log posterior (up to a constant).
    """
    # Likelihood
    ll = log_mutation_likelihood(branch_mutations, node_times, mu, span)
    if ll == -np.inf:
        return -np.inf

    # Prior: extract coalescence times in order
    coal_times = sorted([node_times[n] for n in internal_ids])
    lp = log_coalescent_prior(coal_times, N_e)

    return ll + lp


def mcmc_branch_lengths(root, branch_mutations, mu, span, N_e,
                         n_samples=1000, burn_in=200, sigma=50.0,
                         seed=42):
    """Estimate branch lengths via Metropolis-Hastings MCMC.

    Parameters
    ----------
    root : TreeNode
        Root of the local tree.
    branch_mutations : dict
        {(parent, child): count}.
    mu : float
        Mutation rate.
    span : float
        Genomic span.
    N_e : float
        Effective population size.
    n_samples : int
        Number of MCMC samples (after burn-in).
    burn_in : int
        Number of burn-in steps.
    sigma : float
        Proposal standard deviation.
    seed : int
        Random seed.

    Returns
    -------
    samples : list of dict
        Posterior samples of node times.
    acceptance_rate : float
    """
    rng = np.random.RandomState(seed)

    # Identify leaf and internal nodes
    leaf_ids = []
    internal_ids = []

    def collect_nodes(node):
        if node.is_leaf:
            leaf_ids.append(node.id)
        else:
            internal_ids.append(node.id)
            collect_nodes(node.left)
            collect_nodes(node.right)

    collect_nodes(root)

    # Initialize node times: leaves at 0, internals spaced evenly
    node_times = {}
    for lid in leaf_ids:
        node_times[lid] = 0.0

    # Sort internal nodes by depth (shallowest = youngest first)
    # Use a simple heuristic: assign times based on tree depth
    def assign_initial_times(node, depth=0):
        if node.is_leaf:
            return
        assign_initial_times(node.left, depth + 1)
        assign_initial_times(node.right, depth + 1)
        # Deeper nodes are older
        max_child = max(node_times.get(node.left.id, 0),
                        node_times.get(node.right.id, 0))
        node_times[node.id] = max_child + N_e / 5  # rough spacing

    assign_initial_times(root)

    # Get parent/child relationships for constraint checking
    parent_of = {}
    children_of = {}

    def build_relationships(node):
        children_of[node.id] = []
        if not node.is_leaf:
            children_of[node.id] = [node.left.id, node.right.id]
            parent_of[node.left.id] = node.id
            parent_of[node.right.id] = node.id
            build_relationships(node.left)
            build_relationships(node.right)

    build_relationships(root)

    # Current log posterior
    current_lp = log_posterior(node_times, branch_mutations, mu, span,
                                N_e, internal_ids, leaf_ids)

    # MCMC loop
    samples = []
    n_accept = 0
    total_steps = burn_in + n_samples

    for step in range(total_steps):
        # Pick a random internal node to update
        target = rng.choice(internal_ids)

        # Propose new time
        old_time = node_times[target]
        new_time = old_time + rng.normal(0, sigma)

        # Check constraints: must be > all children, < parent (if exists)
        min_time = max(node_times[c] for c in children_of[target]) \
                   if children_of[target] else 0.0
        max_time = node_times[parent_of[target]] \
                   if target in parent_of else np.inf

        if new_time <= min_time or new_time >= max_time:
            continue  # reject: violates constraints

        # Compute proposed log posterior
        node_times[target] = new_time
        proposed_lp = log_posterior(node_times, branch_mutations, mu,
                                    span, N_e, internal_ids, leaf_ids)

        # Accept/reject
        log_alpha = proposed_lp - current_lp
        if np.log(rng.uniform()) < log_alpha:
            # Accept
            current_lp = proposed_lp
            n_accept += 1
        else:
            # Reject: revert
            node_times[target] = old_time

        # Collect sample (after burn-in)
        if step >= burn_in:
            samples.append(dict(node_times))

    acceptance_rate = n_accept / total_steps
    return samples, acceptance_rate


def posterior_summary(samples, node_id):
    """Summarize the posterior distribution for a node's time.

    Parameters
    ----------
    samples : list of dict
        MCMC samples.
    node_id : int
        Node to summarize.

    Returns
    -------
    dict
        Mean, median, std, and 95% credible interval.
    """
    times = np.array([s[node_id] for s in samples])
    return {
        'mean': np.mean(times),
        'median': np.median(times),
        'std': np.std(times),
        'ci_lower': np.percentile(times, 2.5),
        'ci_upper': np.percentile(times, 97.5),
    }


# ---------------------------------------------------------------------------
# Gear 4: Population Size Estimation
# ---------------------------------------------------------------------------

def make_epochs(max_time, n_epochs):
    """Create logarithmically spaced epoch boundaries.

    Parameters
    ----------
    max_time : float
        Maximum time (most ancient epoch boundary).
    n_epochs : int
        Number of epochs.

    Returns
    -------
    boundaries : ndarray of shape (n_epochs + 1,)
        Epoch boundaries [0, t_1, t_2, ..., max_time].
    """
    # Log-space between a small positive value and max_time
    boundaries = np.zeros(n_epochs + 1)
    boundaries[1:] = np.logspace(
        np.log10(max_time / n_epochs),
        np.log10(max_time),
        n_epochs
    )
    return boundaries


def integrated_rate(t_start, t_end, boundaries, N_e_values):
    """Compute the integrated inverse population size.

    Parameters
    ----------
    t_start, t_end : float
        Time interval [t_start, t_end).
    boundaries : ndarray
        Epoch boundaries.
    N_e_values : ndarray
        Population size in each epoch.

    Returns
    -------
    float
        Integral of 1/N_e(t) from t_start to t_end.
    """
    result = 0.0
    n_epochs = len(N_e_values)

    for j in range(n_epochs):
        epoch_start = boundaries[j]
        epoch_end = boundaries[j + 1]

        # Overlap of [t_start, t_end) with [epoch_start, epoch_end)
        overlap_start = max(t_start, epoch_start)
        overlap_end = min(t_end, epoch_end)

        if overlap_start < overlap_end:
            result += (overlap_end - overlap_start) / N_e_values[j]

    # Handle time beyond the last epoch boundary
    if t_end > boundaries[-1]:
        overlap_start = max(t_start, boundaries[-1])
        result += (t_end - overlap_start) / N_e_values[-1]

    return result


def log_coalescent_prior_variable(coalescence_times, boundaries, N_e_values):
    """Log coalescent prior with piecewise-constant population size.

    Parameters
    ----------
    coalescence_times : list of float
        Coalescence times sorted youngest to oldest.
    boundaries : ndarray
        Epoch boundaries.
    N_e_values : ndarray
        Population size in each epoch.

    Returns
    -------
    float
        Log prior probability.
    """
    n_coal = len(coalescence_times)
    N = n_coal + 1  # number of leaves

    log_prior = 0.0
    prev_time = 0.0

    for idx, t in enumerate(coalescence_times):
        k = N - idx  # number of lineages before this coalescence
        if k < 2:
            break

        coal_rate_k = k * (k - 1) / 2.0

        # Find which epoch t falls in to get instantaneous N_e
        epoch_idx = np.searchsorted(boundaries[1:], t)
        epoch_idx = min(epoch_idx, len(N_e_values) - 1)
        N_e_at_t = N_e_values[epoch_idx]

        # Instantaneous rate
        rate = coal_rate_k / N_e_at_t

        # Survival: integral of coal_rate_k / N_e(s) from prev_time to t
        integral = coal_rate_k * integrated_rate(
            prev_time, t, boundaries, N_e_values)

        # Log density: log(rate) - integral
        log_prior += np.log(rate) - integral
        prev_time = t

    return log_prior


def m_step(coalescence_times_per_tree, num_leaves_per_tree,
           boundaries, span_per_tree):
    """M-step: estimate population sizes from coalescence times.

    Parameters
    ----------
    coalescence_times_per_tree : list of list of float
        For each tree, the sorted coalescence times.
    num_leaves_per_tree : list of int
        Number of leaves in each tree.
    boundaries : ndarray
        Epoch boundaries.
    span_per_tree : list of float
        Genomic span of each tree (for weighting).

    Returns
    -------
    N_e_estimates : ndarray
        Estimated N_e for each epoch.
    """
    n_epochs = len(boundaries) - 1
    total_exposure = np.zeros(n_epochs)  # lineage-time at risk
    total_events = np.zeros(n_epochs)    # coalescence events

    for tree_idx, coal_times in enumerate(coalescence_times_per_tree):
        N = num_leaves_per_tree[tree_idx]
        weight = span_per_tree[tree_idx]
        prev_time = 0.0

        for idx, t in enumerate(coal_times):
            k = N - idx  # lineages before this coalescence
            if k < 2:
                break

            # Distribute exposure across epochs
            for j in range(n_epochs):
                ep_start = boundaries[j]
                ep_end = boundaries[j + 1]

                overlap_start = max(prev_time, ep_start)
                overlap_end = min(t, ep_end)

                if overlap_start < overlap_end:
                    dt = overlap_end - overlap_start
                    exposure = k * (k - 1) / 2.0 * dt * weight
                    total_exposure[j] += exposure

            # Record the coalescence event in the appropriate epoch
            event_epoch = np.searchsorted(boundaries[1:], t)
            event_epoch = min(event_epoch, n_epochs - 1)
            total_events[event_epoch] += weight

            prev_time = t

    # Estimate N_e: exposure / events (avoid division by zero)
    N_e_estimates = np.zeros(n_epochs)
    for j in range(n_epochs):
        if total_events[j] > 0:
            N_e_estimates[j] = total_exposure[j] / total_events[j]
        else:
            # No coalescence events in this epoch -- use neighbor
            N_e_estimates[j] = np.nan

    # Fill NaN epochs by interpolation
    valid = ~np.isnan(N_e_estimates)
    if valid.any():
        epoch_mids = (boundaries[:-1] + boundaries[1:]) / 2
        N_e_estimates[~valid] = np.interp(
            epoch_mids[~valid], epoch_mids[valid], N_e_estimates[valid])

    return N_e_estimates


def em_population_size(trees, haplotypes, mu, initial_N_e,
                        boundaries, n_em_iterations=10,
                        n_mcmc_samples=200):
    """Estimate population size history via EM.

    Parameters
    ----------
    trees : list of dict
        Local trees with topologies.
    haplotypes : ndarray
        Haplotype matrix.
    mu : float
        Mutation rate.
    initial_N_e : float
        Initial (constant) population size guess.
    boundaries : ndarray
        Epoch boundaries.
    n_em_iterations : int
        Number of EM iterations.
    n_mcmc_samples : int
        MCMC samples per tree per E-step.

    Returns
    -------
    N_e_history : list of ndarray
        Population size estimates at each EM iteration.
    """
    n_epochs = len(boundaries) - 1
    N_e_values = np.full(n_epochs, initial_N_e)
    N_e_history = [N_e_values.copy()]

    for em_iter in range(n_em_iterations):
        print(f"EM iteration {em_iter + 1}/{n_em_iterations}")

        # E-step: sample branch lengths using current N_e
        all_coal_times = []
        all_n_leaves = []
        all_spans = []

        for tree_info in trees:
            root = tree_info['root']
            span = tree_info['end'] - tree_info['start']
            site_indices = tree_info.get('site_indices', [])

            # Map mutations
            branch_muts, _ = map_mutations(root, haplotypes, site_indices)

            # Run MCMC with current population size
            # (simplified: using constant N_e equal to the mean)
            mean_N_e = np.mean(N_e_values)
            samples, _ = mcmc_branch_lengths(
                root, branch_muts, mu, span, mean_N_e,
                n_samples=n_mcmc_samples, burn_in=50)

            # Extract coalescence times from the last sample
            if samples:
                last_sample = samples[-1]
                # Get internal node times, sorted
                internal_times = sorted([
                    t for nid, t in last_sample.items()
                    if t > 0  # exclude leaves
                ])
                all_coal_times.append(internal_times)
                n_leaves = sum(1 for t in last_sample.values() if t == 0)
                all_n_leaves.append(n_leaves)
                all_spans.append(span)

        # M-step: update population sizes
        N_e_values = m_step(all_coal_times, all_n_leaves,
                             boundaries, all_spans)
        N_e_history.append(N_e_values.copy())

        print(f"  Mean N_e: {np.nanmean(N_e_values):.0f}")

    return N_e_history


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    """Demonstrate the Relate algorithm components."""
    print("=" * 60)
    print("Mini-Relate: Genome-Wide Genealogy Estimation")
    print("=" * 60)

    # --- Gear 1: Asymmetric Painting ---
    print("\n--- Gear 1: Asymmetric Painting ---")
    mu = 0.01
    e_d = directional_emission(1, 0, mu)  # target derived, ref ancestral
    e_a = directional_emission(0, 1, mu)  # target ancestral, ref derived
    print(f"P(target=1 | ref=0) = {e_d:.4f}")
    print(f"P(target=0 | ref=1) = {e_a:.4f}")
    print(f"Asymmetric? {e_d != e_a}")

    np.random.seed(42)
    K, L = 5, 20
    panel = np.random.binomial(1, 0.3, size=(L, K))
    target = np.random.binomial(1, 0.3, size=L)
    rho = np.full(L, 0.05)
    rho[0] = 0.0

    posterior = forward_backward_relate(target, panel, rho, mu=0.01)
    print(f"Posterior shape: {posterior.shape}")
    print(f"Posterior sums to 1 at each site: "
          f"{np.allclose(posterior.sum(axis=1), 1.0)}")
    print(f"Most likely copying source at site 0: {np.argmax(posterior[0])}")

    # Small distance matrix example
    np.random.seed(123)
    N_haps, L_sites = 6, 30
    haplotypes = np.random.binomial(1, 0.3, size=(N_haps, L_sites))
    positions = np.arange(L_sites, dtype=float) * 1000
    D = compute_distance_matrix(haplotypes, positions, recomb_rate=1e-4,
                                 mu=0.01, focal_snp=15)
    print(f"\nAsymmetric distance matrix (6x6):")
    print(np.round(D, 2))
    print(f"D[0,1] = {D[0,1]:.2f}, D[1,0] = {D[1,0]:.2f}")
    print(f"Asymmetric? {not np.allclose(D, D.T)}")

    # --- Gear 2: Tree Building ---
    print("\n--- Gear 2: Tree Building ---")
    D_example = np.array([
        [0.0, 1.2, 3.5, 3.8],
        [0.8, 0.0, 3.2, 3.5],
        [3.5, 3.2, 0.0, 0.5],
        [3.8, 3.5, 0.7, 0.0],
    ])

    root, merges = build_tree(D_example, N=4)
    print("Merge order:")
    for c1, c2, parent in merges:
        print(f"  Merge {c1} + {c2} -> {parent}")
    print(f"Root: {root}")
    print(f"Root leaves: {root.leaf_ids}")

    newick = to_newick(root) + ";"
    print(f"Newick: {newick}")

    # --- Gear 3: Branch Lengths ---
    print("\n--- Gear 3: Branch Length Estimation ---")

    # Build a simple tree: ((0,1),2),3)
    leaf0 = TreeNode(0)
    leaf1 = TreeNode(1)
    leaf2 = TreeNode(2)
    leaf3 = TreeNode(3)
    node4 = TreeNode(4, left=leaf0, right=leaf1, is_leaf=False)
    node4.leaf_ids = {0, 1}
    node5 = TreeNode(5, left=node4, right=leaf2, is_leaf=False)
    node5.leaf_ids = {0, 1, 2}
    root_tree = TreeNode(6, left=node5, right=leaf3, is_leaf=False)
    root_tree.leaf_ids = {0, 1, 2, 3}

    haps = np.array([
        [1, 1, 1, 0],
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ])

    branch_muts, n_unmapped = map_mutations(root_tree, haps, list(range(4)))
    print("Mutation mapping:")
    for (p, c), count in sorted(branch_muts.items()):
        if count > 0:
            print(f"  Branch ({p} -> {c}): {count} mutation(s)")
    print(f"  Unmapped: {n_unmapped}")

    node_times = {0: 0, 1: 0, 2: 0, 3: 0, 4: 100, 5: 300, 6: 500}
    log_lik = log_mutation_likelihood(branch_muts, node_times,
                                       mu=1.25e-8, span=1e4)
    print(f"Log likelihood: {log_lik:.2f}")

    coal_times = [100, 300, 500]
    log_prior = log_coalescent_prior(coal_times, N_e=10000)
    print(f"Log coalescent prior: {log_prior:.2f}")

    # Run MCMC
    samples, acc_rate = mcmc_branch_lengths(
        root_tree, branch_muts, mu=1.25e-8, span=1e4, N_e=10000,
        n_samples=500, burn_in=200, sigma=100.0)
    print(f"\nMCMC acceptance rate: {acc_rate:.1%}")
    for nid in [4, 5, 6]:
        summary = posterior_summary(samples, nid)
        print(f"  Node {nid}: mean={summary['mean']:.0f} "
              f"[{summary['ci_lower']:.0f}, {summary['ci_upper']:.0f}]")

    # --- Gear 4: Population Size Estimation ---
    print("\n--- Gear 4: Population Size Estimation ---")
    boundaries = make_epochs(100_000, n_epochs=20)
    print(f"Epoch boundaries (first 5): {boundaries[:5].astype(int)}")

    np.random.seed(42)
    true_N_e = 10000
    n_trees = 100
    n_leaves = 10

    coal_times_all = []
    for _ in range(n_trees):
        times = []
        prev_t = 0.0
        for k in range(n_leaves, 1, -1):
            rate = k * (k - 1) / (2.0 * true_N_e)
            dt = np.random.exponential(1.0 / rate)
            prev_t += dt
            times.append(prev_t)
        coal_times_all.append(times)

    boundaries_em = make_epochs(50_000, n_epochs=10)
    spans = np.full(n_trees, 1e4)

    N_e_est = m_step(coal_times_all, [n_leaves] * n_trees,
                       boundaries_em, spans)
    print(f"True N_e: {true_N_e}")
    print(f"Mean estimated N_e: {np.nanmean(N_e_est):.0f}")

    print("\n" + "=" * 60)
    print("Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    demo()
