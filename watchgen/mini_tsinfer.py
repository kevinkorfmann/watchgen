"""
Mini-implementation of the tsinfer algorithm for tree sequence inference.

tsinfer infers a tree sequence from observed genetic variation data. Unlike
MCMC-based methods (such as SINGER and ARGweaver), tsinfer is a deterministic
algorithm that scales to biobank-sized datasets -- hundreds of thousands of
samples and millions of sites -- in hours rather than weeks.

The core idea is that every sample's genome is a mosaic of ancestral
haplotypes, glued together by recombination. The algorithm identifies what
those ancestral pieces are and how they were assembled, thereby reconstructing
the genealogy.

The four gears of tsinfer:

1. Ancestor Generation (Gear 1) -- Infer putative ancestral haplotypes from
   the patterns of derived alleles in the data. Older ancestors carry
   higher-frequency derived alleles.

2. The Copying Model (Gear 2) -- A Li & Stephens HMM engine that finds the
   best way to express one haplotype as a mosaic of others. This is the
   workhorse shared by both the ancestor matching and sample matching phases.

3. Ancestor Matching (Gear 3) -- Match each ancestor against older ancestors
   using the copying model, building a tree sequence of ancestors from the
   root down.

4. Sample Matching (Gear 4) -- Thread each sample through the ancestor tree
   using the same copying model, then post-process to produce the final tree
   sequence.

References
----------
Kelleher, J., Wong, Y., Wohns, A.W. et al. (2019). Inferring whole-genome
histories in large population datasets. Nature Genetics, 51, 1330-1338.
"""

import numpy as np
from collections import defaultdict


# ============================================================================
# Constants
# ============================================================================

NONCOPY = -2
PC_TIME_EPSILON = 1.0 / (2**32)


# ============================================================================
# Gear 1: Ancestor Generation (from ancestor_generation.rst)
# ============================================================================

def select_inference_sites(D, ancestral_known):
    """Select sites suitable for tree inference.

    Parameters
    ----------
    D : ndarray of shape (n, m)
        Variant matrix (0 = ancestral, 1 = derived).
    ancestral_known : ndarray of shape (m,), dtype=bool
        Whether the ancestral allele is known at each site.

    Returns
    -------
    inference_sites : ndarray of int
        Indices of sites that qualify for inference.
    non_inference_sites : ndarray of int
        Indices of sites excluded from inference.
    """
    n, m = D.shape
    is_inference = np.zeros(m, dtype=bool)

    for j in range(m):
        # Skip sites where we don't know which allele is ancestral
        if not ancestral_known[j]:
            continue

        # Count how many samples carry the derived allele (1)
        derived_count = D[:, j].sum()
        # The rest carry the ancestral allele (0)
        ancestral_count = n - derived_count

        # Check all four criteria:
        # - biallelic (exactly 2 distinct values observed)
        # - at least 2 derived copies (no singletons)
        # - at least 1 ancestral copy (not fixed for derived)
        num_alleles = len(np.unique(D[:, j]))
        if (num_alleles == 2 and
                derived_count >= 2 and
                ancestral_count >= 1):
            is_inference[j] = True

    inference_sites = np.where(is_inference)[0]
    non_inference_sites = np.where(~is_inference)[0]
    return inference_sites, non_inference_sites


def compute_ancestor_times(D, inference_sites):
    """Compute time proxy for each inference site.

    Parameters
    ----------
    D : ndarray of shape (n, m)
        Variant matrix.
    inference_sites : ndarray of int
        Indices of inference sites.

    Returns
    -------
    times : ndarray of float
        Time proxy for each inference site (= derived allele frequency).
    """
    n = D.shape[0]
    times = np.zeros(len(inference_sites))
    for k, j in enumerate(inference_sites):
        # Count non-missing entries (in case of missing data)
        non_missing = np.sum(D[:, j] >= 0)
        # Count derived alleles
        derived = np.sum(D[:, j] == 1)
        # Time proxy = derived allele frequency
        times[k] = derived / non_missing
    return times


def get_focal_samples(D, site_index):
    """Get the samples carrying the derived allele at a site.

    Parameters
    ----------
    D : ndarray of shape (n, m)
        Variant matrix.
    site_index : int
        The site index.

    Returns
    -------
    focal : ndarray of int
        Indices of samples carrying allele 1 (the derived allele).
    """
    return np.where(D[:, site_index] == 1)[0]


def build_ancestor(D, inference_sites, times, focal_site_idx):
    """Build an ancestral haplotype by extending from a focal site.

    Parameters
    ----------
    D : ndarray of shape (n, m)
        Variant matrix.
    inference_sites : ndarray of int
        Sorted array of inference site positions.
    times : ndarray of float
        Time proxy for each inference site.
    focal_site_idx : int
        Index into inference_sites (not the site position!).

    Returns
    -------
    ancestor : dict
        'haplotype': allelic states at each inference site in [start, end]
        'start': leftmost inference site index (inclusive)
        'end': rightmost inference site index (exclusive)
        'focal': the focal inference site index
        'time': time proxy
    """
    n_inf = len(inference_sites)
    focal_j = inference_sites[focal_site_idx]
    focal_time = times[focal_site_idx]
    # The focal samples: everyone carrying derived allele at the focal site
    focal_samples = get_focal_samples(D, focal_j)

    # The ancestor's haplotype (over inference sites)
    # -1 = not yet defined; will be filled in as we extend
    haplotype = np.full(n_inf, -1, dtype=int)
    # At the focal site itself, the ancestor carries the derived allele
    haplotype[focal_site_idx] = 1

    # --- Extend leftward ---
    start = focal_site_idx
    for k in range(focal_site_idx - 1, -1, -1):
        site_k = inference_sites[k]

        # Stop if we hit an older site (higher frequency = older)
        if times[k] > focal_time:
            # At this older site, our ancestor carries the ancestral allele
            haplotype[k] = 0
            start = k
            break

        # Consensus vote among focal samples at this site
        alleles = D[focal_samples, site_k]
        ones = np.sum(alleles == 1)
        zeros = np.sum(alleles == 0)

        # Majority wins: if tied, prefer derived (1)
        if ones >= zeros:
            haplotype[k] = 1
        else:
            haplotype[k] = 0

        start = k

    # --- Extend rightward ---
    end = focal_site_idx + 1
    for k in range(focal_site_idx + 1, n_inf):
        site_k = inference_sites[k]

        # Stop if we hit an older site
        if times[k] > focal_time:
            haplotype[k] = 0
            end = k + 1
            break

        # Consensus vote
        alleles = D[focal_samples, site_k]
        ones = np.sum(alleles == 1)
        zeros = np.sum(alleles == 0)

        if ones >= zeros:
            haplotype[k] = 1
        else:
            haplotype[k] = 0

        end = k + 1

    return {
        'haplotype': haplotype[start:end],
        'start': start,
        'end': end,
        'focal': focal_site_idx,
        'time': focal_time,
    }


def generate_ancestors(D, ancestral_known):
    """Generate all putative ancestors from variant data.

    Parameters
    ----------
    D : ndarray of shape (n, m)
        Variant matrix.
    ancestral_known : ndarray of shape (m,), dtype=bool
        Whether the ancestral allele is known at each site.

    Returns
    -------
    ancestors : list of dict
        Each ancestor has 'haplotype', 'start', 'end', 'focal', 'time'.
    inference_sites : ndarray of int
        The inference site indices.
    """
    # First, select which sites will participate in inference
    inference_sites, _ = select_inference_sites(D, ancestral_known)
    # Assign a time (= derived allele frequency) to each inference site
    times = compute_ancestor_times(D, inference_sites)

    ancestors = []
    for idx in range(len(inference_sites)):
        # Build one ancestor per inference site
        anc = build_ancestor(D, inference_sites, times, idx)
        ancestors.append(anc)

    # Sort by time (oldest = highest frequency first)
    # This ordering is critical: during matching, older ancestors
    # must be placed before younger ones.
    ancestors.sort(key=lambda a: -a['time'])

    return ancestors, inference_sites


def group_ancestors_by_time(ancestors):
    """Group ancestors by their time proxy.

    Parameters
    ----------
    ancestors : list of dict
        Ancestors sorted by time (oldest first).

    Returns
    -------
    groups : list of (time, list_of_ancestors)
        Groups sorted by time (oldest first).
    """
    groups = defaultdict(list)
    for anc in ancestors:
        # Group by exact frequency value
        groups[anc['time']].append(anc)

    # Sort by time (descending = oldest first)
    sorted_groups = sorted(groups.items(), key=lambda x: -x[0])
    return sorted_groups


def add_ultimate_ancestor(ancestors, num_inference_sites):
    """Add the ultimate (root) ancestor.

    Parameters
    ----------
    ancestors : list of dict
        Existing ancestors.
    num_inference_sites : int
        Total number of inference sites.

    Returns
    -------
    ancestors : list of dict
        Updated list with the ultimate ancestor prepended.
    """
    ultimate = {
        # All-zero haplotype: ancestral allele at every site
        'haplotype': np.zeros(num_inference_sites, dtype=int),
        'start': 0,
        'end': num_inference_sites,
        'focal': -1,  # No focal site -- this is a virtual ancestor
        'time': 1.0,  # Oldest possible time
    }
    # Prepend so it appears first (oldest) in the sorted list
    return [ultimate] + ancestors


def verify_ancestors(ancestors, D, inference_sites):
    """Verify correctness of generated ancestors."""
    n, m = D.shape
    n_inf = len(inference_sites)

    print("Verification checks:")

    # 1. Each ancestor's time is in (0, 1]
    times = [a['time'] for a in ancestors]
    assert all(0 < t <= 1.0 for t in times), "Times out of range!"
    print(f"  [ok] All times in (0, 1]: min={min(times):.3f}, "
          f"max={max(times):.3f}")

    # 2. Ancestors are sorted by time (oldest first)
    for i in range(len(ancestors) - 1):
        assert ancestors[i]['time'] >= ancestors[i + 1]['time'], \
            "Ancestors not sorted!"
    print(f"  [ok] Ancestors sorted by time (oldest first)")

    # 3. Each ancestor carries the derived allele at its focal site
    for anc in ancestors:
        if anc['focal'] >= 0:  # Skip ultimate ancestor
            focal_in_haplotype = anc['focal'] - anc['start']
            assert anc['haplotype'][focal_in_haplotype] == 1, \
                "Focal site should carry derived allele!"
    print(f"  [ok] All ancestors carry derived allele at focal site")

    # 4. Haplotypes contain only 0s and 1s
    for anc in ancestors:
        assert set(anc['haplotype']).issubset({0, 1}), \
            "Invalid allele!"
    print(f"  [ok] All haplotypes contain only 0s and 1s")

    print(f"\nAll checks passed for {len(ancestors)} ancestors.")


# ============================================================================
# Gear 2: The Copying Model (from copying_model.rst)
# ============================================================================

def compute_recombination_probs(positions, recombination_rate, num_ref):
    """Compute per-site recombination probabilities.

    Parameters
    ----------
    positions : ndarray of float
        Genomic positions of each site (sorted).
    recombination_rate : float
        Per-unit recombination rate.
    num_ref : int
        Number of reference haplotypes (k).

    Returns
    -------
    rho : ndarray of float
        Recombination probability at each site (rho[0] = 0).
    """
    m = len(positions)
    rho = np.zeros(m)
    for ell in range(1, m):
        # Genetic distance between adjacent sites
        d = positions[ell] - positions[ell - 1]
        # Li & Stephens recombination probability with 1/k scaling
        rho[ell] = 1 - np.exp(-d * recombination_rate / num_ref)
    return rho


def compute_mismatch_probs(positions, recombination_rate, mismatch_ratio,
                            num_ref):
    """Compute per-site mismatch probabilities.

    Parameters
    ----------
    positions : ndarray of float
        Genomic positions of each site.
    recombination_rate : float
        Per-unit recombination rate.
    mismatch_ratio : float
        Ratio of mismatch to recombination rate.
    num_ref : int
        Number of reference haplotypes (k).

    Returns
    -------
    mu : ndarray of float
        Mismatch probability at each site.
    """
    m = len(positions)
    mu = np.zeros(m)
    for ell in range(1, m):
        d = positions[ell] - positions[ell - 1]
        # Mismatch probability: same formula as rho, scaled by ratio
        mu[ell] = 1 - np.exp(-d * recombination_rate * mismatch_ratio
                              / num_ref)
    # First site: use a small default (no "previous" site to compute from)
    mu[0] = mu[1] if m > 1 else 1e-6
    return mu


def viterbi_ls(query, panel, rho, mu):
    """Viterbi algorithm for the Li & Stephens model.

    Parameters
    ----------
    query : ndarray of shape (m,)
        Query haplotype (0/1 at each site).
    panel : ndarray of shape (m, k)
        Reference panel (m sites, k haplotypes).
    rho : ndarray of shape (m,)
        Per-site recombination probabilities.
    mu : ndarray of shape (m,)
        Per-site mismatch probabilities.

    Returns
    -------
    path : ndarray of shape (m,)
        Most likely copying path (index into panel columns).
    log_prob : float
        Log probability of the Viterbi path.
    """
    m, k = panel.shape
    # V[ell, j] = probability of best path ending in state j at site ell
    V = np.zeros((m, k))
    # psi[ell, j] = which state at site ell-1 led to the max at (ell, j)
    psi = np.zeros((m, k), dtype=int)  # Traceback pointers

    # --- Initialization (site 0) ---
    for j in range(k):
        # Uniform prior 1/k, times emission probability
        if query[0] == panel[0, j]:
            V[0, j] = (1.0 / k) * (1 - mu[0])  # Match
        else:
            V[0, j] = (1.0 / k) * mu[0]         # Mismatch

    # --- Recursion (sites 1 through m-1) ---
    for ell in range(1, m):
        # O(k) trick: compute the global max of previous Viterbi values
        max_prev = np.max(V[ell - 1])
        argmax_prev = np.argmax(V[ell - 1])

        for j in range(k):
            # Emission probability at this site for this reference
            if query[ell] == panel[ell, j]:
                e = 1 - mu[ell]   # Query matches reference: high prob
            else:
                e = mu[ell]       # Mismatch: low prob

            # Two candidates for the best previous state:
            stay = (1 - rho[ell]) * V[ell - 1, j]       # Stay on j
            switch = (rho[ell] / k) * max_prev           # Switch from best

            if stay >= switch:
                V[ell, j] = e * stay
                psi[ell, j] = j  # Stayed on j
            else:
                V[ell, j] = e * switch
                psi[ell, j] = argmax_prev  # Switched from global best

        # Rescale to prevent underflow (divide by max value)
        scale = np.max(V[ell])
        if scale > 0:
            V[ell] /= scale

    # --- Traceback: follow pointers from the best final state ---
    path = np.zeros(m, dtype=int)
    path[-1] = np.argmax(V[-1])  # Start from the best state at last site

    for ell in range(m - 2, -1, -1):
        # The pointer at site ell+1 tells us which state at site ell
        path[ell] = psi[ell + 1, path[ell + 1]]

    log_prob = np.sum(np.log(np.max(V, axis=1) + 1e-300))
    return path, log_prob


def viterbi_ls_with_noncopy(query, panel, rho, mu):
    """Viterbi algorithm handling NONCOPY entries.

    Parameters
    ----------
    query : ndarray of shape (m,)
        Query haplotype.
    panel : ndarray of shape (m, k)
        Reference panel. Entries equal to NONCOPY (-2) are non-copiable.
    rho : ndarray of shape (m,)
        Per-site recombination probabilities.
    mu : ndarray of shape (m,)
        Per-site mismatch probabilities.

    Returns
    -------
    path : ndarray of shape (m,)
        Most likely copying path.
    """
    m, k = panel.shape
    V = np.zeros((m, k))
    psi = np.zeros((m, k), dtype=int)

    # --- Initialization ---
    # Only initialize copiable references at site 0
    copiable_0 = [j for j in range(k) if panel[0, j] != NONCOPY]
    k_0 = len(copiable_0)
    for j in range(k):
        if panel[0, j] == NONCOPY:
            V[0, j] = 0  # Cannot copy from this reference at site 0
        elif query[0] == panel[0, j]:
            V[0, j] = (1.0 / k_0) * (1 - mu[0])
        else:
            V[0, j] = (1.0 / k_0) * mu[0]

    # --- Recursion ---
    for ell in range(1, m):
        # Count how many references are copiable at this site
        copiable = [j for j in range(k) if panel[ell, j] != NONCOPY]
        k_ell = len(copiable)
        if k_ell == 0:
            continue  # No references available -- skip this site

        # Global max of previous Viterbi values
        max_prev = np.max(V[ell - 1])
        argmax_prev = np.argmax(V[ell - 1])

        for j in range(k):
            if panel[ell, j] == NONCOPY:
                # This reference doesn't exist at this site
                V[ell, j] = 0
                psi[ell, j] = j
                continue

            # Emission: match vs mismatch
            if query[ell] == panel[ell, j]:
                e = 1 - mu[ell]
            else:
                e = mu[ell]

            # Two candidates, using site-specific panel size k_ell
            stay = (1 - rho[ell]) * V[ell - 1, j]
            switch = (rho[ell] / k_ell) * max_prev

            if stay >= switch:
                V[ell, j] = e * stay
                psi[ell, j] = j
            else:
                V[ell, j] = e * switch
                psi[ell, j] = argmax_prev

        # Rescale to prevent underflow
        scale = np.max(V[ell])
        if scale > 0:
            V[ell] /= scale

    # --- Traceback ---
    path = np.zeros(m, dtype=int)
    path[-1] = np.argmax(V[-1])

    for ell in range(m - 2, -1, -1):
        path[ell] = psi[ell + 1, path[ell + 1]]

    return path


def path_to_edges(path, positions, child_id, ref_node_ids):
    """Convert a Viterbi path to tree sequence edges.

    Parameters
    ----------
    path : ndarray of shape (m,)
        Copying path (index into reference panel).
    positions : ndarray of float
        Genomic positions of each site.
    child_id : int
        Node ID of the query haplotype.
    ref_node_ids : ndarray of int
        Node IDs corresponding to each reference index.

    Returns
    -------
    edges : list of (left, right, parent, child)
        Tree sequence edges.
    """
    edges = []
    m = len(path)

    # Walk through the path, merging consecutive identical segments
    seg_start = 0
    current_ref = path[0]

    for ell in range(1, m):
        if path[ell] != current_ref:
            # The copying source changed -- emit an edge for the old segment
            left = positions[seg_start]
            right = positions[ell]  # Exclusive right boundary
            parent = ref_node_ids[current_ref]
            edges.append((left, right, parent, child_id))

            # Start new segment
            seg_start = ell
            current_ref = path[ell]

    # Emit final segment (extends to the end of the sequence)
    left = positions[seg_start]
    right = positions[-1] + 1  # Or sequence_length
    parent = ref_node_ids[current_ref]
    edges.append((left, right, parent, child_id))

    return edges


def find_breakpoints(path, positions):
    """Find recombination breakpoints from a Viterbi path.

    Parameters
    ----------
    path : ndarray of shape (m,)
        Copying path.
    positions : ndarray of float
        Genomic positions.

    Returns
    -------
    breakpoints : list of (position, from_ref, to_ref)
    """
    breakpoints = []
    for ell in range(1, len(path)):
        if path[ell] != path[ell - 1]:
            breakpoints.append((
                positions[ell],
                path[ell - 1],  # Which reference we were copying from
                path[ell]       # Which reference we switch to
            ))
    return breakpoints


# ============================================================================
# Gear 3: Ancestor Matching (from ancestor_matching.rst)
# ============================================================================

def matching_order(ancestors):
    """Determine the order for ancestor matching.

    Parameters
    ----------
    ancestors : list of dict
        Ancestors with 'time' field, sorted oldest first.

    Returns
    -------
    groups : list of list of dict
        Groups of ancestors at the same time.
    """
    groups = []
    current_time = None
    current_group = []

    for anc in ancestors:
        if anc['time'] != current_time:
            # New time group -- flush the previous group
            if current_group:
                groups.append(current_group)
            current_group = [anc]
            current_time = anc['time']
        else:
            # Same time -- add to current group
            current_group.append(anc)

    if current_group:
        groups.append(current_group)

    return groups


def build_reference_panel(placed_ancestors, num_inference_sites):
    """Build the reference panel from already-placed ancestors.

    Parameters
    ----------
    placed_ancestors : list of dict
        Ancestors already in the tree sequence.
    num_inference_sites : int
        Total number of inference sites.

    Returns
    -------
    panel : ndarray of shape (num_inference_sites, k)
        Reference panel with NONCOPY entries.
    node_ids : list of int
        Node ID for each column of the panel.
    """
    k = len(placed_ancestors)
    # Initialize everything as NONCOPY (not available for copying)
    panel = np.full((num_inference_sites, k), NONCOPY, dtype=int)

    for col, anc in enumerate(placed_ancestors):
        start = anc['start']
        end = anc['end']
        # Fill in the ancestor's haplotype where it is defined
        panel[start:end, col] = anc['haplotype']

    node_ids = [anc.get('node_id', col) for col, anc in
                enumerate(placed_ancestors)]
    return panel, node_ids


def path_compress(edges, nodes):
    """Apply path compression to a set of edges.

    Parameters
    ----------
    edges : list of (left, right, parent, child)
        Raw edges from matching.
    nodes : list of dict
        Node information.

    Returns
    -------
    new_edges : list of (left, right, parent, child)
        Compressed edges.
    new_nodes : list of dict
        Updated nodes (may include new PC nodes).
    """
    # Group edges by (left, right, parent) to find shared patterns
    groups = defaultdict(list)
    for left, right, parent, child in edges:
        groups[(left, right, parent)].append(child)

    new_edges = []
    new_nodes = list(nodes)
    next_id = max(n['id'] for n in nodes) + 1

    for (left, right, parent), children in groups.items():
        if len(children) <= 1:
            # Only one child -- no compression needed
            for child in children:
                new_edges.append((left, right, parent, child))
        else:
            # Multiple children share the same parent and interval
            # Insert a PC node between parent and children
            parent_time = None
            for n in nodes:
                if n['id'] == parent:
                    parent_time = n['time']
                    break

            # PC node sits just below the parent in time
            pc_time = parent_time - PC_TIME_EPSILON
            pc_node = {'id': next_id, 'time': pc_time,
                       'is_sample': False}
            new_nodes.append(pc_node)

            # Parent -> PC node (single edge replaces multiple)
            new_edges.append((left, right, parent, next_id))

            # PC node -> each child
            for child in children:
                new_edges.append((left, right, next_id, child))

            next_id += 1

    return new_edges, new_nodes


class TreeSequenceBuilder:
    """Incrementally builds a tree sequence from matching results.

    This is a simplified version of tsinfer's internal builder.
    It accumulates nodes and edges as ancestors and samples are matched.
    """

    def __init__(self, sequence_length, num_inference_sites, positions):
        self.sequence_length = sequence_length
        self.positions = positions
        self.num_inference_sites = num_inference_sites
        self.nodes = []   # (time, is_sample)
        self.edges = []   # (left, right, parent, child)
        self.next_id = 0

    def add_node(self, time, is_sample=False):
        """Add a node and return its ID."""
        node_id = self.next_id
        self.nodes.append({'id': node_id, 'time': time,
                           'is_sample': is_sample})
        self.next_id += 1
        return node_id

    def add_edges_from_path(self, path, child_id, ref_node_ids):
        """Convert a Viterbi path to edges and add them.

        Parameters
        ----------
        path : ndarray of shape (m,)
            Viterbi path (index into reference panel).
        child_id : int
            Node ID of the child.
        ref_node_ids : list of int
            Node IDs for each reference index.
        """
        m = len(path)
        # Walk through the path, emitting one edge per contiguous segment
        seg_start = 0
        current_ref = path[0]

        for ell in range(1, m):
            if path[ell] != current_ref:
                # Copying source changed -- emit edge for old segment
                left = self.positions[seg_start]
                right = self.positions[ell]
                parent = ref_node_ids[current_ref]
                self.edges.append((left, right, parent, child_id))
                seg_start = ell
                current_ref = path[ell]

        # Final segment extends to end of sequence
        left = self.positions[seg_start]
        right = self.positions[m - 1] + 1  # Or sequence_length
        parent = ref_node_ids[current_ref]
        self.edges.append((left, right, parent, child_id))

    def summary(self):
        """Print a summary of the tree sequence."""
        print(f"Nodes: {len(self.nodes)}")
        print(f"Edges: {len(self.edges)}")
        samples = sum(1 for n in self.nodes if n['is_sample'])
        print(f"Samples: {samples}")


def match_ancestors(ancestors, inference_sites, positions,
                    recombination_rate, mismatch_ratio,
                    sequence_length):
    """Run the complete ancestor matching phase.

    Parameters
    ----------
    ancestors : list of dict
        Ancestors sorted by time (oldest first). First is the ultimate
        ancestor.
    inference_sites : ndarray of int
        Inference site positions.
    positions : ndarray of float
        Genomic positions of inference sites.
    recombination_rate : float
        Per-unit recombination rate.
    mismatch_ratio : float
        Mismatch-to-recombination ratio.
    sequence_length : float
        Total sequence length.

    Returns
    -------
    builder : TreeSequenceBuilder
        The constructed tree sequence.
    """
    m = len(inference_sites)
    builder = TreeSequenceBuilder(sequence_length, m, positions)

    # Phase 1: Add the ultimate ancestor as root (the mainplate)
    ultimate = ancestors[0]
    root_id = builder.add_node(time=ultimate['time'])
    ultimate['node_id'] = root_id

    placed = [ultimate]

    # Phase 2: Process remaining ancestors by time groups
    groups = matching_order(ancestors[1:])

    for group_idx, group in enumerate(groups):
        # Build reference panel from all placed (older) ancestors
        panel, ref_node_ids = build_reference_panel(placed, m)
        k = len(ref_node_ids)

        # Compute HMM parameters for this panel size
        rho = np.zeros(m)
        mu = np.zeros(m)
        for ell in range(1, m):
            d = positions[ell] - positions[ell - 1]
            rho[ell] = 1 - np.exp(-d * recombination_rate / max(k, 1))
            mu[ell] = 1 - np.exp(-d * recombination_rate * mismatch_ratio
                                  / max(k, 1))
        mu[0] = mu[1] if m > 1 else 1e-6

        # Match each ancestor in this group against the panel
        for anc in group:
            node_id = builder.add_node(time=anc['time'])
            anc['node_id'] = node_id

            # Build the query (ancestor's haplotype over its interval)
            query = np.full(m, -1, dtype=int)  # -1 = undefined
            query[anc['start']:anc['end']] = anc['haplotype']

            # Run Viterbi (only over the ancestor's interval)
            start, end = anc['start'], anc['end']
            if end - start < 2:
                # Too short for HMM -- just parent to root
                left = positions[start]
                right = positions[end - 1] + 1
                builder.edges.append((left, right, root_id, node_id))
            else:
                sub_query = query[start:end]
                sub_panel = panel[start:end]
                sub_rho = rho[start:end]
                sub_mu = mu[start:end]
                sub_rho[0] = 0.0  # No recombination at first site

                # Only use columns that have copiable entries
                copiable_cols = []
                for col in range(k):
                    if np.any(sub_panel[:, col] != NONCOPY):
                        copiable_cols.append(col)

                if len(copiable_cols) == 0:
                    # No references available -- parent to root
                    left = positions[start]
                    right = positions[end - 1] + 1
                    builder.edges.append((left, right, root_id, node_id))
                else:
                    sub_panel_c = sub_panel[:, copiable_cols]
                    sub_ref_ids = [ref_node_ids[c] for c in copiable_cols]
                    # Use viterbi_ls_with_noncopy from Gear 2
                    path = viterbi_ls_with_noncopy(
                        sub_query, sub_panel_c, sub_rho, sub_mu)
                    # Map path back to node IDs
                    mapped_path = np.array([copiable_cols[p]
                                            for p in path])
                    builder.add_edges_from_path(
                        mapped_path, node_id,
                        ref_node_ids=ref_node_ids)

            # This ancestor is now placed and available for future groups
            placed.append(anc)

        print(f"  Group {group_idx}: time={group[0]['time']:.2f}, "
              f"matched {len(group)} ancestors, "
              f"panel size={k}")

    return builder


def verify_ancestor_tree(builder):
    """Verify basic properties of the ancestor tree sequence."""
    print("Ancestor tree verification:")

    # 1. Every non-root node has at least one parent edge
    root_id = 0  # Ultimate ancestor
    children_seen = set()
    for left, right, parent, child in builder.edges:
        children_seen.add(child)
    non_root_nodes = {n['id'] for n in builder.nodes if n['id'] != root_id}
    orphans = non_root_nodes - children_seen
    print(f"  [{'ok' if len(orphans) == 0 else 'FAIL'}] "
          f"All non-root nodes have parent edges "
          f"(orphans: {len(orphans)})")

    # 2. No self-loops (a node cannot be its own parent)
    self_loops = [(l, r, p, c) for l, r, p, c in builder.edges
                  if p == c]
    print(f"  [{'ok' if len(self_loops) == 0 else 'FAIL'}] "
          f"No self-loops")

    # 3. Parent time > child time for all edges
    time_map = {n['id']: n['time'] for n in builder.nodes}
    bad_times = []
    for left, right, parent, child in builder.edges:
        if time_map.get(parent, 0) <= time_map.get(child, 0):
            bad_times.append((parent, child))
    print(f"  [{'ok' if len(bad_times) == 0 else 'FAIL'}] "
          f"Parent time > child time for all edges "
          f"(violations: {len(bad_times)})")

    # 4. Summary statistics
    print(f"\n  Nodes: {len(builder.nodes)}")
    print(f"  Edges: {len(builder.edges)}")
    print(f"  Time range: [{min(time_map.values()):.4f}, "
          f"{max(time_map.values()):.4f}]")


# ============================================================================
# Gear 4: Sample Matching & Post-Processing (from sample_matching.rst)
# ============================================================================

def match_samples(samples, ancestors, inference_sites, positions,
                  recombination_rate, mismatch_ratio, builder):
    """Match all samples against the ancestor tree.

    Parameters
    ----------
    samples : ndarray of shape (n, m_inf)
        Sample genotypes at inference sites only.
    ancestors : list of dict
        All ancestors (with 'node_id' assigned during ancestor matching).
    inference_sites : ndarray of int
        Inference site indices.
    positions : ndarray of float
        Genomic positions of inference sites.
    recombination_rate : float
        Per-unit recombination rate.
    mismatch_ratio : float
        Mismatch-to-recombination ratio.
    builder : TreeSequenceBuilder
        The tree sequence builder (already contains ancestor nodes/edges).

    Returns
    -------
    builder : TreeSequenceBuilder
        Updated builder with sample nodes and edges.
    """
    n, m = samples.shape
    k = len(ancestors)  # Total number of ancestors in the panel

    # Build the full ancestor panel (all ancestors at once)
    panel = np.full((m, k), NONCOPY, dtype=int)
    ref_node_ids = []
    for col, anc in enumerate(ancestors):
        # Fill in each ancestor's haplotype where it is defined
        panel[anc['start']:anc['end'], col] = anc['haplotype']
        ref_node_ids.append(anc['node_id'])

    # HMM parameters (fixed for all samples, since panel doesn't change)
    rho = np.zeros(m)
    mu = np.zeros(m)
    for ell in range(1, m):
        d = positions[ell] - positions[ell - 1]
        # Recombination and mismatch probabilities scale with 1/k
        rho[ell] = 1 - np.exp(-d * recombination_rate / max(k, 1))
        mu[ell] = 1 - np.exp(-d * recombination_rate * mismatch_ratio
                              / max(k, 1))
    mu[0] = mu[1] if m > 1 else 1e-6

    # Match each sample independently
    for i in range(n):
        # Samples are at time 0.0 (the present)
        node_id = builder.add_node(time=0.0, is_sample=True)
        query = samples[i]  # This sample's genotype at inference sites

        # Run Viterbi against the full ancestor panel
        path = viterbi_ls_with_noncopy(query, panel, rho, mu)

        # Convert the Viterbi path to tree sequence edges
        builder.add_edges_from_path(path, node_id,
                                     ref_node_ids=ref_node_ids)

        if (i + 1) % 100 == 0 or i == n - 1:
            print(f"  Matched sample {i + 1}/{n}")

    return builder


def fitch_parsimony(tree_parent, tree_children, leaf_alleles, root):
    """Place mutations by Fitch parsimony on a single tree.

    Parameters
    ----------
    tree_parent : dict
        Mapping from node -> parent node. Root maps to None.
    tree_children : dict
        Mapping from node -> list of child nodes.
    leaf_alleles : dict
        Mapping from leaf node -> observed allele.
    root : int
        Root node ID.

    Returns
    -------
    mutations : list of (node, parent_allele, child_allele)
        Mutations placed on edges.
    """
    # --- Bottom-up pass: compute Fitch sets ---
    # The Fitch set at each node is the set of alleles that
    # minimize the number of mutations in the subtree below.
    fitch_set = {}

    # Post-order traversal (children before parents)
    def bottom_up(node):
        if node not in tree_children or len(tree_children[node]) == 0:
            # Leaf node: Fitch set = {observed allele}
            fitch_set[node] = {leaf_alleles[node]}
            return

        child_sets = []
        for child in tree_children[node]:
            bottom_up(child)
            child_sets.append(fitch_set[child])

        # If children agree (non-empty intersection), take intersection
        common = child_sets[0]
        for s in child_sets[1:]:
            common = common & s

        if len(common) > 0:
            # Children agree -- no mutation needed at this node
            fitch_set[node] = common
        else:
            # Children disagree -- take union, mutation needed somewhere
            union = set()
            for s in child_sets:
                union = union | s
            fitch_set[node] = union

    bottom_up(root)

    # --- Top-down pass: assign alleles and place mutations ---
    assigned = {}
    mutations = []

    def top_down(node, parent_allele):
        # If the parent's allele is in this node's Fitch set, keep it
        # (no mutation needed). Otherwise, pick from the Fitch set.
        if parent_allele in fitch_set[node]:
            assigned[node] = parent_allele
        else:
            assigned[node] = min(fitch_set[node])  # Deterministic tie-break

        if node in tree_children:
            for child in tree_children[node]:
                top_down(child, assigned[node])
                # If child got a different allele, that's a mutation
                if assigned[child] != assigned[node]:
                    mutations.append((child, assigned[node],
                                      assigned[child]))

    # Root gets any allele from its Fitch set
    root_allele = min(fitch_set[root])
    assigned[root] = root_allele
    if root in tree_children:
        for child in tree_children[root]:
            top_down(child, root_allele)
            if assigned[child] != root_allele:
                mutations.append((child, root_allele, assigned[child]))

    return mutations


def remove_virtual_root(edges, nodes, virtual_root_id):
    """Remove the virtual root node.

    Children of the virtual root become roots of their subtrees.

    Parameters
    ----------
    edges : list of (left, right, parent, child)
    nodes : list of dict
    virtual_root_id : int

    Returns
    -------
    filtered_edges : list of (left, right, parent, child)
    filtered_nodes : list of dict
    """
    # Simply remove all edges where the virtual root is the parent
    filtered_edges = [(l, r, p, c) for l, r, p, c in edges
                      if p != virtual_root_id]
    # Remove the virtual root node itself
    filtered_nodes = [n for n in nodes if n['id'] != virtual_root_id]
    return filtered_edges, filtered_nodes


def erase_flanks(edges, leftmost_position, rightmost_position):
    """Trim edges that extend beyond the data range.

    Parameters
    ----------
    edges : list of (left, right, parent, child)
    leftmost_position : float
        Leftmost inference site position.
    rightmost_position : float
        Rightmost inference site position.

    Returns
    -------
    trimmed_edges : list of (left, right, parent, child)
    """
    trimmed = []
    for left, right, parent, child in edges:
        # Clamp the edge to the data range
        new_left = max(left, leftmost_position)
        new_right = min(right, rightmost_position)
        # Only keep edges that still have positive length
        if new_left < new_right:
            trimmed.append((new_left, new_right, parent, child))
    return trimmed


def simplify_tree_sequence(nodes, edges, sample_ids):
    """Simplified illustration of the simplify algorithm.

    In practice, use tskit's built-in simplify().

    Parameters
    ----------
    nodes : list of dict
    edges : list of (left, right, parent, child)
    sample_ids : set of int

    Returns
    -------
    kept_nodes : set of int
        Node IDs retained after simplification.
    kept_edges : list of (left, right, parent, child)
        Edges retained.
    """
    # Find all nodes ancestral to at least one sample
    # by traversing upward from the samples through edges
    ancestral = set(sample_ids)
    edge_map = {}
    for left, right, parent, child in edges:
        if child not in edge_map:
            edge_map[child] = []
        edge_map[child].append((left, right, parent))

    # BFS upward from samples to find all ancestors
    queue = list(sample_ids)
    while queue:
        node = queue.pop(0)
        if node in edge_map:
            for left, right, parent in edge_map[node]:
                if parent not in ancestral:
                    ancestral.add(parent)
                    queue.append(parent)

    # Keep only edges between ancestral nodes
    kept_edges = [(l, r, p, c) for l, r, p, c in edges
                  if p in ancestral and c in ancestral]
    kept_nodes = ancestral

    return kept_nodes, kept_edges


def tsinfer_pipeline(D, positions, ancestral_known,
                     recombination_rate=1e-8,
                     mismatch_ratio=1.0,
                     sequence_length=None):
    """Run the complete tsinfer pipeline.

    Parameters
    ----------
    D : ndarray of shape (n, m)
        Variant matrix (0 = ancestral, 1 = derived).
    positions : ndarray of float
        Genomic positions of all sites.
    ancestral_known : ndarray of bool
        Whether the ancestral allele is known at each site.
    recombination_rate : float
        Per-base-pair recombination rate.
    mismatch_ratio : float
        Mismatch-to-recombination ratio.
    sequence_length : float or None
        Total genome length (defaults to max position + 1).

    Returns
    -------
    builder : TreeSequenceBuilder
        The final tree sequence.
    """
    if sequence_length is None:
        sequence_length = positions[-1] + 1

    n, m = D.shape
    print(f"=== tsinfer pipeline ===")
    print(f"Samples: {n}, Sites: {m}")

    # --- Phase 1: Ancestor generation (Gear 1) ---
    # "Extracting the template gears"
    print(f"\nPhase 1: Generating ancestors...")
    ancestors, inference_sites = generate_ancestors(D, ancestral_known)
    inf_positions = positions[inference_sites]
    ancestors = add_ultimate_ancestor(ancestors, len(inference_sites))
    print(f"  Generated {len(ancestors)} ancestors "
          f"({len(inference_sites)} inference sites)")

    # --- Phase 2: Ancestor matching (Gear 3, using Gear 2's engine) ---
    # "Assembling the movement"
    print(f"\nPhase 2: Matching ancestors...")
    builder = match_ancestors(
        ancestors, inference_sites, inf_positions,
        recombination_rate, mismatch_ratio, sequence_length)
    print(f"  Ancestor tree: {len(builder.nodes)} nodes, "
          f"{len(builder.edges)} edges")

    # --- Phase 3: Sample matching (Gear 4, using Gear 2's engine) ---
    # "Fitting the hands to the dial"
    print(f"\nPhase 3: Matching samples...")
    samples_at_inf = D[:, inference_sites]
    builder = match_samples(
        samples_at_inf, ancestors, inference_sites, inf_positions,
        recombination_rate, mismatch_ratio, builder)
    print(f"  After samples: {len(builder.nodes)} nodes, "
          f"{len(builder.edges)} edges")

    # --- Phase 4: Post-processing ---
    # "Final polishing and regulation"
    print(f"\nPhase 4: Post-processing...")

    # Flank erasure: trim edges beyond data range
    builder.edges = erase_flanks(
        builder.edges,
        leftmost_position=inf_positions[0],
        rightmost_position=inf_positions[-1] + 1)
    print(f"  After flank erasure: {len(builder.edges)} edges")

    # Simplification: remove nodes/edges not ancestral to samples
    sample_ids = {n['id'] for n in builder.nodes if n['is_sample']}
    kept_nodes, kept_edges = simplify_tree_sequence(
        builder.nodes, builder.edges, sample_ids)
    print(f"  After simplification: {len(kept_nodes)} nodes, "
          f"{len(kept_edges)} edges")

    print(f"\n=== Done ===")
    return builder


def verify_pipeline(builder, D, inference_sites):
    """Verify the output of the tsinfer pipeline."""
    print("Pipeline verification:")

    # 1. All samples are present as nodes
    sample_nodes = [n for n in builder.nodes if n['is_sample']]
    print(f"  [ok] {len(sample_nodes)} sample nodes present")

    # 2. All samples have at least one parent edge
    children_with_edges = set()
    for l, r, p, c in builder.edges:
        children_with_edges.add(c)
    samples_with_parents = sum(
        1 for n in sample_nodes if n['id'] in children_with_edges)
    print(f"  [{'ok' if samples_with_parents == len(sample_nodes) else 'FAIL'}] "
          f"All samples have parent edges "
          f"({samples_with_parents}/{len(sample_nodes)})")

    # 3. Edge coordinates are within bounds
    all_lefts = [l for l, r, p, c in builder.edges]
    all_rights = [r for l, r, p, c in builder.edges]
    print(f"  [ok] Edge range: [{min(all_lefts):.0f}, "
          f"{max(all_rights):.0f})")

    # 4. No negative times
    all_times = [n['time'] for n in builder.nodes]
    print(f"  [{'ok' if min(all_times) >= 0 else 'FAIL'}] "
          f"All times >= 0")

    # 5. Summary
    non_sample = len(builder.nodes) - len(sample_nodes)
    print(f"\n  Summary:")
    print(f"    Total nodes: {len(builder.nodes)}")
    print(f"    Ancestor nodes: {non_sample}")
    print(f"    Sample nodes: {len(sample_nodes)}")
    print(f"    Edges: {len(builder.edges)}")
    print(f"    Inference sites used: {len(inference_sites)}")


# ============================================================================
# Demo
# ============================================================================

def demo():
    """Demonstrate the tsinfer pipeline on a small simulated dataset."""
    print("=" * 60)
    print("tsinfer Mini-Implementation Demo")
    print("=" * 60)

    # --- Gear 1: Ancestor Generation ---
    print("\n--- Gear 1: Ancestor Generation ---")
    np.random.seed(42)
    n, m = 20, 15
    D = np.random.binomial(1, 0.3, size=(n, m))
    # Force some edge cases
    D[:, 0] = 1       # Fixed derived -- should be excluded
    D[:, 1] = 0       # Fixed ancestral -- should be excluded
    D[0, 2] = 1; D[1:, 2] = 0  # Singleton -- should be excluded
    ancestral_known = np.ones(m, dtype=bool)
    ancestral_known[3] = False  # Unknown ancestral -- should be excluded

    inf_sites, non_inf_sites = select_inference_sites(D, ancestral_known)
    print(f"Total sites: {m}")
    print(f"Inference sites: {inf_sites}")
    print(f"Non-inference sites: {non_inf_sites}")
    for j in inf_sites:
        freq = D[:, j].sum() / n
        print(f"  Site {j}: derived freq = {freq:.2f}")

    # Compute times
    times = compute_ancestor_times(D, inf_sites)
    print("\nInference sites with time proxies:")
    order = np.argsort(-times)
    for idx in order:
        j = inf_sites[idx]
        print(f"  Site {j}: freq = {times[idx]:.2f} (time proxy)")

    # Focal samples
    for j in inf_sites[:3]:
        focal = get_focal_samples(D, j)
        print(f"Site {j}: focal samples = {focal}, count = {len(focal)}")

    # Build ancestors for the first few inference sites
    for idx in range(min(3, len(inf_sites))):
        anc = build_ancestor(D, inf_sites, times, idx)
        print(f"Ancestor for site {inf_sites[idx]}:")
        print(f"  Time: {anc['time']:.2f}")
        print(f"  Span: sites {anc['start']} to {anc['end']}")
        print(f"  Haplotype: {anc['haplotype']}")

    # Generate all ancestors
    ancestors, inf_sites = generate_ancestors(D, ancestral_known)
    print(f"\nGenerated {len(ancestors)} ancestors")
    print(f"Ancestors (oldest first):")
    for i, anc in enumerate(ancestors):
        site = inf_sites[anc['focal']]
        print(f"  {i}: site={site}, time={anc['time']:.2f}, "
              f"span=[{anc['start']},{anc['end']}), "
              f"len={len(anc['haplotype'])}")

    # Group by time
    groups = group_ancestors_by_time(ancestors)
    print(f"\nNumber of time groups: {len(groups)}")
    for time_val, group in groups:
        print(f"  Time {time_val:.2f}: {len(group)} ancestors")

    # Add ultimate ancestor
    ancestors_with_root = add_ultimate_ancestor(ancestors, len(inf_sites))
    print(f"\nUltimate ancestor: time={ancestors_with_root[0]['time']}, "
          f"haplotype={ancestors_with_root[0]['haplotype'][:5]}...")

    # Verify
    verify_ancestors(ancestors_with_root, D, inf_sites)

    # --- Gear 2: Copying Model ---
    print("\n--- Gear 2: The Copying Model ---")
    positions = np.arange(0, 10000, 1000, dtype=float)
    rho = compute_recombination_probs(positions, recombination_rate=1e-4,
                                       num_ref=50)
    print(f"Recombination probabilities: {np.round(rho, 6)}")

    mu = compute_mismatch_probs(positions, recombination_rate=1e-4,
                                 mismatch_ratio=1.0, num_ref=50)
    print(f"Mismatch probabilities: {np.round(mu, 6)}")

    # Viterbi on a small panel with a mosaic query
    np.random.seed(42)
    k = 5
    m_v = 20
    panel = np.random.binomial(1, 0.3, size=(m_v, k))
    true_path = np.array([1] * 10 + [3] * 10)
    query = np.array([panel[ell, true_path[ell]] for ell in range(m_v)])
    rho_v = np.full(m_v, 0.05)
    rho_v[0] = 0.0
    mu_v = np.full(m_v, 0.01)

    path, log_p = viterbi_ls(query, panel, rho_v, mu_v)
    accuracy = np.mean(path == true_path)
    print(f"\nTrue path:    {true_path}")
    print(f"Viterbi path: {path}")
    print(f"Accuracy: {accuracy:.0%}")
    print(f"Log probability: {log_p:.2f}")

    # Deterministic verification
    panel_det = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 0, 1],
    ])
    query_det = np.array([0, 0, 1, 0, 1])
    rho_det = np.array([0.0, 0.05, 0.05, 0.05, 0.05])
    mu_det = np.full(5, 0.001)
    path_det, _ = viterbi_ls(query_det, panel_det, rho_det, mu_det)
    print(f"\nDeterministic verification:")
    print(f"  Query:       {query_det}")
    print(f"  Viterbi path: {path_det}")
    for ell in range(5):
        assert query_det[ell] == panel_det[ell, path_det[ell]], \
            f"Mismatch at site {ell}!"
    print("  [ok] Path has zero mismatches")

    # Path to edges
    positions_e = np.arange(0, 20000, 1000, dtype=float)
    path_example = np.array([1] * 7 + [3] * 8 + [1] * 5)
    ref_ids = np.array([100, 101, 102, 103, 104])
    edges = path_to_edges(path_example, positions_e, child_id=200,
                           ref_node_ids=ref_ids)
    print(f"\nEdges from Viterbi path:")
    for left, right, parent, child in edges:
        print(f"  [{left:.0f}, {right:.0f}): parent={parent}, child={child}")

    # Breakpoints
    bps = find_breakpoints(path_example, positions_e)
    print(f"\nBreakpoints ({len(bps)}):")
    for pos, from_ref, to_ref in bps:
        print(f"  Position {pos:.0f}: ref {from_ref} -> ref {to_ref}")

    # --- Gear 3: Ancestor Matching ---
    print("\n--- Gear 3: Ancestor Matching ---")
    ancestors_example = [
        {'time': 1.0, 'focal': -1},
        {'time': 0.8, 'focal': 3},
        {'time': 0.8, 'focal': 7},
        {'time': 0.6, 'focal': 1},
        {'time': 0.4, 'focal': 5},
        {'time': 0.4, 'focal': 9},
        {'time': 0.4, 'focal': 12},
        {'time': 0.2, 'focal': 2},
    ]
    mo_groups = matching_order(ancestors_example)
    for i, group in enumerate(mo_groups):
        t = [a['time'] for a in group]
        f = [a['focal'] for a in group]
        print(f"Group {i}: time={t[0]:.1f}, "
              f"{len(group)} ancestors, focals={f}")

    # Path compression example
    edges_raw = [
        (0, 5000, 0, 1),
        (0, 5000, 0, 2),
        (0, 5000, 0, 3),
        (5000, 10000, 0, 1),
        (5000, 10000, 1, 4),
    ]
    nodes_raw = [
        {'id': 0, 'time': 1.0, 'is_sample': False},
        {'id': 1, 'time': 0.8, 'is_sample': False},
        {'id': 2, 'time': 0.6, 'is_sample': False},
        {'id': 3, 'time': 0.4, 'is_sample': False},
        {'id': 4, 'time': 0.2, 'is_sample': True},
    ]
    compressed_edges, compressed_nodes = path_compress(edges_raw, nodes_raw)
    print(f"\nPath compression:")
    print(f"  Original edges: {len(edges_raw)}")
    print(f"  Compressed edges: {len(compressed_edges)}")
    print(f"  New PC nodes: {len(compressed_nodes) - len(nodes_raw)}")

    # --- Gear 4: Sample Matching & Post-Processing ---
    print("\n--- Gear 4: Sample Matching & Post-Processing ---")

    # Fitch parsimony example
    tree_parent = {3: 1, 4: 1, 1: 0, 2: 0, 0: None}
    tree_children = {0: [1, 2], 1: [3, 4], 2: [], 3: [], 4: []}
    leaf_alleles = {2: 0, 3: 1, 4: 1}
    mutations = fitch_parsimony(tree_parent, tree_children,
                                 leaf_alleles, root=0)
    print(f"Fitch parsimony mutations ({len(mutations)}):")
    for node, p_allele, c_allele in mutations:
        print(f"  Edge to node {node}: {p_allele} -> {c_allele}")

    # Flank erasure example
    edges_flank = [
        (0, 10000, 1, 5),
        (2000, 8000, 2, 6),
        (7000, 15000, 3, 7),
    ]
    trimmed = erase_flanks(edges_flank,
                            leftmost_position=1000,
                            rightmost_position=9000)
    print(f"\nFlank erasure:")
    print(f"  Original: {len(edges_flank)} edges")
    print(f"  Trimmed: {len(trimmed)} edges")
    for l, r, p, c in trimmed:
        print(f"    [{l:.0f}, {r:.0f}): {p} -> {c}")

    # Simplification example
    nodes_ex = [
        {'id': 0, 'time': 1.0, 'is_sample': False},
        {'id': 1, 'time': 0.8, 'is_sample': False},
        {'id': 2, 'time': 0.5, 'is_sample': False},
        {'id': 3, 'time': 0.0, 'is_sample': True},
        {'id': 4, 'time': 0.0, 'is_sample': True},
    ]
    edges_ex = [
        (0, 10000, 0, 1),
        (0, 10000, 1, 3),
        (0, 10000, 1, 4),
        (0, 10000, 0, 2),
    ]
    sample_ids = {3, 4}
    kept_nodes, kept_edges = simplify_tree_sequence(nodes_ex, edges_ex,
                                                      sample_ids)
    print(f"\nSimplification:")
    print(f"  Nodes before: {len(nodes_ex)}, after: {len(kept_nodes)}")
    print(f"  Edges before: {len(edges_ex)}, after: {len(kept_edges)}")
    print(f"  Removed nodes: {set(n['id'] for n in nodes_ex) - kept_nodes}")

    print("\n" + "=" * 60)
    print("Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    demo()
