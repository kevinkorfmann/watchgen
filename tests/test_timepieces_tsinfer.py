"""
Tests for tsinfer timepiece code extracted from RST documentation.

Covers:
- ancestor_generation.rst: select_inference_sites, compute_ancestor_times,
    get_focal_samples, build_ancestor, generate_ancestors,
    group_ancestors_by_time, add_ultimate_ancestor, verify_ancestors
- ancestor_matching.rst: matching_order, build_reference_panel, path_compress,
    TreeSequenceBuilder
- copying_model.rst: compute_recombination_probs, compute_mismatch_probs,
    viterbi_ls, viterbi_ls_with_noncopy, path_to_edges, find_breakpoints
- overview.rst: No standalone Python code.
- sample_matching.rst: fitch_parsimony, remove_virtual_root, erase_flanks,
    simplify_tree_sequence
"""

import numpy as np
from collections import defaultdict
from scipy.special import comb


# ============================================================================
# Functions from ancestor_generation.rst
# ============================================================================

def select_inference_sites(D, ancestral_known):
    """Select sites suitable for tree inference."""
    n, m = D.shape
    is_inference = np.zeros(m, dtype=bool)

    for j in range(m):
        if not ancestral_known[j]:
            continue
        derived_count = D[:, j].sum()
        ancestral_count = n - derived_count
        num_alleles = len(np.unique(D[:, j]))
        if (num_alleles == 2 and
            derived_count >= 2 and
            ancestral_count >= 1):
            is_inference[j] = True

    inference_sites = np.where(is_inference)[0]
    non_inference_sites = np.where(~is_inference)[0]
    return inference_sites, non_inference_sites


def compute_ancestor_times(D, inference_sites):
    """Compute time proxy for each inference site."""
    n = D.shape[0]
    times = np.zeros(len(inference_sites))
    for k, j in enumerate(inference_sites):
        non_missing = np.sum(D[:, j] >= 0)
        derived = np.sum(D[:, j] == 1)
        times[k] = derived / non_missing
    return times


def get_focal_samples(D, site_index):
    """Get the samples carrying the derived allele at a site."""
    return np.where(D[:, site_index] == 1)[0]


def build_ancestor(D, inference_sites, times, focal_site_idx):
    """Build an ancestral haplotype by extending from a focal site."""
    n_inf = len(inference_sites)
    focal_j = inference_sites[focal_site_idx]
    focal_time = times[focal_site_idx]
    focal_samples = get_focal_samples(D, focal_j)

    haplotype = np.full(n_inf, -1, dtype=int)
    haplotype[focal_site_idx] = 1

    start = focal_site_idx
    for k in range(focal_site_idx - 1, -1, -1):
        site_k = inference_sites[k]
        if times[k] > focal_time:
            haplotype[k] = 0
            start = k
            break
        alleles = D[focal_samples, site_k]
        ones = np.sum(alleles == 1)
        zeros = np.sum(alleles == 0)
        if ones >= zeros:
            haplotype[k] = 1
        else:
            haplotype[k] = 0
        start = k

    end = focal_site_idx + 1
    for k in range(focal_site_idx + 1, n_inf):
        site_k = inference_sites[k]
        if times[k] > focal_time:
            haplotype[k] = 0
            end = k + 1
            break
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
    """Generate all putative ancestors from variant data."""
    inference_sites, _ = select_inference_sites(D, ancestral_known)
    times = compute_ancestor_times(D, inference_sites)

    ancestors = []
    for idx in range(len(inference_sites)):
        anc = build_ancestor(D, inference_sites, times, idx)
        ancestors.append(anc)

    ancestors.sort(key=lambda a: -a['time'])
    return ancestors, inference_sites


def group_ancestors_by_time(ancestors):
    """Group ancestors by their time proxy."""
    groups = defaultdict(list)
    for anc in ancestors:
        groups[anc['time']].append(anc)
    sorted_groups = sorted(groups.items(), key=lambda x: -x[0])
    return sorted_groups


def add_ultimate_ancestor(ancestors, num_inference_sites):
    """Add the ultimate (root) ancestor."""
    ultimate = {
        'haplotype': np.zeros(num_inference_sites, dtype=int),
        'start': 0,
        'end': num_inference_sites,
        'focal': -1,
        'time': 1.0,
    }
    return [ultimate] + ancestors


# ============================================================================
# Functions from ancestor_matching.rst
# ============================================================================

NONCOPY = -2


def matching_order(ancestors):
    """Determine the order for ancestor matching."""
    groups = []
    current_time = None
    current_group = []

    for anc in ancestors:
        if anc['time'] != current_time:
            if current_group:
                groups.append(current_group)
            current_group = [anc]
            current_time = anc['time']
        else:
            current_group.append(anc)

    if current_group:
        groups.append(current_group)

    return groups


def build_reference_panel(placed_ancestors, num_inference_sites):
    """Build the reference panel from already-placed ancestors."""
    k = len(placed_ancestors)
    panel = np.full((num_inference_sites, k), NONCOPY, dtype=int)

    for col, anc in enumerate(placed_ancestors):
        start = anc['start']
        end = anc['end']
        panel[start:end, col] = anc['haplotype']

    node_ids = [anc.get('node_id', col) for col, anc in
                enumerate(placed_ancestors)]
    return panel, node_ids


PC_TIME_EPSILON = 1.0 / (2**32)


def path_compress(edges, nodes):
    """Apply path compression to a set of edges."""
    groups = defaultdict(list)
    for left, right, parent, child in edges:
        groups[(left, right, parent)].append(child)

    new_edges = []
    new_nodes = list(nodes)
    next_id = max(n['id'] for n in nodes) + 1

    for (left, right, parent), children in groups.items():
        if len(children) <= 1:
            for child in children:
                new_edges.append((left, right, parent, child))
        else:
            parent_time = None
            for n in nodes:
                if n['id'] == parent:
                    parent_time = n['time']
                    break

            pc_time = parent_time - PC_TIME_EPSILON
            pc_node = {'id': next_id, 'time': pc_time,
                       'is_sample': False}
            new_nodes.append(pc_node)

            new_edges.append((left, right, parent, next_id))
            for child in children:
                new_edges.append((left, right, next_id, child))

            next_id += 1

    return new_edges, new_nodes


class TreeSequenceBuilder:
    """Incrementally builds a tree sequence from matching results."""

    def __init__(self, sequence_length, num_inference_sites, positions):
        self.sequence_length = sequence_length
        self.positions = positions
        self.num_inference_sites = num_inference_sites
        self.nodes = []
        self.edges = []
        self.next_id = 0

    def add_node(self, time, is_sample=False):
        node_id = self.next_id
        self.nodes.append({'id': node_id, 'time': time,
                           'is_sample': is_sample})
        self.next_id += 1
        return node_id

    def add_edges_from_path(self, path, child_id, ref_node_ids):
        m = len(path)
        seg_start = 0
        current_ref = path[0]

        for ell in range(1, m):
            if path[ell] != current_ref:
                left = self.positions[seg_start]
                right = self.positions[ell]
                parent = ref_node_ids[current_ref]
                self.edges.append((left, right, parent, child_id))
                seg_start = ell
                current_ref = path[ell]

        left = self.positions[seg_start]
        right = self.positions[m - 1] + 1
        parent = ref_node_ids[current_ref]
        self.edges.append((left, right, parent, child_id))

    def summary(self):
        print(f"Nodes: {len(self.nodes)}")
        print(f"Edges: {len(self.edges)}")
        samples = sum(1 for n in self.nodes if n['is_sample'])
        print(f"Samples: {samples}")


# ============================================================================
# Functions from copying_model.rst
# ============================================================================

def compute_recombination_probs(positions, recombination_rate, num_ref):
    """Compute per-site recombination probabilities."""
    m = len(positions)
    rho = np.zeros(m)
    for ell in range(1, m):
        d = positions[ell] - positions[ell - 1]
        rho[ell] = 1 - np.exp(-d * recombination_rate / num_ref)
    return rho


def compute_mismatch_probs(positions, recombination_rate, mismatch_ratio,
                            num_ref):
    """Compute per-site mismatch probabilities."""
    m = len(positions)
    mu = np.zeros(m)
    for ell in range(1, m):
        d = positions[ell] - positions[ell - 1]
        mu[ell] = 1 - np.exp(-d * recombination_rate * mismatch_ratio
                              / num_ref)
    mu[0] = mu[1] if m > 1 else 1e-6
    return mu


def viterbi_ls(query, panel, rho, mu):
    """Viterbi algorithm for the Li & Stephens model."""
    m, k = panel.shape
    V = np.zeros((m, k))
    psi = np.zeros((m, k), dtype=int)

    for j in range(k):
        if query[0] == panel[0, j]:
            V[0, j] = (1.0 / k) * (1 - mu[0])
        else:
            V[0, j] = (1.0 / k) * mu[0]

    for ell in range(1, m):
        max_prev = np.max(V[ell - 1])
        argmax_prev = np.argmax(V[ell - 1])

        for j in range(k):
            if query[ell] == panel[ell, j]:
                e = 1 - mu[ell]
            else:
                e = mu[ell]

            stay = (1 - rho[ell]) * V[ell - 1, j]
            switch = (rho[ell] / k) * max_prev

            if stay >= switch:
                V[ell, j] = e * stay
                psi[ell, j] = j
            else:
                V[ell, j] = e * switch
                psi[ell, j] = argmax_prev

        scale = np.max(V[ell])
        if scale > 0:
            V[ell] /= scale

    path = np.zeros(m, dtype=int)
    path[-1] = np.argmax(V[-1])

    for ell in range(m - 2, -1, -1):
        path[ell] = psi[ell + 1, path[ell + 1]]

    log_prob = np.sum(np.log(np.max(V, axis=1) + 1e-300))
    return path, log_prob


def viterbi_ls_with_noncopy(query, panel, rho, mu):
    """Viterbi algorithm handling NONCOPY entries."""
    m, k = panel.shape
    V = np.zeros((m, k))
    psi = np.zeros((m, k), dtype=int)

    copiable_0 = [j for j in range(k) if panel[0, j] != NONCOPY]
    k_0 = len(copiable_0)
    for j in range(k):
        if panel[0, j] == NONCOPY:
            V[0, j] = 0
        elif query[0] == panel[0, j]:
            V[0, j] = (1.0 / k_0) * (1 - mu[0])
        else:
            V[0, j] = (1.0 / k_0) * mu[0]

    for ell in range(1, m):
        copiable = [j for j in range(k) if panel[ell, j] != NONCOPY]
        k_ell = len(copiable)
        if k_ell == 0:
            continue

        max_prev = np.max(V[ell - 1])
        argmax_prev = np.argmax(V[ell - 1])

        for j in range(k):
            if panel[ell, j] == NONCOPY:
                V[ell, j] = 0
                psi[ell, j] = j
                continue

            if query[ell] == panel[ell, j]:
                e = 1 - mu[ell]
            else:
                e = mu[ell]

            stay = (1 - rho[ell]) * V[ell - 1, j]
            switch = (rho[ell] / k_ell) * max_prev

            if stay >= switch:
                V[ell, j] = e * stay
                psi[ell, j] = j
            else:
                V[ell, j] = e * switch
                psi[ell, j] = argmax_prev

        scale = np.max(V[ell])
        if scale > 0:
            V[ell] /= scale

    path = np.zeros(m, dtype=int)
    path[-1] = np.argmax(V[-1])

    for ell in range(m - 2, -1, -1):
        path[ell] = psi[ell + 1, path[ell + 1]]

    return path


def path_to_edges(path, positions, child_id, ref_node_ids):
    """Convert a Viterbi path to tree sequence edges."""
    edges = []
    m = len(path)
    seg_start = 0
    current_ref = path[0]

    for ell in range(1, m):
        if path[ell] != current_ref:
            left = positions[seg_start]
            right = positions[ell]
            parent = ref_node_ids[current_ref]
            edges.append((left, right, parent, child_id))
            seg_start = ell
            current_ref = path[ell]

    left = positions[seg_start]
    right = positions[-1] + 1
    parent = ref_node_ids[current_ref]
    edges.append((left, right, parent, child_id))

    return edges


def find_breakpoints(path, positions):
    """Find recombination breakpoints from a Viterbi path."""
    breakpoints = []
    for ell in range(1, len(path)):
        if path[ell] != path[ell - 1]:
            breakpoints.append((
                positions[ell],
                path[ell - 1],
                path[ell]
            ))
    return breakpoints


# ============================================================================
# Functions from sample_matching.rst
# ============================================================================

def fitch_parsimony(tree_parent, tree_children, leaf_alleles, root):
    """Place mutations by Fitch parsimony on a single tree."""
    fitch_set = {}

    def bottom_up(node):
        if node not in tree_children or len(tree_children[node]) == 0:
            fitch_set[node] = {leaf_alleles[node]}
            return

        child_sets = []
        for child in tree_children[node]:
            bottom_up(child)
            child_sets.append(fitch_set[child])

        common = child_sets[0]
        for s in child_sets[1:]:
            common = common & s

        if len(common) > 0:
            fitch_set[node] = common
        else:
            union = set()
            for s in child_sets:
                union = union | s
            fitch_set[node] = union

    bottom_up(root)

    assigned = {}
    mutations = []

    def top_down(node, parent_allele):
        if parent_allele in fitch_set[node]:
            assigned[node] = parent_allele
        else:
            assigned[node] = min(fitch_set[node])

        if node in tree_children:
            for child in tree_children[node]:
                top_down(child, assigned[node])
                if assigned[child] != assigned[node]:
                    mutations.append((child, assigned[node],
                                      assigned[child]))

    root_allele = min(fitch_set[root])
    assigned[root] = root_allele
    if root in tree_children:
        for child in tree_children[root]:
            top_down(child, root_allele)
            if assigned[child] != root_allele:
                mutations.append((child, root_allele, assigned[child]))

    return mutations


def remove_virtual_root(edges, nodes, virtual_root_id):
    """Remove the virtual root node."""
    filtered_edges = [(l, r, p, c) for l, r, p, c in edges
                      if p != virtual_root_id]
    filtered_nodes = [n for n in nodes if n['id'] != virtual_root_id]
    return filtered_edges, filtered_nodes


def erase_flanks(edges, leftmost_position, rightmost_position):
    """Trim edges that extend beyond the data range."""
    trimmed = []
    for left, right, parent, child in edges:
        new_left = max(left, leftmost_position)
        new_right = min(right, rightmost_position)
        if new_left < new_right:
            trimmed.append((new_left, new_right, parent, child))
    return trimmed


def simplify_tree_sequence(nodes, edges, sample_ids):
    """Simplified illustration of the simplify algorithm."""
    ancestral = set(sample_ids)
    edge_map = {}
    for left, right, parent, child in edges:
        if child not in edge_map:
            edge_map[child] = []
        edge_map[child].append((left, right, parent))

    queue = list(sample_ids)
    while queue:
        node = queue.pop(0)
        if node in edge_map:
            for left, right, parent in edge_map[node]:
                if parent not in ancestral:
                    ancestral.add(parent)
                    queue.append(parent)

    kept_edges = [(l, r, p, c) for l, r, p, c in edges
                  if p in ancestral and c in ancestral]
    kept_nodes = ancestral

    return kept_nodes, kept_edges


# ============================================================================
# Tests for ancestor_generation.rst
# ============================================================================

class TestSelectInferenceSites:
    """Tests for site selection."""

    def test_exclude_fixed_derived(self):
        """Sites where all samples are derived should be excluded."""
        D = np.ones((10, 3), dtype=int)
        ancestral_known = np.ones(3, dtype=bool)
        inf, _ = select_inference_sites(D, ancestral_known)
        assert len(inf) == 0

    def test_exclude_fixed_ancestral(self):
        """Sites where all samples are ancestral should be excluded."""
        D = np.zeros((10, 3), dtype=int)
        ancestral_known = np.ones(3, dtype=bool)
        inf, _ = select_inference_sites(D, ancestral_known)
        assert len(inf) == 0

    def test_exclude_singleton(self):
        """Sites with only one derived copy should be excluded."""
        D = np.zeros((10, 3), dtype=int)
        D[0, 0] = 1  # singleton
        D[0, 1] = 1; D[1, 1] = 1  # doubleton (valid)
        D[0, 2] = 1  # singleton
        ancestral_known = np.ones(3, dtype=bool)
        inf, _ = select_inference_sites(D, ancestral_known)
        assert 0 not in inf
        assert 2 not in inf
        assert 1 in inf

    def test_exclude_unknown_ancestral(self):
        """Sites with unknown ancestral allele should be excluded."""
        D = np.zeros((10, 3), dtype=int)
        D[:5, :] = 1  # Half derived
        ancestral_known = np.array([True, False, True])
        inf, _ = select_inference_sites(D, ancestral_known)
        assert 1 not in inf

    def test_partition(self):
        """Inference and non-inference sites should partition all sites."""
        np.random.seed(42)
        D = np.random.binomial(1, 0.3, size=(20, 10))
        ancestral_known = np.ones(10, dtype=bool)
        inf, non_inf = select_inference_sites(D, ancestral_known)
        assert len(inf) + len(non_inf) == 10
        assert len(set(inf) & set(non_inf)) == 0

    def test_valid_inference_sites(self):
        """All selected inference sites should meet the criteria."""
        np.random.seed(42)
        n, m = 20, 15
        D = np.random.binomial(1, 0.3, size=(n, m))
        ancestral_known = np.ones(m, dtype=bool)
        inf, _ = select_inference_sites(D, ancestral_known)
        for j in inf:
            derived_count = D[:, j].sum()
            ancestral_count = n - derived_count
            assert derived_count >= 2
            assert ancestral_count >= 1
            assert len(np.unique(D[:, j])) == 2


class TestComputeAncestorTimes:
    """Tests for ancestor time computation."""

    def test_range(self):
        """Times should be in (0, 1) for valid inference sites."""
        np.random.seed(42)
        n, m = 20, 10
        D = np.random.binomial(1, 0.3, size=(n, m))
        ancestral_known = np.ones(m, dtype=bool)
        inf, _ = select_inference_sites(D, ancestral_known)
        if len(inf) > 0:
            times = compute_ancestor_times(D, inf)
            # Inference sites have >= 2 derived and >= 1 ancestral
            for t in times:
                assert 0 < t < 1

    def test_equals_derived_frequency(self):
        """Times should equal the derived allele frequency."""
        D = np.zeros((10, 3), dtype=int)
        D[:5, 0] = 1  # freq 0.5
        D[:3, 1] = 1  # freq 0.3
        D[:8, 2] = 1  # freq 0.8
        inf = np.array([0, 1, 2])
        times = compute_ancestor_times(D, inf)
        assert np.isclose(times[0], 0.5)
        assert np.isclose(times[1], 0.3)
        assert np.isclose(times[2], 0.8)


class TestGetFocalSamples:
    """Tests for focal sample identification."""

    def test_correct_samples(self):
        """Should return indices where D[:, j] == 1."""
        D = np.array([[0, 1, 0],
                       [1, 0, 0],
                       [1, 1, 0],
                       [0, 0, 1]])
        focal = get_focal_samples(D, 0)
        assert set(focal) == {1, 2}

    def test_all_derived(self):
        """If all samples are derived, all should be focal."""
        D = np.ones((5, 3), dtype=int)
        focal = get_focal_samples(D, 0)
        assert len(focal) == 5

    def test_none_derived(self):
        """If no samples are derived, result should be empty."""
        D = np.zeros((5, 3), dtype=int)
        focal = get_focal_samples(D, 0)
        assert len(focal) == 0


class TestBuildAncestor:
    """Tests for ancestor construction."""

    def test_focal_site_is_derived(self):
        """The ancestor should carry derived allele at its focal site."""
        np.random.seed(42)
        D = np.random.binomial(1, 0.3, size=(20, 10))
        ancestral_known = np.ones(10, dtype=bool)
        inf, _ = select_inference_sites(D, ancestral_known)
        if len(inf) > 0:
            times = compute_ancestor_times(D, inf)
            anc = build_ancestor(D, inf, times, 0)
            focal_in_hap = anc['focal'] - anc['start']
            assert anc['haplotype'][focal_in_hap] == 1

    def test_haplotype_is_binary(self):
        """Ancestor haplotype should contain only 0s and 1s."""
        np.random.seed(42)
        D = np.random.binomial(1, 0.3, size=(20, 10))
        ancestral_known = np.ones(10, dtype=bool)
        inf, _ = select_inference_sites(D, ancestral_known)
        if len(inf) > 0:
            times = compute_ancestor_times(D, inf)
            anc = build_ancestor(D, inf, times, 0)
            assert set(anc['haplotype']).issubset({0, 1})

    def test_span_within_bounds(self):
        """Start and end should be within valid range."""
        np.random.seed(42)
        D = np.random.binomial(1, 0.3, size=(20, 10))
        ancestral_known = np.ones(10, dtype=bool)
        inf, _ = select_inference_sites(D, ancestral_known)
        if len(inf) > 0:
            times = compute_ancestor_times(D, inf)
            for idx in range(len(inf)):
                anc = build_ancestor(D, inf, times, idx)
                assert 0 <= anc['start'] <= anc['focal']
                assert anc['focal'] < anc['end'] <= len(inf)


class TestGenerateAncestors:
    """Tests for the full ancestor generation algorithm."""

    def test_sorted_by_time_descending(self):
        """Ancestors should be sorted by time (oldest first)."""
        np.random.seed(42)
        D = np.random.binomial(1, 0.3, size=(20, 10))
        ancestral_known = np.ones(10, dtype=bool)
        ancestors, _ = generate_ancestors(D, ancestral_known)
        for i in range(len(ancestors) - 1):
            assert ancestors[i]['time'] >= ancestors[i + 1]['time']

    def test_one_ancestor_per_inference_site(self):
        """Should generate one ancestor per inference site."""
        np.random.seed(42)
        D = np.random.binomial(1, 0.3, size=(20, 10))
        ancestral_known = np.ones(10, dtype=bool)
        ancestors, inf_sites = generate_ancestors(D, ancestral_known)
        assert len(ancestors) == len(inf_sites)


class TestGroupAncestorsByTime:
    """Tests for ancestor grouping."""

    def test_grouping(self):
        """Ancestors at the same time should be in the same group."""
        ancestors = [
            {'time': 0.8, 'focal': 1},
            {'time': 0.8, 'focal': 2},
            {'time': 0.5, 'focal': 3},
            {'time': 0.3, 'focal': 4},
            {'time': 0.3, 'focal': 5},
            {'time': 0.3, 'focal': 6},
        ]
        groups = group_ancestors_by_time(ancestors)
        assert len(groups) == 3
        # Oldest first
        assert groups[0][0] == 0.8
        assert len(groups[0][1]) == 2
        assert groups[1][0] == 0.5
        assert len(groups[1][1]) == 1
        assert groups[2][0] == 0.3
        assert len(groups[2][1]) == 3

    def test_all_ancestors_accounted_for(self):
        """Total ancestors across groups should equal input count."""
        ancestors = [
            {'time': 0.8, 'focal': 1},
            {'time': 0.5, 'focal': 2},
            {'time': 0.5, 'focal': 3},
        ]
        groups = group_ancestors_by_time(ancestors)
        total = sum(len(g) for _, g in groups)
        assert total == len(ancestors)


class TestAddUltimateAncestor:
    """Tests for the ultimate ancestor."""

    def test_time_is_one(self):
        """Ultimate ancestor should be at time 1.0."""
        ancestors = [{'time': 0.5, 'focal': 0}]
        result = add_ultimate_ancestor(ancestors, 10)
        assert result[0]['time'] == 1.0

    def test_all_zeros(self):
        """Ultimate ancestor should carry ancestral allele everywhere."""
        result = add_ultimate_ancestor([], 10)
        assert np.all(result[0]['haplotype'] == 0)

    def test_full_span(self):
        """Ultimate ancestor should span all inference sites."""
        result = add_ultimate_ancestor([], 15)
        assert result[0]['start'] == 0
        assert result[0]['end'] == 15

    def test_prepended(self):
        """Ultimate ancestor should be first in the list."""
        ancestors = [{'time': 0.5, 'focal': 0}]
        result = add_ultimate_ancestor(ancestors, 10)
        assert len(result) == 2
        assert result[0]['focal'] == -1


# ============================================================================
# Tests for ancestor_matching.rst
# ============================================================================

class TestMatchingOrder:
    """Tests for ancestor matching order."""

    def test_groups_by_time(self):
        """Ancestors at same time should be grouped together."""
        ancestors = [
            {'time': 1.0, 'focal': -1},
            {'time': 0.8, 'focal': 3},
            {'time': 0.8, 'focal': 7},
            {'time': 0.6, 'focal': 1},
            {'time': 0.4, 'focal': 5},
        ]
        groups = matching_order(ancestors)
        assert len(groups) == 4

    def test_preserves_order(self):
        """Groups should appear in the same order as input."""
        ancestors = [
            {'time': 0.9, 'focal': 1},
            {'time': 0.7, 'focal': 2},
            {'time': 0.5, 'focal': 3},
        ]
        groups = matching_order(ancestors)
        times = [g[0]['time'] for g in groups]
        assert times == [0.9, 0.7, 0.5]

    def test_single_group(self):
        """If all ancestors have the same time, should be one group."""
        ancestors = [
            {'time': 0.5, 'focal': i} for i in range(5)
        ]
        groups = matching_order(ancestors)
        assert len(groups) == 1
        assert len(groups[0]) == 5


class TestBuildReferencePanel:
    """Tests for reference panel construction."""

    def test_shape(self):
        """Panel shape should be (num_sites, num_ancestors)."""
        placed = [
            {'haplotype': np.zeros(10, dtype=int), 'start': 0, 'end': 10,
             'node_id': 0},
        ]
        panel, _ = build_reference_panel(placed, num_inference_sites=10)
        assert panel.shape == (10, 1)

    def test_noncopy_default(self):
        """Positions outside an ancestor's interval should be NONCOPY."""
        placed = [
            {'haplotype': np.array([1, 0, 1]), 'start': 3, 'end': 6,
             'node_id': 0},
        ]
        panel, _ = build_reference_panel(placed, num_inference_sites=10)
        assert panel[0, 0] == NONCOPY
        assert panel[3, 0] == 1
        assert panel[5, 0] == 1
        assert panel[6, 0] == NONCOPY

    def test_multiple_ancestors(self):
        """Multiple ancestors should fill different columns."""
        placed = [
            {'haplotype': np.zeros(10, dtype=int), 'start': 0, 'end': 10,
             'node_id': 0},
            {'haplotype': np.ones(5, dtype=int), 'start': 2, 'end': 7,
             'node_id': 1},
        ]
        panel, _ = build_reference_panel(placed, num_inference_sites=10)
        assert panel.shape == (10, 2)
        assert panel[0, 0] == 0
        assert panel[0, 1] == NONCOPY
        assert panel[3, 1] == 1


class TestPathCompress:
    """Tests for path compression."""

    def test_no_shared_edges(self):
        """With no shared (left, right, parent), no compression occurs."""
        edges = [
            (0, 5000, 0, 1),
            (5000, 10000, 0, 2),
        ]
        nodes = [
            {'id': 0, 'time': 1.0, 'is_sample': False},
            {'id': 1, 'time': 0.5, 'is_sample': True},
            {'id': 2, 'time': 0.5, 'is_sample': True},
        ]
        new_edges, new_nodes = path_compress(edges, nodes)
        assert len(new_edges) == 2
        assert len(new_nodes) == 3

    def test_shared_edges_compressed(self):
        """Multiple children sharing same parent and interval get a PC node."""
        edges = [
            (0, 5000, 0, 1),
            (0, 5000, 0, 2),
            (0, 5000, 0, 3),
        ]
        nodes = [
            {'id': 0, 'time': 1.0, 'is_sample': False},
            {'id': 1, 'time': 0.5, 'is_sample': True},
            {'id': 2, 'time': 0.5, 'is_sample': True},
            {'id': 3, 'time': 0.5, 'is_sample': True},
        ]
        new_edges, new_nodes = path_compress(edges, nodes)
        # Should have 1 edge from parent to PC node + 3 edges from PC to children
        assert len(new_edges) == 4
        # One new PC node
        assert len(new_nodes) == 5

    def test_pc_node_time(self):
        """PC node time should be parent_time - epsilon."""
        edges = [
            (0, 5000, 0, 1),
            (0, 5000, 0, 2),
        ]
        nodes = [
            {'id': 0, 'time': 1.0, 'is_sample': False},
            {'id': 1, 'time': 0.5, 'is_sample': True},
            {'id': 2, 'time': 0.5, 'is_sample': True},
        ]
        new_edges, new_nodes = path_compress(edges, nodes)
        pc_nodes = [n for n in new_nodes if n['id'] >= 3]
        assert len(pc_nodes) == 1
        assert np.isclose(pc_nodes[0]['time'], 1.0 - PC_TIME_EPSILON)


class TestTreeSequenceBuilder:
    """Tests for the TreeSequenceBuilder."""

    def test_add_node(self):
        """Adding a node should increment the ID."""
        positions = np.arange(0, 10000, 1000, dtype=float)
        builder = TreeSequenceBuilder(10000, 10, positions)
        id0 = builder.add_node(time=1.0)
        id1 = builder.add_node(time=0.5)
        assert id0 == 0
        assert id1 == 1
        assert len(builder.nodes) == 2

    def test_add_edges_constant_path(self):
        """A constant path should produce one edge."""
        positions = np.arange(0, 5000, 1000, dtype=float)
        builder = TreeSequenceBuilder(5000, 5, positions)
        root_id = builder.add_node(time=1.0)
        child_id = builder.add_node(time=0.0, is_sample=True)
        path = np.zeros(5, dtype=int)
        builder.add_edges_from_path(path, child_id, ref_node_ids=[root_id])
        assert len(builder.edges) == 1

    def test_add_edges_with_switch(self):
        """A path with one switch should produce two edges."""
        positions = np.arange(0, 10000, 1000, dtype=float)
        builder = TreeSequenceBuilder(10000, 10, positions)
        id0 = builder.add_node(time=1.0)
        id1 = builder.add_node(time=0.8)
        child = builder.add_node(time=0.0, is_sample=True)
        path = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        builder.add_edges_from_path(path, child, ref_node_ids=[id0, id1])
        assert len(builder.edges) == 2


# ============================================================================
# Tests for copying_model.rst
# ============================================================================

class TestComputeRecombinationProbs:
    """Tests for recombination probability computation."""

    def test_first_site_zero(self):
        """Recombination at site 0 should be 0."""
        positions = np.array([0.0, 1000.0, 2000.0])
        rho = compute_recombination_probs(positions, 1e-4, 50)
        assert rho[0] == 0.0

    def test_positive_for_positive_distance(self):
        """Recombination prob should be positive for positive distance."""
        positions = np.array([0.0, 1000.0, 2000.0])
        rho = compute_recombination_probs(positions, 1e-4, 50)
        assert rho[1] > 0
        assert rho[2] > 0

    def test_bounded_below_one(self):
        """Recombination probability should be < 1."""
        positions = np.array([0.0, 1000.0])
        rho = compute_recombination_probs(positions, 1e-4, 50)
        assert rho[1] < 1.0

    def test_uniform_spacing_equal_probs(self):
        """Uniformly spaced sites should have equal recombination probs."""
        positions = np.arange(0, 5000, 1000, dtype=float)
        rho = compute_recombination_probs(positions, 1e-4, 50)
        assert np.allclose(rho[1:], rho[1])

    def test_larger_distance_higher_prob(self):
        """Larger distance should give higher recombination probability."""
        positions = np.array([0.0, 1000.0, 5000.0])
        rho = compute_recombination_probs(positions, 1e-4, 50)
        assert rho[2] > rho[1]


class TestComputeMismatchProbs:
    """Tests for mismatch probability computation."""

    def test_first_site_equals_second(self):
        """Mismatch at site 0 should be set to mu[1]."""
        positions = np.array([0.0, 1000.0, 2000.0])
        mu = compute_mismatch_probs(positions, 1e-4, 1.0, 50)
        assert mu[0] == mu[1]

    def test_positive(self):
        """Mismatch probabilities should be positive."""
        positions = np.arange(0, 5000, 1000, dtype=float)
        mu = compute_mismatch_probs(positions, 1e-4, 1.0, 50)
        assert np.all(mu > 0)

    def test_ratio_effect(self):
        """Higher mismatch ratio should give higher mismatch probs."""
        positions = np.array([0.0, 1000.0])
        mu_low = compute_mismatch_probs(positions, 1e-4, 0.1, 50)
        mu_high = compute_mismatch_probs(positions, 1e-4, 10.0, 50)
        assert mu_high[1] > mu_low[1]


class TestViterbiLS:
    """Tests for the Li & Stephens Viterbi algorithm."""

    def test_perfect_copy(self):
        """Query that is an exact copy of one reference should match it."""
        np.random.seed(42)
        k, m = 5, 20
        panel = np.random.binomial(1, 0.3, size=(m, k))
        query = panel[:, 2].copy()  # Exact copy of ref 2
        rho = np.full(m, 0.05)
        rho[0] = 0.0
        mu = np.full(m, 0.001)
        path, _ = viterbi_ls(query, panel, rho, mu)
        # Path should be all 2 (or very close)
        assert np.mean(path == 2) > 0.8

    def test_mosaic_recovery(self):
        """A mosaic query should recover the correct segments."""
        np.random.seed(42)
        k, m = 5, 30
        panel = np.random.binomial(1, 0.3, size=(m, k))
        # Mosaic: ref 1 for first half, ref 3 for second half
        true_path = np.array([1]*15 + [3]*15)
        query = np.array([panel[ell, true_path[ell]] for ell in range(m)])
        rho = np.full(m, 0.05)
        rho[0] = 0.0
        mu = np.full(m, 0.001)
        path, _ = viterbi_ls(query, panel, rho, mu)
        accuracy = np.mean(path == true_path)
        assert accuracy > 0.6

    def test_output_length(self):
        """Path length should equal number of sites."""
        panel = np.random.binomial(1, 0.3, size=(10, 3))
        query = panel[:, 0]
        rho = np.full(10, 0.05)
        rho[0] = 0.0
        mu = np.full(10, 0.01)
        path, _ = viterbi_ls(query, panel, rho, mu)
        assert len(path) == 10

    def test_valid_state_indices(self):
        """Path values should be valid indices into the panel."""
        np.random.seed(42)
        k, m = 4, 15
        panel = np.random.binomial(1, 0.3, size=(m, k))
        query = np.random.binomial(1, 0.3, size=m)
        rho = np.full(m, 0.05)
        rho[0] = 0.0
        mu = np.full(m, 0.01)
        path, _ = viterbi_ls(query, panel, rho, mu)
        assert np.all(path >= 0) and np.all(path < k)

    def test_deterministic_example(self):
        """Verify on a hand-crafted deterministic example."""
        panel = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 1],
        ])
        query = np.array([0, 0, 1, 0, 1])
        rho = np.array([0.0, 0.05, 0.05, 0.05, 0.05])
        mu = np.full(5, 0.001)
        path, _ = viterbi_ls(query, panel, rho, mu)
        # Path should match without any mismatches
        for ell in range(5):
            assert query[ell] == panel[ell, path[ell]]


class TestViterbiLSWithNoncopy:
    """Tests for the NONCOPY-aware Viterbi."""

    def test_no_noncopy_selected(self):
        """No NONCOPY reference should appear in the path."""
        np.random.seed(42)
        m, k = 20, 5
        panel = np.full((m, k), NONCOPY, dtype=int)
        panel[:15, 0] = np.random.binomial(1, 0.3, 15)
        panel[:, 1] = np.random.binomial(1, 0.3, m)
        panel[5:, 2] = np.random.binomial(1, 0.3, 15)
        panel[:, 3] = np.random.binomial(1, 0.3, m)
        panel[:, 4] = np.random.binomial(1, 0.3, m)

        query = np.random.binomial(1, 0.3, m)
        rho = np.full(m, 0.05)
        rho[0] = 0.0
        mu = np.full(m, 0.01)

        path = viterbi_ls_with_noncopy(query, panel, rho, mu)
        for ell in range(m):
            assert panel[ell, path[ell]] != NONCOPY

    def test_output_length(self):
        """Path length should equal number of sites."""
        m, k = 10, 3
        panel = np.random.binomial(1, 0.3, size=(m, k))
        query = np.random.binomial(1, 0.3, m)
        rho = np.full(m, 0.05)
        rho[0] = 0.0
        mu = np.full(m, 0.01)
        path = viterbi_ls_with_noncopy(query, panel, rho, mu)
        assert len(path) == m

    def test_all_copiable_same_as_standard(self):
        """When no NONCOPY entries, should give same result as standard Viterbi."""
        np.random.seed(42)
        m, k = 15, 3
        panel = np.random.binomial(1, 0.3, size=(m, k))
        query = panel[:, 0].copy()
        rho = np.full(m, 0.05)
        rho[0] = 0.0
        mu = np.full(m, 0.01)
        path_nc = viterbi_ls_with_noncopy(query, panel, rho, mu)
        path_std, _ = viterbi_ls(query, panel, rho, mu)
        assert np.array_equal(path_nc, path_std)


class TestPathToEdges:
    """Tests for converting Viterbi paths to edges."""

    def test_constant_path_one_edge(self):
        """A constant path should produce one edge."""
        positions = np.arange(0, 5000, 1000, dtype=float)
        path = np.zeros(5, dtype=int)
        ref_ids = np.array([100])
        edges = path_to_edges(path, positions, child_id=200, ref_node_ids=ref_ids)
        assert len(edges) == 1
        assert edges[0] == (0.0, 4001.0, 100, 200)

    def test_two_segments(self):
        """A path with one switch should produce two edges."""
        positions = np.arange(0, 10000, 1000, dtype=float)
        path = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        ref_ids = np.array([100, 101])
        edges = path_to_edges(path, positions, child_id=200, ref_node_ids=ref_ids)
        assert len(edges) == 2

    def test_edges_cover_full_range(self):
        """Edges should cover the full genomic range."""
        positions = np.arange(0, 20000, 1000, dtype=float)
        path = np.array([1]*7 + [3]*8 + [1]*5)
        ref_ids = np.array([100, 101, 102, 103, 104])
        edges = path_to_edges(path, positions, child_id=200, ref_node_ids=ref_ids)
        # First edge starts at positions[0], last edge ends at positions[-1]+1
        assert edges[0][0] == positions[0]
        assert edges[-1][1] == positions[-1] + 1


class TestFindBreakpoints:
    """Tests for breakpoint detection."""

    def test_constant_path_no_breakpoints(self):
        """A constant path should have no breakpoints."""
        path = np.zeros(10, dtype=int)
        positions = np.arange(0, 10000, 1000, dtype=float)
        bps = find_breakpoints(path, positions)
        assert len(bps) == 0

    def test_one_switch(self):
        """A path with one switch should have one breakpoint."""
        path = np.array([0, 0, 0, 1, 1, 1])
        positions = np.arange(0, 6000, 1000, dtype=float)
        bps = find_breakpoints(path, positions)
        assert len(bps) == 1
        assert bps[0] == (3000.0, 0, 1)

    def test_multiple_switches(self):
        """Multiple switches should produce corresponding breakpoints."""
        path = np.array([0, 0, 1, 1, 2, 2])
        positions = np.arange(0, 6000, 1000, dtype=float)
        bps = find_breakpoints(path, positions)
        assert len(bps) == 2


# ============================================================================
# Tests for sample_matching.rst
# ============================================================================

class TestFitchParsimony:
    """Tests for Fitch parsimony mutation placement."""

    def test_no_mutations_all_same(self):
        """If all leaves have the same allele, no mutations needed."""
        tree_parent = {1: 0, 2: 0, 0: None}
        tree_children = {0: [1, 2], 1: [], 2: []}
        leaf_alleles = {1: 0, 2: 0}
        mutations = fitch_parsimony(tree_parent, tree_children,
                                     leaf_alleles, root=0)
        assert len(mutations) == 0

    def test_one_mutation(self):
        """A single derived leaf should produce one mutation."""
        #     0 (root)
        #    / \
        #   1   2
        tree_parent = {1: 0, 2: 0, 0: None}
        tree_children = {0: [1, 2], 1: [], 2: []}
        leaf_alleles = {1: 0, 2: 1}
        mutations = fitch_parsimony(tree_parent, tree_children,
                                     leaf_alleles, root=0)
        assert len(mutations) == 1

    def test_parsimony_optimal(self):
        """Fitch should find the minimum number of mutations."""
        #       0 (root)
        #      / \
        #     1   2
        #    / \
        #   3   4
        tree_parent = {3: 1, 4: 1, 1: 0, 2: 0, 0: None}
        tree_children = {0: [1, 2], 1: [3, 4], 2: [], 3: [], 4: []}
        leaf_alleles = {2: 0, 3: 1, 4: 1}
        mutations = fitch_parsimony(tree_parent, tree_children,
                                     leaf_alleles, root=0)
        # Optimal: one mutation on edge to node 1 (or equivalently on the
        # branch leading to the clade {3,4})
        assert len(mutations) == 1

    def test_two_mutations(self):
        """Alternating alleles require at least two mutations (no single edge explains both)."""
        #       0
        #      / \
        #     1   2
        #    / \   / \
        #   3   4 5   6
        # Leaf 3=0, 4=1, 5=0, 6=1
        # No single edge can explain the pattern, so at least 2 mutations needed
        tree_parent = {3: 1, 4: 1, 5: 2, 6: 2, 1: 0, 2: 0, 0: None}
        tree_children = {0: [1, 2], 1: [3, 4], 2: [5, 6],
                         3: [], 4: [], 5: [], 6: []}
        leaf_alleles = {3: 0, 4: 1, 5: 0, 6: 1}
        mutations = fitch_parsimony(tree_parent, tree_children,
                                     leaf_alleles, root=0)
        assert len(mutations) == 2

    def test_star_tree(self):
        """Star tree: root with multiple leaves."""
        #     0
        #   / | \
        #  1  2  3
        tree_parent = {1: 0, 2: 0, 3: 0, 0: None}
        tree_children = {0: [1, 2, 3], 1: [], 2: [], 3: []}
        leaf_alleles = {1: 0, 2: 0, 3: 0}
        mutations = fitch_parsimony(tree_parent, tree_children,
                                     leaf_alleles, root=0)
        assert len(mutations) == 0


class TestRemoveVirtualRoot:
    """Tests for virtual root removal."""

    def test_removes_root_edges(self):
        """Edges with virtual root as parent should be removed."""
        edges = [
            (0, 10000, 0, 1),  # root -> 1
            (0, 10000, 1, 2),  # 1 -> 2
        ]
        nodes = [
            {'id': 0, 'time': 1.0, 'is_sample': False},
            {'id': 1, 'time': 0.5, 'is_sample': False},
            {'id': 2, 'time': 0.0, 'is_sample': True},
        ]
        filtered_edges, filtered_nodes = remove_virtual_root(edges, nodes, 0)
        assert len(filtered_edges) == 1
        assert filtered_edges[0] == (0, 10000, 1, 2)

    def test_removes_root_node(self):
        """Virtual root node should be removed."""
        edges = [(0, 10000, 0, 1)]
        nodes = [
            {'id': 0, 'time': 1.0, 'is_sample': False},
            {'id': 1, 'time': 0.0, 'is_sample': True},
        ]
        _, filtered_nodes = remove_virtual_root(edges, nodes, 0)
        ids = [n['id'] for n in filtered_nodes]
        assert 0 not in ids


class TestEraseFlanks:
    """Tests for flank erasure."""

    def test_trim_left(self):
        """Edge extending left of data range should be trimmed."""
        edges = [(0, 10000, 1, 2)]
        trimmed = erase_flanks(edges, leftmost_position=1000,
                                rightmost_position=10000)
        assert len(trimmed) == 1
        assert trimmed[0][0] == 1000

    def test_trim_right(self):
        """Edge extending right of data range should be trimmed."""
        edges = [(0, 15000, 1, 2)]
        trimmed = erase_flanks(edges, leftmost_position=0,
                                rightmost_position=10000)
        assert len(trimmed) == 1
        assert trimmed[0][1] == 10000

    def test_remove_out_of_range(self):
        """Edge entirely outside range should be removed."""
        edges = [(20000, 30000, 1, 2)]
        trimmed = erase_flanks(edges, leftmost_position=0,
                                rightmost_position=10000)
        assert len(trimmed) == 0

    def test_within_range_unchanged(self):
        """Edge within range should be unchanged."""
        edges = [(2000, 8000, 1, 2)]
        trimmed = erase_flanks(edges, leftmost_position=1000,
                                rightmost_position=9000)
        assert len(trimmed) == 1
        assert trimmed[0] == (2000, 8000, 1, 2)

    def test_example_from_docs(self):
        """Test the example from the RST documentation."""
        edges_raw = [
            (0, 10000, 1, 5),
            (2000, 8000, 2, 6),
            (7000, 15000, 3, 7),
        ]
        trimmed = erase_flanks(edges_raw, leftmost_position=1000,
                                rightmost_position=9000)
        assert len(trimmed) == 3
        assert trimmed[0] == (1000, 9000, 1, 5)
        assert trimmed[1] == (2000, 8000, 2, 6)
        assert trimmed[2] == (7000, 9000, 3, 7)


class TestSimplifyTreeSequence:
    """Tests for tree sequence simplification."""

    def test_removes_non_ancestral_nodes(self):
        """Nodes not ancestral to any sample should be removed."""
        nodes = [
            {'id': 0, 'time': 1.0, 'is_sample': False},
            {'id': 1, 'time': 0.8, 'is_sample': False},
            {'id': 2, 'time': 0.5, 'is_sample': False},  # orphan
            {'id': 3, 'time': 0.0, 'is_sample': True},
            {'id': 4, 'time': 0.0, 'is_sample': True},
        ]
        edges = [
            (0, 10000, 0, 1),
            (0, 10000, 1, 3),
            (0, 10000, 1, 4),
            (0, 10000, 0, 2),  # Node 2 has no sample descendants
        ]
        sample_ids = {3, 4}
        kept_nodes, kept_edges = simplify_tree_sequence(nodes, edges,
                                                         sample_ids)
        assert 2 not in kept_nodes
        assert 3 in kept_nodes
        assert 4 in kept_nodes
        assert 0 in kept_nodes
        assert 1 in kept_nodes

    def test_keeps_edges_between_ancestral(self):
        """Edges between kept nodes should be preserved."""
        nodes = [
            {'id': 0, 'time': 1.0, 'is_sample': False},
            {'id': 1, 'time': 0.0, 'is_sample': True},
        ]
        edges = [(0, 10000, 0, 1)]
        sample_ids = {1}
        kept_nodes, kept_edges = simplify_tree_sequence(nodes, edges,
                                                         sample_ids)
        assert len(kept_edges) == 1

    def test_removes_orphan_edges(self):
        """Edges involving removed nodes should not be kept."""
        nodes = [
            {'id': 0, 'time': 1.0, 'is_sample': False},
            {'id': 1, 'time': 0.5, 'is_sample': False},
            {'id': 2, 'time': 0.0, 'is_sample': True},
        ]
        edges = [
            (0, 10000, 0, 2),
            (0, 10000, 0, 1),  # Node 1 has no sample descendants
        ]
        sample_ids = {2}
        kept_nodes, kept_edges = simplify_tree_sequence(nodes, edges,
                                                         sample_ids)
        # Node 1 is not ancestral to any sample
        assert 1 not in kept_nodes
        # Only the edge from 0 to 2 should remain
        assert len(kept_edges) == 1
        assert kept_edges[0] == (0, 10000, 0, 2)
