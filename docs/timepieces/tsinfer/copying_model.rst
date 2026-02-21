.. _tsinfer_copying_model:

==============================
Gear 2: The Copying Model
==============================

   *Every genome is a patchwork quilt, stitched from ancestral cloth by the
   needle of recombination.*

The copying model is tsinfer's workhorse: a Li & Stephens Hidden Markov Model
that expresses one haplotype as a **mosaic** of reference haplotypes. It is
used twice -- once to match ancestors against older ancestors, and once to
match samples against the ancestor tree. This chapter derives the model
from scratch and implements the Viterbi algorithm that finds the best
mosaic path.

If tsinfer is a quartz movement, the copying model is its **oscillator
circuit** -- the component that drives every tick. In a quartz watch, the
crystal vibrates at a precise frequency and the circuit counts those
vibrations. Here, the HMM "vibrates" through hidden states (which reference
haplotype is being copied) at each site, and the Viterbi algorithm counts
out the most likely sequence of states. Without this engine, neither
ancestor matching (Gear 3) nor sample matching (Gear 4) can function.

.. admonition:: Relationship to the Li & Stephens Timepiece

   This chapter covers the specific parameterization and Viterbi
   implementation used by tsinfer. For the full derivation of the
   Li & Stephens model -- initial distribution, transition structure,
   emission probabilities, the :math:`O(n)` trick, and forward-backward
   algorithms -- see the :ref:`Li & Stephens Timepiece <lshmm_timepiece>`.
   We reference those results here and focus on what tsinfer does differently.

.. admonition:: Prerequisites

   Make sure you have read:

   - :ref:`tsinfer_ancestor_generation` (Gear 1), so you understand what
     the ancestors look like and why they have limited genomic extent
   - The :ref:`Li & Stephens HMM chapter <lshmm_timepiece>`, for the
     foundational derivation of the copying model, transition and emission
     probabilities, and the :math:`O(k)` computational trick


The Copying Metaphor
=====================

Imagine constructing a new haplotype by **copying** from a panel of
:math:`k` reference haplotypes. At each site, you copy the allele from
one reference. Between adjacent sites, you may **switch** to a different
reference (recombination) or **stay** with the current one. Occasionally,
the copied allele is **mutated** so it doesn't match the reference.

In tsinfer's context:

- During **ancestor matching**: the "query" is an ancestor, the "panel"
  is the set of older ancestors already in the tree
- During **sample matching**: the "query" is a sample, the "panel" is
  the complete set of ancestors

The HMM hidden state :math:`Z_\ell` at site :math:`\ell` is the index of
the reference haplotype being copied. The observation :math:`X_\ell` is
the query allele at site :math:`\ell`.

Now let's formalize the two components of the HMM: transitions (how
the copying source changes between sites) and emissions (how likely the
observed allele is, given the copying source).


Step 1: Transition Probabilities
==================================

The transition probability governs how the copying source changes between
adjacent sites. tsinfer uses the standard Li & Stephens formulation:

.. math::

   P(Z_\ell = j \mid Z_{\ell-1} = i) =
   \begin{cases}
   1 - \rho + \rho / k & \text{if } i = j \quad \text{(stay)} \\
   \rho / k & \text{if } i \neq j \quad \text{(switch)}
   \end{cases}

where :math:`k` is the number of reference haplotypes in the panel and
:math:`\rho` is the recombination probability.

**The recombination probability** between sites :math:`\ell - 1` and
:math:`\ell` is computed from the genetic distance :math:`d_\ell`
(in base pairs or genetic map units):

.. math::

   \rho_\ell = 1 - e^{-d_\ell \cdot r_{\text{rate}} / k}

where :math:`r_{\text{rate}}` is the per-unit recombination rate and the
:math:`1/k` scaling follows from the Li & Stephens approximation (see
:ref:`copying_model` for the coalescent justification).

**Why divide by** :math:`k` **?** With more reference haplotypes, each one
covers a smaller fraction of the genealogical space. After a recombination,
the probability of landing on any specific haplotype decreases proportionally.

.. admonition:: Probability Aside -- The :math:`1/k` Scaling

   The :math:`1/k` factor in the recombination probability has a coalescent
   interpretation. In a panel of :math:`k` haplotypes, the coalescent
   rate between any two lineages is :math:`1/k` (up to constants). When a
   recombination occurs, the new lineage "lands" on one of the :math:`k`
   references with equal probability :math:`1/k`. As :math:`k` grows, each
   individual reference becomes less likely to be the recipient of a switch,
   but the total switching probability (summed over all :math:`k` alternatives)
   remains :math:`\rho`. This ensures the model is self-consistent regardless
   of panel size.

.. code-block:: python

   import numpy as np

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

   # Example: 10 sites, uniform spacing
   positions = np.arange(0, 10000, 1000, dtype=float)
   rho = compute_recombination_probs(positions, recombination_rate=1e-4,
                                      num_ref=50)
   print(f"Recombination probabilities: {np.round(rho, 6)}")
   print(f"Sum of row for k=50: stay + 49*switch = "
         f"{(1-rho[1]) + 49*rho[1]/50:.6f}")  # Should be 1.0

With transitions defined, we now turn to how the observed allele relates
to the hidden copying source.


Step 2: Emission Probabilities
================================

The emission probability governs how likely the query allele is, given the
reference allele being copied.

tsinfer uses a slightly different parameterization from the standard
Li & Stephens model. The **mismatch probability** :math:`\mu_\ell` at site
:math:`\ell` is computed from the genetic distance and a **mismatch ratio**:

.. math::

   \mu_\ell = 1 - e^{-d_\ell \cdot r_{\text{rate}} \cdot \text{ratio} / k}

where :math:`\text{ratio}` is the mismatch-to-recombination ratio (typically
a small value like 1.0).

The emission probabilities are then:

.. math::

   P(X_\ell \mid Z_\ell = j) =
   \begin{cases}
   1 - \mu_\ell & \text{if } X_\ell = H_{j\ell} \quad \text{(match)} \\
   \mu_\ell & \text{if } X_\ell \neq H_{j\ell} \quad \text{(mismatch)}
   \end{cases}

For biallelic sites, there's only one alternative allele, so the full
mutation probability goes to the mismatch.

**Why use genetic distance for both** :math:`\rho` **and** :math:`\mu` **?**
tsinfer assumes that sites are not uniformly spaced. Two sites 10 kb apart
should have higher recombination *and* mismatch probabilities than two sites
10 bp apart. The mismatch ratio controls the relative strength of mutation
vs. recombination.

.. admonition:: Probability Aside -- Mismatch vs. Mutation Rate

   The mismatch probability :math:`\mu_\ell` is *not* the biological mutation
   rate. It is a model parameter that controls how tolerant the HMM is of
   disagreements between the query and the reference. A low :math:`\mu` (say
   0.001) means mismatches are very unlikely and the model strongly prefers
   switching to a different reference over tolerating a mismatch. A high
   :math:`\mu` (say 0.1) makes the model tolerant of mismatches and
   reluctant to switch. The mismatch ratio :math:`\mu / \rho` controls
   this trade-off: values less than 1 prefer switching over mismatching,
   values greater than 1 prefer mismatching over switching. In practice,
   a ratio near 1.0 works well for most datasets.

.. code-block:: python

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

   # Example
   mu = compute_mismatch_probs(positions, recombination_rate=1e-4,
                                mismatch_ratio=1.0, num_ref=50)
   print(f"Mismatch probabilities: {np.round(mu, 6)}")

With both transition and emission probabilities in place, we can now
implement the algorithm that finds the best mosaic path: the Viterbi
algorithm.


Step 3: The Viterbi Algorithm
===============================

Unlike SINGER (which uses forward-backward and stochastic traceback),
tsinfer uses the **Viterbi algorithm**: it finds the single most likely
path through the HMM. This is appropriate because tsinfer produces a
point estimate of the tree sequence, not posterior samples.

.. admonition:: Probability Aside -- Viterbi vs. Forward-Backward

   The **forward-backward algorithm** computes the marginal posterior
   probability of each hidden state at each site:
   :math:`P(Z_\ell = j \mid X_1, \ldots, X_m)`. This is useful when you
   want to know how *uncertain* the copying source is at each position.
   The **Viterbi algorithm** instead finds the single most probable
   *sequence* of hidden states:
   :math:`\arg\max_{z_1,\ldots,z_m} P(Z=z \mid X)`. These are different
   questions! The marginal mode at each site need not equal the Viterbi
   path (a phenomenon called the "Viterbi paradox"). tsinfer uses Viterbi
   because it needs a single, definite mosaic to convert into tree sequence
   edges. For a full derivation of both algorithms in the Li & Stephens
   context, see the :ref:`Li & Stephens HMM chapter <lshmm_timepiece>`.

The Viterbi recursion
-----------------------

Define :math:`V_j(\ell)` as the probability of the most likely path ending
in state :math:`j` at site :math:`\ell`:

.. math::

   V_j(\ell) = P(X_\ell \mid Z_\ell = j) \cdot
               \max_{i} \left[ V_i(\ell - 1) \cdot P(Z_\ell = j \mid Z_{\ell-1} = i) \right]

**Initialization** (site 0):

.. math::

   V_j(0) = \frac{1}{k} \cdot P(X_0 \mid Z_0 = j)

**Traceback pointer**:

.. math::

   \psi_j(\ell) = \arg\max_{i} \left[ V_i(\ell - 1) \cdot P(Z_\ell = j \mid Z_{\ell-1} = i) \right]

This records which previous state led to the maximum at :math:`(\ell, j)`.

The :math:`O(k)` trick for Viterbi
-------------------------------------

Just as with the forward algorithm (see :ref:`copying_model`), the
Li & Stephens transition structure allows us to compute each Viterbi step
in :math:`O(k)` instead of :math:`O(k^2)`.

Substituting the transition probabilities:

.. math::

   V_j(\ell) = e_j(\ell) \cdot \max\Big\{
      (1 - \rho) V_j(\ell-1), \;
      \frac{\rho}{k} \max_i V_i(\ell-1)
   \Big\}

The key insight: :math:`\max_i V_i(\ell-1)` is a single value computed
**once** in :math:`O(k)` time. Then for each state :math:`j`, we compare
two candidates:

1. **Stay**: :math:`(1 - \rho) V_j(\ell-1)` -- continue copying from :math:`j`
2. **Switch**: :math:`\frac{\rho}{k} \max_i V_i(\ell-1)` -- switch from the
   globally best state

The traceback pointer is:

- :math:`\psi_j(\ell) = j` if staying is better
- :math:`\psi_j(\ell) = i^*` (the global argmax) if switching is better

.. code-block:: python

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

   # Example: a small panel with a mosaic query
   np.random.seed(42)
   k = 5
   m = 20
   panel = np.random.binomial(1, 0.3, size=(m, k))

   # Construct a mosaic query: copy from ref 1 for first half, ref 3 for second
   true_path = np.array([1]*10 + [3]*10)
   query = np.array([panel[ell, true_path[ell]] for ell in range(m)])

   rho = np.full(m, 0.05)
   rho[0] = 0.0
   mu = np.full(m, 0.01)

   path, log_p = viterbi_ls(query, panel, rho, mu)
   accuracy = np.mean(path == true_path)
   print(f"True path:    {true_path}")
   print(f"Viterbi path: {path}")
   print(f"Accuracy: {accuracy:.0%}")
   print(f"Log probability: {log_p:.2f}")

The basic Viterbi algorithm works well when every reference spans every
site. But in tsinfer, ancestors have limited genomic extent -- they only
span a subset of sites. The next step handles this complication.


Step 4: Handling NONCOPY States
=================================

In tsinfer, the reference panel is not a simple matrix. Ancestors have
**limited genomic extent** -- they don't span all sites. At sites outside
an ancestor's interval, that ancestor is marked as **NONCOPY**, meaning
it cannot be the copying source.

This is implemented by setting the emission probability to 0 for NONCOPY
entries, which forces the Viterbi algorithm to avoid those states:

.. math::

   P(X_\ell \mid Z_\ell = j) = 0 \quad \text{if ancestor } j \text{ is NONCOPY at site } \ell

In practice, the NONCOPY status also affects the transition probabilities.
The number of "copiable" references :math:`k_\ell` varies by site, and the
switching probability uses :math:`k_\ell` instead of the total panel size:

.. math::

   P(Z_\ell = j \mid Z_{\ell-1} = i) =
   \begin{cases}
   1 - \rho_\ell + \rho_\ell / k_\ell & \text{if } i = j \text{ and } j \text{ is copiable} \\
   \rho_\ell / k_\ell & \text{if } i \neq j \text{ and } j \text{ is copiable} \\
   0 & \text{if } j \text{ is NONCOPY}
   \end{cases}

.. admonition:: Confusion Buster -- Why Ancestors Have Limited Extent

   Recall from :ref:`Gear 1 <tsinfer_ancestor_generation>` that each ancestor
   is built by extending left and right from a focal site, stopping when an
   older site is encountered or when the consensus breaks down. This means
   most ancestors do *not* span the entire genome. At sites outside an
   ancestor's interval, that ancestor simply did not exist yet (or had already
   been superseded by a different lineage), so it makes no sense to copy from
   it. The NONCOPY mechanism enforces this biological constraint within the
   HMM framework.

.. code-block:: python

   NONCOPY = -2

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

   # Example: ancestor panel where each ancestor spans a limited interval
   panel_nc = np.full((m, k), NONCOPY, dtype=int)
   # Ancestor 0 spans sites 0-14
   panel_nc[:15, 0] = np.random.binomial(1, 0.3, 15)
   # Ancestor 1 spans sites 0-19 (full)
   panel_nc[:, 1] = np.random.binomial(1, 0.3, m)
   # Ancestor 2 spans sites 5-19
   panel_nc[5:, 2] = np.random.binomial(1, 0.3, 15)
   # Ancestors 3, 4 span full range
   panel_nc[:, 3] = np.random.binomial(1, 0.3, m)
   panel_nc[:, 4] = np.random.binomial(1, 0.3, m)

   query_nc = np.random.binomial(1, 0.3, m)
   path_nc = viterbi_ls_with_noncopy(query_nc, panel_nc, rho, mu)
   print(f"Path with NONCOPY handling: {path_nc}")

   # Verify: no NONCOPY references selected
   for ell in range(m):
       assert panel_nc[ell, path_nc[ell]] != NONCOPY, \
           f"Selected NONCOPY at site {ell}!"
   print("Verification: no NONCOPY references in path [ok]")

Now that we can find the best mosaic path, we need to convert it into
the edges that form a tree sequence. This is the bridge between the HMM
and the tree sequence data structure.


Step 5: From Viterbi Path to Edges
=====================================

The Viterbi path tells us which reference haplotype is being copied at
each site. To build a tree sequence, we convert this path into **edges**:
contiguous segments where the same reference is the parent.

An edge is a tuple :math:`(l, r, \text{parent}, \text{child})` meaning:
"over the genomic interval :math:`[l, r)`, node ``parent`` is the parent
of node ``child``."

This conversion is the moment where the HMM output becomes genealogical
structure -- where the oscillator circuit's signal becomes the movement
of the hands.

.. code-block:: python

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

   # Example
   positions = np.arange(0, 20000, 1000, dtype=float)
   path_example = np.array([1]*7 + [3]*8 + [1]*5)
   ref_ids = np.array([100, 101, 102, 103, 104])  # Node IDs
   edges = path_to_edges(path_example, positions, child_id=200,
                          ref_node_ids=ref_ids)

   print("Edges from Viterbi path:")
   for left, right, parent, child in edges:
       print(f"  [{left:.0f}, {right:.0f}): parent={parent}, child={child}")

   # Verify: edges should cover the full genomic range
   total = sum(r - l for l, r, _, _ in edges)
   print(f"Total coverage: {total:.0f} bp")

Each transition in the Viterbi path corresponds to an inferred
recombination event. Let's extract those breakpoints explicitly.


Step 6: Recombination Breakpoints
===================================

Each transition in the Viterbi path (where the copying source changes)
represents an inferred **recombination breakpoint**. The breakpoint
is placed at the genomic position of the site where the switch occurs.

.. math::

   \text{breakpoints} = \{p_\ell : \text{path}[\ell] \neq \text{path}[\ell - 1]\}

In the tree sequence, each breakpoint creates a new set of edges: the
old parent-child edge ends, and a new one begins.

.. code-block:: python

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

   # Example
   bps = find_breakpoints(path_example, positions)
   print(f"Breakpoints ({len(bps)}):")
   for pos, from_ref, to_ref in bps:
       print(f"  Position {pos:.0f}: ref {from_ref} -> ref {to_ref}")


Verification
=============

Let's verify the Viterbi implementation against a known scenario:

.. code-block:: python

   def verify_viterbi():
       """Verify Viterbi on a fully deterministic example."""
       # Panel: 3 distinct haplotypes
       panel = np.array([
           [0, 0, 1],
           [0, 1, 0],
           [1, 0, 0],
           [1, 1, 0],
           [0, 0, 1],
       ])  # 5 sites, 3 refs

       # Query: exact copy of ref 0 at first 3 sites, ref 2 at last 2
       query = np.array([0, 0, 1, 0, 1])

       rho = np.array([0.0, 0.05, 0.05, 0.05, 0.05])
       mu = np.full(5, 0.001)

       path, log_p = viterbi_ls(query, panel, rho, mu)

       print("Verification:")
       print(f"  Query:       {query}")
       print(f"  Ref 0:       {panel[:, 0]}")
       print(f"  Ref 2:       {panel[:, 2]}")
       print(f"  Viterbi path: {path}")

       # At sites 0-2, query matches ref 0 perfectly
       # At sites 3-4, query matches ref 2 perfectly
       # So we expect path to be [0, 0, 0, 2, 2] (or similar)
       for ell in range(5):
           assert query[ell] == panel[ell, path[ell]], \
               f"Mismatch at site {ell}!"
       print("  [ok] Path has zero mismatches")
       print("  [ok] Viterbi found a valid mosaic")

   verify_viterbi()

With the copying model fully implemented, we have the engine that drives
both matching phases. In the next chapter, we use this engine to assemble
the ancestor tree -- fitting the gears together from the oldest to the
youngest.


Exercises
==========

.. admonition:: Exercise 1: Viterbi vs. forward-backward

   Implement the forward-backward algorithm for the Li & Stephens model
   (see :ref:`haploid_algorithms`). Compare the marginal posterior at each
   site (from forward-backward) with the Viterbi path. Are there sites
   where the Viterbi path disagrees with the posterior mode? When does
   this happen?

.. admonition:: Exercise 2: Effect of the mismatch ratio

   Run the Viterbi algorithm on a simulated mosaic query with varying
   mismatch ratios (0.01, 0.1, 1.0, 10.0). How does the ratio affect
   the number of breakpoints? What happens when the ratio is too low
   (too few mismatches allowed) or too high (too many)?

.. admonition:: Exercise 3: Scaling behavior

   Time the Viterbi algorithm for panel sizes :math:`k = 10, 100, 1000`
   and site counts :math:`m = 1000, 10000, 100000`. Verify that the
   runtime scales as :math:`O(mk)`. Plot the results.

Next: :ref:`tsinfer_ancestor_matching` -- using the copying model to build the ancestor tree.
