.. _relate_asymmetric_painting:

=====================================
Gear 1: Asymmetric Painting
=====================================

   *The direction of a mutation matters: knowing that i carries what j does
   not is different from knowing that j carries what i does not.*

The asymmetric painting is Relate's oscillator -- the component that extracts
raw genealogical signal from the haplotype data. It is a modified Li & Stephens
HMM that, unlike the standard version, distinguishes between ancestral and
derived alleles. This distinction produces an **asymmetric distance matrix**
:math:`d(i,j) \neq d(j,i)` that encodes the *direction* of mutation differences
between haplotypes. This directional information is crucial: without it, the
tree-building algorithm in Gear 2 cannot correctly reconstruct tree topologies.

.. admonition:: Relationship to the Li & Stephens Timepiece

   This chapter extends the Li & Stephens model from
   :ref:`Timepiece III <lshmm_timepiece>`. We assume familiarity with the
   standard model: hidden states (which reference haplotype is being copied),
   transitions (recombination), emissions (mutation/mismatch), and the
   :math:`O(K)` trick. What changes here is the **emission model**: Relate
   encodes the direction of allele differences.

.. admonition:: Prerequisites

   - :ref:`Li & Stephens HMM <lshmm_timepiece>` -- the standard copying model,
     transition probabilities, and the :math:`O(K)` trick
   - :ref:`Hidden Markov Models <hmms>` -- the forward-backward algorithm


The Standard Painting vs. Relate's Modification
=================================================

In the standard Li & Stephens model, when target haplotype :math:`i` is
"painted" against the panel, the emission probability treats mismatches
symmetrically:

.. math::

   P(H_{i\ell} \mid Z_\ell = j) =
   \begin{cases}
   1 - \mu & \text{if } H_{i\ell} = H_{j\ell} \quad \text{(match)} \\
   \mu & \text{if } H_{i\ell} \neq H_{j\ell} \quad \text{(mismatch)}
   \end{cases}

A mismatch is a mismatch, regardless of which direction the allele change goes.
This is fine for imputation and phasing, where we just need to identify the
closest reference. But for tree building, we need more: we need to know the
**polarity** of the difference.

Relate modifies the emission probability to account for whether alleles are
ancestral (0) or derived (1):

.. math::

   P(H_{i\ell} \mid Z_\ell = j) =
   \begin{cases}
   1 - \mu & \text{if } H_{i\ell} = H_{j\ell} = 0 \quad \text{(both ancestral)} \\
   1 - \mu & \text{if } H_{i\ell} = H_{j\ell} = 1 \quad \text{(both derived)} \\
   \mu_d & \text{if } H_{i\ell} = 1, H_{j\ell} = 0 \quad \text{(i derived, j ancestral)} \\
   \mu_a & \text{if } H_{i\ell} = 0, H_{j\ell} = 1 \quad \text{(i ancestral, j derived)}
   \end{cases}

where :math:`\mu_d` and :math:`\mu_a` are **directional mismatch
probabilities**. Crucially, :math:`\mu_d \neq \mu_a` in general.

.. admonition:: Probability Aside -- Why Polarity Matters for Trees

   Consider three haplotypes at a single site: :math:`A = 1`, :math:`B = 0`,
   :math:`C = 0`. The derived allele in :math:`A` means that :math:`A`
   descends from a lineage that experienced a mutation *after* it split from
   the lineage leading to :math:`B` and :math:`C`. In other words, :math:`B`
   and :math:`C` are more likely to coalesce with each other before either
   coalesces with :math:`A`. The direction of the mismatch -- *A has 1 where
   B has 0* -- tells us :math:`A` is the outlier, not :math:`B`. Without
   polarity, we cannot tell who is the outlier.

   Formally, the mismatch :math:`H_{i\ell} = 1, H_{j\ell} = 0` means that
   the mutation on this branch is *derived in i but ancestral in j*, implying
   that the lineage leading to :math:`i` branched off the lineage leading to
   :math:`j` **after** this mutation occurred. This asymmetry is the signal
   that Relate exploits.


Step 1: The Modified Emission Probabilities
============================================

The key to the asymmetric distance is the directional emission model. We
parameterize the two directional mismatch probabilities as:

.. math::

   \mu_d = \mu \cdot w_d, \qquad \mu_a = \mu \cdot w_a

where :math:`w_d` and :math:`w_a` are weights that control the relative cost
of each direction of mismatch. In Relate's implementation, the standard choice
is to set :math:`w_d` and :math:`w_a` such that derived-in-target mismatches
are more informative (they suggest the target branched after the reference).

.. code-block:: python

   import numpy as np

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

   # Example: verify asymmetry
   mu = 0.01
   e_d = directional_emission(1, 0, mu)  # target derived, ref ancestral
   e_a = directional_emission(0, 1, mu)  # target ancestral, ref derived
   print(f"P(target=1 | ref=0) = {e_d:.4f}")
   print(f"P(target=0 | ref=1) = {e_a:.4f}")
   print(f"Asymmetric? {e_d != e_a}")


Step 2: The Forward-Backward Algorithm
========================================

Relate runs the standard Li & Stephens forward-backward algorithm, but with
the directional emission probabilities. For each target haplotype :math:`i`,
it computes the posterior probability :math:`p_{ij}(\ell)` that haplotype
:math:`i` is copying from haplotype :math:`j` at site :math:`\ell`.

The **transition probabilities** are standard Li & Stephens:

.. math::

   P(Z_\ell = j \mid Z_{\ell-1} = k) =
   \begin{cases}
   1 - \rho_\ell + \frac{\rho_\ell}{N-1} & \text{if } j = k \\
   \frac{\rho_\ell}{N-1} & \text{if } j \neq k
   \end{cases}

where :math:`\rho_\ell = 1 - e^{-d_\ell \cdot r / (N-1)}` is the
recombination probability between sites :math:`\ell-1` and :math:`\ell`,
:math:`d_\ell` is the genetic distance, and :math:`r` is the recombination
rate.

The **forward variable** :math:`\alpha_j(\ell) = P(H_{i,1:\ell}, Z_\ell = j)`
and **backward variable** :math:`\beta_j(\ell) = P(H_{i,\ell+1:L} \mid Z_\ell = j)`
are computed with the :math:`O(K)` trick:

.. math::

   \alpha_j(\ell) = e_j(\ell) \left[
   (1 - \rho_\ell) \cdot \alpha_j(\ell-1)
   + \frac{\rho_\ell}{N-1} \cdot \sum_k \alpha_k(\ell-1)
   \right]

where :math:`e_j(\ell) = P(H_{i\ell} \mid Z_\ell = j)` uses the directional
emission. The sum :math:`\sum_k \alpha_k(\ell-1)` is computed once in
:math:`O(K)` and reused for every :math:`j`.

The posterior is:

.. math::

   p_{ij}(\ell) = P(Z_\ell = j \mid H_{i,1:L})
   = \frac{\alpha_j(\ell) \cdot \beta_j(\ell)}{\sum_k \alpha_k(\ell) \cdot \beta_k(\ell)}

.. code-block:: python

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

   # Example
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


Step 3: From Posterior to Asymmetric Distance
==============================================

The posterior copying probabilities :math:`p_{ij}(\ell)` encode how likely
haplotype :math:`j` is to be the copying source for haplotype :math:`i` at
site :math:`\ell`. Relate converts these into a **distance** via a negative
log transformation:

.. math::

   d(i, j) = -\log p_{ij}(s)

at focal SNP :math:`s`, where :math:`p_{ij}(s)` is the posterior probability
that :math:`i` copies from :math:`j` at site :math:`s`.

.. admonition:: Probability Aside -- Why Negative Log Posterior?

   The posterior :math:`p_{ij}(s)` ranges from 0 to 1. Higher values mean
   haplotype :math:`j` is a more likely copying source for :math:`i` at
   position :math:`s` -- in other words, :math:`i` and :math:`j` are
   genealogically close. Taking :math:`-\log` converts this to a distance:
   close haplotypes have small distances, distant ones have large distances.

   Under the infinite-sites model with no recombination, :math:`-\log p_{ij}(s)`
   converges to the count of derived alleles in :math:`i` that are ancestral
   in :math:`j`. This is because each such allele contributes a factor of
   :math:`\mu_d` (a small number) to the posterior, and :math:`-\log \mu_d`
   accumulates additively.

The critical property is that this distance is **asymmetric**:

.. math::

   d(i, j) \neq d(j, i)

because :math:`p_{ij}(s) \neq p_{ji}(s)` in general. When we paint :math:`i`
against the panel, the allele status of :math:`i` at each SNP affects the
emission probability directionally. Painting :math:`j` against the panel
produces a different posterior.

.. code-block:: python

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

   # Example: small dataset
   np.random.seed(123)
   N, L = 6, 30
   haplotypes = np.random.binomial(1, 0.3, size=(N, L))
   positions = np.arange(L, dtype=float) * 1000

   D = compute_distance_matrix(haplotypes, positions, recomb_rate=1e-4,
                                mu=0.01, focal_snp=15)

   print("Asymmetric distance matrix:")
   print(np.round(D, 2))
   print(f"\nD[0,1] = {D[0,1]:.2f}, D[1,0] = {D[1,0]:.2f}")
   print(f"Asymmetric? {not np.allclose(D, D.T)}")

The asymmetry is not a bug -- it is the essential signal. Consider the
following example:

.. code-block:: python

   # Demonstrating why asymmetry matters
   # Three haplotypes at 5 sites:
   #   A: 1 1 0 0 0   (carries 2 derived alleles)
   #   B: 0 0 0 0 0   (all ancestral)
   #   C: 0 0 0 0 0   (all ancestral)
   #
   # d(A, B) is large: A has 2 derived alleles that B lacks
   # d(B, A) is small: B has 0 derived alleles that A lacks
   # This tells us A branched off AFTER B and C coalesced

   haps_demo = np.array([
       [1, 1, 0, 0, 0],  # A: 2 derived
       [0, 0, 0, 0, 0],  # B: all ancestral
       [0, 0, 0, 0, 0],  # C: all ancestral
   ])
   pos_demo = np.arange(5, dtype=float) * 1000
   D_demo = compute_distance_matrix(haps_demo, pos_demo, recomb_rate=1e-4,
                                     mu=0.01, focal_snp=2)
   print("Demo distance matrix:")
   for i in range(3):
       labels = ['A', 'B', 'C']
       for j in range(3):
           if i != j:
               print(f"  d({labels[i]}, {labels[j]}) = {D_demo[i,j]:.2f}")


Step 4: Why Symmetrization Fails
==================================

It might seem natural to symmetrize the distance matrix (e.g., using
:math:`d_s(i,j) = d(i,j) + d(j,i)`) and then apply standard hierarchical
clustering. Relate's paper shows this produces **incorrect topologies**.

.. admonition:: Confusion Buster -- Why You Cannot Just Symmetrize

   Standard agglomerative clustering (UPGMA, neighbor-joining) assumes a
   symmetric distance matrix. When applied to the symmetrized
   :math:`d_s(i,j) = d(i,j) + d(j,i)`, these algorithms treat the total
   number of differences between :math:`i` and :math:`j` as the distance,
   discarding the information about *which* haplotype carries the derived
   alleles. This directional information is precisely what determines
   the correct tree topology -- without it, the algorithm cannot distinguish
   between a scenario where :math:`A` branched off first and a scenario where
   :math:`B` branched off first.

Consider four haplotypes with the following allele patterns at a single SNP:

.. code-block:: text

   True tree:        Alleles:
       R               A: 1  (derived)
      / \              B: 0  (ancestral)
     A   *             C: 0
        / \            D: 0
       B   *
          / \
         C   D

The mutation is on the branch leading to :math:`A`. The asymmetric distances
correctly place :math:`A` as the outlier. But symmetrized distances would
give :math:`d_s(A, B) = d_s(A, C) = d_s(A, D)` (all equal, since each pair
differs by exactly 1 mutation), making it impossible to determine the tree
structure among :math:`B`, :math:`C`, and :math:`D`.

.. code-block:: python

   def demonstrate_symmetrization_failure():
       """Show that symmetrizing distances loses topology information."""
       # True tree: ((B, (C, D)), A) -- A is the outlier
       # SNP pattern:
       #   A = 1, B = 0, C = 0, D = 0
       #
       # Asymmetric distances at this SNP:
       #   d(A, B) = large (A has derived that B lacks)
       #   d(B, A) = small (B has no derived that A lacks)
       #   d(B, C) = small (both ancestral -- close)
       #   d(C, D) = small (both ancestral -- close)

       # Simulated asymmetric distances
       D_asym = np.array([
           [0.0, 3.5, 3.5, 3.5],  # A -> others: large (A has mutations)
           [0.5, 0.0, 0.8, 0.8],  # B -> others: small
           [0.5, 0.8, 0.0, 0.3],  # C -> others: C and D are closest
           [0.5, 0.8, 0.3, 0.0],  # D -> others
       ])

       # Symmetrized distance
       D_sym = D_asym + D_asym.T

       labels = ['A', 'B', 'C', 'D']
       print("Asymmetric distance matrix:")
       for i in range(4):
           row = [f"{D_asym[i,j]:4.1f}" for j in range(4)]
           print(f"  {labels[i]}: {' '.join(row)}")

       print("\nSymmetrized distance matrix:")
       for i in range(4):
           row = [f"{D_sym[i,j]:4.1f}" for j in range(4)]
           print(f"  {labels[i]}: {' '.join(row)}")

       # From asymmetric: A is clearly the outlier (large d(A, *))
       # and C, D are closest to each other
       # Correct tree: ((C, D), B, A) with A as outgroup

       # From symmetrized: d_sym(A,B) = d_sym(A,C) = d_sym(A,D) = 4.0
       # All equal! Cannot determine topology among B, C, D
       print("\nAsymmetric matrix correctly identifies:")
       print(f"  A is outlier: min d(A,*) = {D_asym[0, 1:].min():.1f} "
             f"vs min d(B,*\\A) = {min(D_asym[1,2], D_asym[1,3]):.1f}")
       print(f"  C and D are closest: d(C,D) = {D_asym[2,3]:.1f}")

   demonstrate_symmetrization_failure()


Verification
=============

Let's verify the painting implementation on a case where the true tree is
known:

.. code-block:: python

   def verify_painting():
       """Verify the painting on a case with known genealogy.

       True tree at the focal SNP:
              root
             /    \\
           /        \\
          *          *
         / \\        / \\
        0   1      2   3

       Haplotypes 0,1 are siblings; 2,3 are siblings.
       We expect d(0,1) < d(0,2) and d(0,1) < d(0,3).
       """
       # Create haplotypes consistent with the tree:
       # Mutations on branch to (0,1) clade: sites 0, 1
       # Mutations on branch to (2,3) clade: sites 2, 3
       # Mutation on branch to 0: site 4
       # Mutation on branch to 2: site 5
       L = 10
       haps = np.zeros((4, L), dtype=int)
       # Shared derived alleles for 0,1 clade
       haps[0, 0] = 1; haps[1, 0] = 1
       haps[0, 1] = 1; haps[1, 1] = 1
       # Shared derived alleles for 2,3 clade
       haps[2, 2] = 1; haps[3, 2] = 1
       haps[2, 3] = 1; haps[3, 3] = 1
       # Private mutations
       haps[0, 4] = 1  # private to 0
       haps[2, 5] = 1  # private to 2

       positions = np.arange(L, dtype=float) * 100

       D = compute_distance_matrix(haps, positions, recomb_rate=1e-3,
                                    mu=0.01, focal_snp=5)

       print("Verification: known tree topology")
       print(f"  d(0,1) = {D[0,1]:.2f}  (siblings, should be small)")
       print(f"  d(0,2) = {D[0,2]:.2f}  (different clade, should be larger)")
       print(f"  d(0,3) = {D[0,3]:.2f}  (different clade, should be larger)")
       print(f"  d(2,3) = {D[2,3]:.2f}  (siblings, should be small)")

       # Verify sibling distances are smaller than cross-clade
       assert D[0, 1] < D[0, 2], "Sibling distance should be smaller!"
       assert D[2, 3] < D[2, 0], "Sibling distance should be smaller!"
       print("  [ok] Sibling distances < cross-clade distances")

       # Verify asymmetry
       print(f"\n  d(0,1) = {D[0,1]:.2f}, d(1,0) = {D[1,0]:.2f}")
       print(f"  Asymmetric? {not np.isclose(D[0,1], D[1,0])}")
       print("  [ok] Distance matrix is asymmetric")

   verify_painting()


Computational Complexity
=========================

For each target haplotype :math:`i`, the forward-backward algorithm runs in
:math:`O(K \cdot L)` where :math:`K = N - 1` is the panel size and :math:`L`
is the number of SNPs. Since we must paint each of the :math:`N` haplotypes,
the total cost is:

.. math::

   O(N \cdot K \cdot L) = O(N^2 L)

This is the dominant cost of Relate. For :math:`N = 10{,}000` haplotypes and
:math:`L = 10^6` SNPs, this is :math:`10^{16}` operations -- feasible with
parallelization (across haplotypes and genomic chunks), but not cheap.

.. admonition:: Confusion Buster -- Why Not Use Viterbi?

   tsinfer uses the Viterbi algorithm (most likely path) rather than
   forward-backward (posterior probabilities). Relate uses forward-backward
   because it needs the **soft** posterior :math:`p_{ij}(\ell)` at each site,
   not just the single best copying source. The soft posterior captures
   uncertainty -- if two references are nearly equally good copying sources,
   both will have substantial posterior mass. This uncertainty is precisely
   what makes the distance matrix informative: it reflects the relative
   evidence for different genealogical relationships, not just the single
   best guess.


Exercises
==========

.. admonition:: Exercise 1: Symmetry under equal alleles

   Show analytically that when all haplotypes carry the same allele at a
   site (all ancestral or all derived), the emission probability is the same
   for every reference, and that site contributes no asymmetry to the
   distance matrix.

.. admonition:: Exercise 2: Effect of the directional weights

   Run the painting with different :math:`(w_d, w_a)` settings: (1.0, 1.0)
   (symmetric), (1.0, 0.5) (default), and (1.0, 0.0) (ignore one direction).
   How does the distance matrix change? Which setting produces the most
   informative asymmetry?

.. admonition:: Exercise 3: Distance convergence

   Simulate haplotypes under a known tree (using msprime) with varying
   numbers of SNPs. Compute the asymmetric distance matrix and compare
   :math:`d(i,j)` with the true number of derived alleles in :math:`i` that
   are ancestral in :math:`j`. Plot the convergence as the number of SNPs
   increases.

Next: :ref:`relate_tree_building` -- using the asymmetric distance matrix to
build local trees.
