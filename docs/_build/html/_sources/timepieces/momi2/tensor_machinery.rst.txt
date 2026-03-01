.. _tensor_machinery:

=========================
Tensor Machinery
=========================

   *The gear train of the watch: assembling the expected SFS from a product of tensors, one demographic event at a time.*

.. epigraph::

   "The expected SFS can be written as a tensor product over the events in the
   demographic history."

   -- Kamm, Terhorst, Song, and Durbin (2017)

.. admonition:: Biology Aside -- From populations to tensors

   Imagine you have sequenced individuals from three human populations --
   say, Yoruba, Han Chinese, and French. At each polymorphic site in the
   genome, you can count how many derived alleles appear in each sample:
   perhaps 3 out of 20 Yoruba chromosomes, 7 out of 20 Han Chinese
   chromosomes, and 5 out of 20 French chromosomes. The three-dimensional
   table that tallies all such combinations across the genome is the
   **joint site frequency spectrum** (joint SFS). It is the multi-population
   generalization of the SFS introduced in the moments timepiece.

   The likelihood tensor is ``momi2``'s internal representation for computing
   the *expected* joint SFS under a demographic model. Each axis of the
   tensor corresponds to one population's allele count. The tensor
   machinery described in this chapter shows how to build this expected
   SFS piece by piece, by walking backward through the evolutionary history
   of the populations -- from present-day samples back to the common
   ancestral population.

Step 1: What Is a Likelihood Tensor?
======================================

At the heart of ``momi2``'s computation is the **likelihood tensor** -- a
multi-dimensional array that tracks the probability of observing a given allele
configuration across populations.

For a single population with sample size :math:`n`, the likelihood tensor is a
vector :math:`L` of length :math:`n+1`, where :math:`L_i` represents the
probability weight associated with :math:`i` derived alleles in the sample.

For :math:`k` populations with sample sizes :math:`n_1, \ldots, n_k`, the
likelihood tensor is a :math:`k`-dimensional array of shape
:math:`(n_1+1) \times \cdots \times (n_k+1)`.

The tensor is initialized at the **leaves** of the demographic event tree (the
sampled populations at present) and transformed by each event as we traverse
the tree backward in time toward the root (the ancestral population).

.. code-block:: python

   import numpy as np

   def initialize_leaf_tensor(n):
       """Initialize the likelihood tensor for a leaf (sampled population).

       For computing the full SFS, the initial tensor is the identity:
       L_i = delta_{i, config[pop]}, iterated over all configurations.

       For a single configuration (b_1, ..., b_k), population p
       gets an indicator vector: L[b_p] = 1, all others 0.
       """
       # For the full expected SFS computation, momi2 uses a batch
       # approach: all configurations are processed simultaneously.
       # Here we show the single-config case for clarity.
       L = np.zeros(n + 1)
       return L  # set L[b] = 1 for the desired allele count b

Step 2: The Three Core Operations
===================================

Every computation in ``momi2`` is built from three tensor operations:

**1. Matrix multiplication (Moran transition)**

When the event tree passes through an epoch of duration :math:`t` for population
:math:`k`, the likelihood tensor is multiplied along axis :math:`k` by the Moran
transition matrix :math:`P(t)`:

.. math::

   L'_{i_1, \ldots, i_k, \ldots} = \sum_{j_k} L_{i_1, \ldots, j_k, \ldots} \, P_{j_k, i_k}(t)

This accounts for all possible drift and coalescence events during the epoch.

.. admonition:: Plain-language summary -- Matrix multiplication on tensors

   Think of the tensor as a multi-dimensional spreadsheet where each cell
   holds the probability of a particular allele configuration. When time
   passes in one population (while the others are frozen), allele frequencies
   in that population change due to drift. The matrix multiplication
   "smears" the probabilities along that population's axis, reflecting the
   fact that a configuration with 5 derived alleles might evolve into one
   with 4, 5, or 6 alleles. The Moran transition matrix (from the previous
   chapter) tells us exactly how much probability flows between each pair of
   counts.

.. code-block:: python

   def apply_moran_transition(tensor, axis, t, n):
       """Apply Moran transition to a specific axis of the tensor."""
       P = moran_transition(t, n)  # (n+1) x (n+1) matrix
       # move the target axis to the last position, multiply, move back
       tensor = np.moveaxis(tensor, axis, -1)
       tensor = tensor @ P
       tensor = np.moveaxis(tensor, -1, axis)
       return tensor

**2. Convolution (population merge)**

When two populations merge into one (a split event, viewed backward), the
lineages from both child populations enter a single ancestral population. The
likelihood tensors from the two children are combined by **convolution**: the
number of derived alleles in the ancestor is the sum of derived alleles from
each child.

.. admonition:: Biology Aside -- Population splits as merges

   In demographic models, we describe events forward in time: an ancestral
   population **splits** into two daughter populations that then evolve
   independently (for example, the human-Neanderthal divergence). But
   ``momi2`` computes the SFS by tracing lineages **backward** in time --
   from present-day samples toward the ancestral population. Viewed backward,
   a population split becomes a **merge**: the lineages from the two daughter
   populations enter the same ancestral gene pool. The convolution operation
   reflects this -- if 3 derived alleles came from population A and 4 from
   population B, the ancestor had 7 derived alleles. All possible
   combinations must be summed over, weighted by their probabilities.

For child populations with :math:`n_1` and :math:`n_2` lineages, the naive
convolution sums over all ways to partition :math:`i` derived alleles between
the two children:

.. math::

   L^{\text{anc}}_{i} = \sum_{j+k=i} L^{(1)}_{j} \cdot L^{(2)}_{k}

However, this simple sum does not account for the **combinatorics of sampling**.
The probability that the ancestor has :math:`i` derived alleles out of
:math:`n_1 + n_2` total, given that child 1 contributes :math:`j` out of
:math:`n_1` and child 2 contributes :math:`k` out of :math:`n_2`, involves
the hypergeometric distribution. The correct formula weights each partition
by the number of ways to arrange the alleles:

.. math::

   L^{\text{anc}}_{i} = \frac{1}{\binom{n_1+n_2}{i}} \sum_{j+k=i} \binom{n_1}{j} L^{(1)}_j \cdot \binom{n_2}{k} L^{(2)}_k

**Where do the binomial coefficients come from?** The term :math:`\binom{n_1}{j}`
counts how many ways :math:`j` of the :math:`n_1` lineages from child 1 can
carry the derived allele, and :math:`\binom{n_2}{k}` does the same for child 2.
Their product :math:`\binom{n_1}{j}\binom{n_2}{k}` is the number of ways to
assign derived alleles across both children such that :math:`j + k = i`. Dividing
by :math:`\binom{n_1+n_2}{i}` -- the total number of ways to choose :math:`i`
derived out of :math:`n_1 + n_2` -- normalizes the result. This is exactly the
hypergeometric probability:

.. math::

   P(j \text{ from child 1} \mid i \text{ total}) = \frac{\binom{n_1}{j}\binom{n_2}{i-j}}{\binom{n_1+n_2}{i}}

In practice, ``momi2`` first multiplies each tensor by binomial coefficients,
convolves via polynomial multiplication, then divides out the ancestral binomial
coefficients -- which is algebraically equivalent but computationally efficient.

.. code-block:: python

   from scipy.special import comb

   def convolve_populations(L1, L2, n1, n2):
       """Merge two population tensors via convolution.

       L1: likelihood vector for child population 1, length n1+1
       L2: likelihood vector for child population 2, length n2+1

       Returns: likelihood vector for ancestral population, length n1+n2+1
       """
       # weight by binomial coefficients
       b1 = np.array([comb(n1, j, exact=True) for j in range(n1 + 1)])
       b2 = np.array([comb(n2, k, exact=True) for k in range(n2 + 1)])
       weighted_L1 = L1 * b1
       weighted_L2 = L2 * b2

       # convolve (polynomial multiplication)
       conv = np.convolve(weighted_L1, weighted_L2)

       # divide out ancestral binomial coefficients
       n_anc = n1 + n2
       b_anc = np.array([comb(n_anc, i, exact=True) for i in range(n_anc + 1)])
       L_anc = conv / b_anc

       return L_anc

.. code-block:: python

   # Verification: convolving two uniform vectors should produce a valid
   # probability distribution (after normalization)
   n1, n2 = 5, 5
   L1 = np.ones(n1 + 1) / (n1 + 1)
   L2 = np.ones(n2 + 1) / (n2 + 1)
   L_anc = convolve_populations(L1, L2, n1, n2)
   assert len(L_anc) == n1 + n2 + 1

**3. Antidiagonal summation (coalescence within an epoch)**

During each epoch, coalescence events reduce the number of lineages. The
contribution of each epoch to the SFS is accumulated by contracting the
likelihood tensor with the **truncated SFS** of that epoch (computed from the
expected coalescence times :math:`E[T_{jj}]` and the W-matrix):

.. math::

   \text{sfs\_contribution} = \sum_{i} L_i \cdot \text{truncated\_sfs}_i

This is where the accumulated SFS gets its entries: each epoch contributes
mutations that occurred during that epoch, weighted by the likelihood of
observing them in the current allele configuration.

Step 3: The Junction Tree Algorithm
=====================================

A demographic model is a tree (or DAG, when admixture is present) of events.
``momi2`` processes this tree using a **post-order traversal** (leaves to root),
which corresponds to moving backward in time from the present to the ancestral
population.

.. admonition:: Biology Aside -- The demographic event tree mirrors evolutionary history

   The event tree is a direct representation of the evolutionary relationships
   among populations. The leaves are today's populations (sampled individuals
   from, say, Europe, Africa, and East Asia). Internal nodes correspond to
   historical events: population splits (divergence), size changes
   (bottlenecks, expansions), and admixture pulses (gene flow). The root
   represents the ancestral population from which all sampled populations
   ultimately descend. By processing this tree from leaves to root, ``momi2``
   incrementally builds up the expected SFS, accumulating the contribution of
   mutations that fell on branches at each stage of the history.

.. code-block:: text

   Present (leaves)
     |         |
     Pop A     Pop B
     |         |
     |  Moran  |  Moran transition
     |  trans.  |  (epoch 1)
     |         |
     +---------+
          |
      Merge (split event, backward)
          |
          |  Moran transition
          |  (epoch 2)
          |
        Root (ancestral population)

At each node in the tree:

1. **Leaf node**: Initialize the likelihood tensor
2. **Epoch boundary**: Apply the Moran transition matrix for the elapsed time
3. **Merge node**: Convolve the likelihood tensors of the two child populations
4. **Pulse node**: Apply the admixture tensor (see Step 4)
5. **Accumulate SFS**: Add the current epoch's contribution to the running SFS

.. code-block:: python

   import networkx as nx

   def compute_expected_sfs(demography, sample_sizes):
       """Compute the expected SFS by traversing the event tree.

       Simplified sketch of momi2's algorithm.
       """
       # build the event tree from the demographic model
       event_tree = demography.event_tree

       # dictionary to hold likelihood tensors, keyed by population name
       tensors = {}
       sfs = 0.0  # accumulated expected SFS

       for event in nx.dfs_postorder_nodes(event_tree):
           if event.type == 'leaf':
               n = sample_sizes[event.pop]
               tensors[event.pop] = initialize_leaf_tensor(n)

           elif event.type == 'epoch':
               pop = event.pop
               t = event.scaled_time  # 2*T/N or integral for exp growth
               n = tensors[pop].shape[-1] - 1
               # accumulate SFS contribution from this epoch
               trunc_sfs = compute_truncated_sfs(n, t, event.size_history)
               sfs = sfs + contract_with_truncated_sfs(tensors[pop], trunc_sfs)
               # apply Moran transition
               tensors[pop] = apply_moran_transition(
                   tensors[pop], axis=-1, t=t, n=n)

           elif event.type == 'merge':
               child1, child2 = event.children
               parent = event.parent
               tensors[parent] = convolve_populations(
                   tensors[child1], tensors[child2],
                   n1=tensors[child1].shape[-1] - 1,
                   n2=tensors[child2].shape[-1] - 1)
               del tensors[child1], tensors[child2]

           elif event.type == 'pulse':
               apply_admixture_tensor(tensors, event)

       return sfs

Step 4: Admixture (Pulse) Events
==================================

Admixture is the most complex operation. When a fraction :math:`f` of population
:math:`A`'s ancestry comes from population :math:`B`, each lineage in :math:`A`
independently "jumps" to :math:`B` with probability :math:`f` (viewed backward
in time).

.. admonition:: Biology Aside -- What admixture looks like in genomes

   Admixture is the mixing of previously separated populations -- for example,
   gene flow between Neanderthals and modern humans approximately 50,000
   years ago left ~2% of Neanderthal DNA in non-African genomes today.
   Viewed backward in time, this means that for each chromosomal segment
   in a modern non-African individual, there is roughly a 2% chance that
   its lineage "jumps" from the modern human gene pool into the Neanderthal
   one at the time of admixture. The binomial 3-tensor below encodes exactly
   this stochastic assignment: given :math:`n` lineages, each independently
   chooses one of two ancestral populations with probability :math:`f` or
   :math:`1 - f`.

If population :math:`A` has :math:`n` lineages, the number that move to
:math:`B` follows a binomial distribution. This is encoded as a **3-tensor**
:math:`T` of shape :math:`(n+1) \times (n+1) \times (n+1)`:

.. math::

   T_{i, j, k} = \binom{n}{k} \binom{k}{j} f^j (1-f)^{k-j} \cdot \mathbf{1}[i + j = k]

where :math:`k` is the original count in :math:`A`, :math:`j` lineages move to
:math:`B`, and :math:`i = k - j` remain in :math:`A`.

.. code-block:: python

   from scipy.special import comb as binom

   def admixture_tensor(n, f):
       """Compute the admixture 3-tensor for a pulse event.

       n: number of lineages in the receiving population
       f: fraction of ancestry from the source population

       Returns T of shape (n+1, n+1, n+1):
       T[i, j, k] = probability that k lineages split into i staying and j moving
       """
       T = np.zeros((n + 1, n + 1, n + 1))
       for k in range(n + 1):
           for j in range(k + 1):
               i = k - j
               T[i, j, k] = binom(k, j) * f**j * (1 - f)**(k - j)
       return T

   def apply_admixture(L_A, L_B, n_A, n_B, f):
       """Apply an admixture pulse: fraction f of A's ancestry comes from B.

       Returns updated likelihood tensors for both populations.
       """
       T = admixture_tensor(n_A, f)
       # contract: new_L[i_A, i_B] = sum_k sum_j L_A[k] * L_B[i_B + j] * T[i_A, j, k]
       # (simplified -- actual implementation handles multi-dimensional tensors)
       return new_L_A, new_L_B

.. admonition:: Probability Aside -- Why admixture requires a 3-tensor

   In a population merge (split backward in time), every lineage from child
   :math:`A` and every lineage from child :math:`B` ends up in the same
   ancestral population. The operation is deterministic given the lineage counts.
   In admixture, however, each lineage *independently* chooses which source
   population to join. This stochastic splitting requires a full tensor to
   encode all possible binomial outcomes.

Step 5: Reducing Lineage Counts
================================

As we move backward in time, the number of lineages in a population can only
decrease (due to coalescence). After a merge event, the ancestral population has
:math:`n_1 + n_2` lineages, which may be larger than needed for the computation.

``momi2`` can optionally reduce the lineage count via the **hypergeometric
quasi-inverse**: a matrix that projects from :math:`N` lineages down to
:math:`n < N` lineages in a way that preserves the expected SFS. This is the
reverse of the projection (downsampling) operation used in SFS analysis.

.. admonition:: Plain-language summary -- Reducing lineage counts

   After two populations merge, the ancestral population suddenly has
   :math:`n_1 + n_2` lineages -- which may be more than needed and would
   make subsequent computations expensive. The hypergeometric quasi-inverse
   is a principled way to "thin" the lineages down to a manageable number
   without distorting the expected SFS. Think of it as subsampling the
   ancestral chromosomes in a way that preserves the statistical properties
   we care about. The hypergeometric distribution appears here because it
   describes sampling without replacement from a finite pool -- the same
   distribution that governs how allele counts change when you subsample a
   dataset.

.. code-block:: python

   def hypergeom_quasi_inverse(N, n):
       """Compute the quasi-inverse for reducing lineage count from N to n.

       Returns a (N+1) x (n+1) matrix M such that applying M to a likelihood
       vector of length N+1 produces a valid likelihood vector of length n+1,
       preserving the expected SFS.
       """
       from scipy.stats import hypergeom
       M = np.zeros((N + 1, n + 1))
       for i in range(N + 1):
           for j in range(n + 1):
               M[i, j] = hypergeom.pmf(j, N, i, n)
       return M

Step 6: Putting It All Together -- A Two-Population Example
=============================================================

Let's trace through the tensor computation for a simple two-population model:
populations A and B diverged at time :math:`T` from an ancestral population,
with constant sizes throughout.

.. code-block:: text

   Time 0 (present):  Pop A (n_A=5)    Pop B (n_B=5)
                          |                  |
                          | Moran(t_1)       | Moran(t_1)
                          |                  |
   Time T (split):   -----+------------------+-----
                                  |
                                  | Moran(t_2)
                                  |
   Time infinity:           Ancestral (n=10)

The computation proceeds:

.. code-block:: python

   def two_pop_divergence_sfs(n_A, n_B, T, N_A, N_B, N_anc):
       """Compute expected SFS for a simple two-population divergence model."""
       sfs = np.zeros((n_A + 1, n_B + 1))

       # Step 1: Initialize leaf tensors
       # (using identity for full SFS computation)

       # Step 2: Apply Moran transitions for the recent epoch
       t_A = 2.0 * T / N_A  # scaled time for pop A
       t_B = 2.0 * T / N_B  # scaled time for pop B

       # Accumulate SFS contributions from the recent epoch
       # (mutations falling on branches within this epoch)

       # Step 3: At the split, convolve the two population tensors
       # Ancestral population has n_A + n_B lineages

       # Step 4: Apply Moran transition for the ancestral epoch
       # (going back to infinity -- all lineages coalesce)

       # Step 5: Accumulate SFS contributions from the ancestral epoch

       return sfs

.. admonition:: Calculus Aside -- Convolution as polynomial multiplication

   The convolution operation for merging two populations is mathematically
   equivalent to multiplying two generating polynomials:

   .. math::

      G_1(x) = \sum_j \binom{n_1}{j} L^{(1)}_j x^j, \quad
      G_2(x) = \sum_k \binom{n_2}{k} L^{(2)}_k x^k

   The product :math:`G_1(x) \cdot G_2(x)` has coefficients that give the
   convolved tensor, after dividing by the ancestral binomial coefficients.
   ``momi2`` exploits this by using ``numpy.convolve`` for the polynomial
   multiplication, giving :math:`O(n \log n)` performance via FFT for large
   sample sizes.

Step 7: How momi2 Handles the Tensor Computation
===================================================

In the actual ``momi2`` implementation, the tensor computation is organized
around the ``LikelihoodTensor`` and ``LikelihoodTensorList`` classes:

.. code-block:: python

   # Simplified from momi/compute_sfs.py

   class LikelihoodTensor:
       """Stores a multi-dimensional likelihood tensor and accumulated SFS."""

       def __init__(self, liks, sfs=0.0):
           self.liks = liks  # multi-dim array, one axis per population
           self.sfs = sfs    # accumulated scalar SFS contribution

       def matmul_last_axis(self, matrix):
           """Apply a matrix to the last axis (Moran transition)."""
           self.liks = self.liks @ matrix

       def add_last_axis_sfs(self, truncated_sfs):
           """Accumulate SFS contribution from current epoch."""
           self.sfs = self.sfs + np.dot(self.liks[..., 0, :], truncated_sfs)

       def mul_trailing_binoms(self, divide=False):
           """Multiply (or divide) by binomial coefficients for convolution."""
           n = self.liks.shape[-1] - 1
           binoms = np.array([comb(n, j) for j in range(n + 1)])
           if divide:
               self.liks = self.liks / binoms
           else:
               self.liks = self.liks * binoms

       def convolve_trailing_axes(self, other):
           """Convolve two likelihood tensors (population merge)."""
           # implemented via optimized Cython in momi/convolution.pyx
           self.liks = convolve_sum_axes(self.liks, other.liks)

The key performance optimizations in the real implementation:

- **Cython convolution** (``convolution.pyx``): parallelized antidiagonal
  summation and convolution using OpenMP
- **Batched einsum** (``einsum2/``): custom batched matrix multiplication that
  processes all configurations simultaneously
- **Memoized eigensystems**: computed once per sample size, cached across all
  likelihood evaluations

Exercises
=========

.. admonition:: Exercise 1: Manual tensor computation

   For a two-population model with :math:`n_A = n_B = 3` and a simple divergence
   at time :math:`T`, manually trace the tensor operations. Start with indicator
   vectors at the leaves and apply each operation step by step.

.. admonition:: Exercise 2: Convolution verification

   Verify that ``convolve_populations`` correctly computes the merged likelihood
   for the case where both child populations have all derived alleles (i.e.,
   :math:`L_1 = [0, 0, \ldots, 1]` and :math:`L_2 = [0, 0, \ldots, 1]`).

.. admonition:: Exercise 3: Admixture effects

   For a pulse admixture event with fraction :math:`f = 0.5`, compute the
   admixture tensor for :math:`n = 4` and verify that the expected number of
   lineages moving to the source population is :math:`n \cdot f = 2`.

.. admonition:: Exercise 4: Scaling with populations

   Compare the computational complexity of computing the expected SFS for
   :math:`k` populations using (a) a :math:`k`-dimensional grid (dadi-style)
   and (b) the tensor-tree approach (momi2-style). For what values of :math:`k`
   and :math:`n` does the tensor approach become advantageous?

Solutions
=========

.. admonition:: Solution 1: Manual tensor computation

   We trace through a two-population divergence with :math:`n_A = n_B = 3`,
   computing the SFS for a single configuration (e.g., :math:`b_A = 1, b_B = 2`).

   .. code-block:: python

      import numpy as np

      n_A, n_B = 3, 3
      b_A, b_B = 1, 2

      # Step 1: Initialize indicator vectors at the leaves
      L_A = np.zeros(n_A + 1)
      L_A[b_A] = 1.0  # L_A = [0, 1, 0, 0]

      L_B = np.zeros(n_B + 1)
      L_B[b_B] = 1.0  # L_B = [0, 0, 1, 0]

      print(f"Leaf A: {L_A}")
      print(f"Leaf B: {L_B}")

      # Step 2: Apply Moran transitions for the recent epoch (time T)
      T, N_A, N_B, N_anc = 500, 1000, 1000, 2000
      t_A = 2.0 * T / N_A
      t_B = 2.0 * T / N_B

      P_A = moran_transition(t_A, n_A)
      P_B = moran_transition(t_B, n_B)

      L_A_after = L_A @ P_A  # transform through recent epoch
      L_B_after = L_B @ P_B
      print(f"After Moran (A): {L_A_after}")
      print(f"After Moran (B): {L_B_after}")

      # Step 3: Convolve at the merge point
      from scipy.special import comb

      L_anc = convolve_populations(L_A_after, L_B_after, n_A, n_B)
      n_anc = n_A + n_B
      print(f"After merge (n_anc={n_anc}): {L_anc}")

      # Step 4: Apply Moran transition for ancestral epoch (t -> infinity)
      t_anc = 100.0  # large value approximating infinity
      P_anc = moran_transition(t_anc, n_anc)
      L_final = L_anc @ P_anc
      print(f"After ancestral epoch: {L_final}")

   The key insight is that each operation transforms the likelihood vector
   step by step: initialization sets the observed configuration, the Moran
   transition "smears" probabilities backward through drift, and the
   convolution combines lineages from both populations. At the end,
   :math:`L_{\text{final}}` gives the probability of the observed
   configuration under the model.

.. admonition:: Solution 2: Convolution verification

   When both child populations have all derived alleles (:math:`L_1 = [0, \ldots, 0, 1]`
   and :math:`L_2 = [0, \ldots, 0, 1]`), the ancestor must also have all derived
   alleles: :math:`L_{\text{anc}} = [0, \ldots, 0, 1]`.

   .. code-block:: python

      import numpy as np
      from scipy.special import comb

      for n1, n2 in [(3, 3), (5, 5), (3, 7), (10, 10)]:
          L1 = np.zeros(n1 + 1)
          L1[n1] = 1.0  # all derived

          L2 = np.zeros(n2 + 1)
          L2[n2] = 1.0  # all derived

          L_anc = convolve_populations(L1, L2, n1, n2)
          n_anc = n1 + n2

          # Expected: L_anc should be [0, 0, ..., 0, 1]
          expected = np.zeros(n_anc + 1)
          expected[n_anc] = 1.0

          assert np.allclose(L_anc, expected, atol=1e-10), \
              f"Failed for n1={n1}, n2={n2}: {L_anc}"
          print(f"n1={n1}, n2={n2}: L_anc = {L_anc} (correct)")

   This works because the convolution formula sums over all ways to partition
   :math:`i` derived alleles between the two children. When :math:`L_1` has mass
   only at :math:`n_1` and :math:`L_2` has mass only at :math:`n_2`, the only
   nonzero term in the sum is :math:`j = n_1, k = n_2`, giving
   :math:`i = n_1 + n_2 = n_{\text{anc}}`.

   Similarly, one can verify the case where both populations have zero derived
   alleles:

   .. code-block:: python

      L1 = np.zeros(n1 + 1)
      L1[0] = 1.0
      L2 = np.zeros(n2 + 1)
      L2[0] = 1.0
      L_anc = convolve_populations(L1, L2, n1, n2)
      assert np.allclose(L_anc[0], 1.0) and np.allclose(L_anc[1:], 0.0)

.. admonition:: Solution 3: Admixture effects

   For :math:`f = 0.5` and :math:`n = 4`, each lineage independently moves to
   the source with probability 0.5.

   .. code-block:: python

      import numpy as np
      from scipy.special import comb as binom

      n = 4
      f = 0.5
      T = admixture_tensor(n, f)

      print(f"Admixture tensor shape: {T.shape}")
      print(f"T[i, j, k] = Pr(i stay, j move | k original)")
      print()

      # Verify: for each k, the tensor should be a valid probability distribution
      for k in range(n + 1):
          total = 0.0
          for j in range(k + 1):
              i = k - j
              total += T[i, j, k]
              if T[i, j, k] > 1e-10:
                  print(f"  k={k}: i={i}, j={j}, T={T[i, j, k]:.4f}")
          assert abs(total - 1.0) < 1e-10, f"k={k}: total = {total}"
          print(f"  k={k}: total = {total:.6f}")
          print()

      # Expected number of lineages moving to source, starting from k lineages
      for k in range(n + 1):
          E_j = sum(j * T[k - j, j, k] for j in range(k + 1))
          print(f"  k={k}: E[j] = {E_j:.4f}, expected = {k * f:.4f}")
          assert abs(E_j - k * f) < 1e-10

   For :math:`k = n = 4` lineages, the expected number moving to the source is
   :math:`n \cdot f = 4 \times 0.5 = 2`. The distribution of :math:`j` (number
   moving) is :math:`\text{Binomial}(k, f)`:

   .. math::

      E[j \mid k] = k \cdot f = 4 \times 0.5 = 2

      \text{Var}(j \mid k) = k \cdot f(1-f) = 4 \times 0.25 = 1

   So with :math:`f = 0.5`, we get maximum variance in the splitting -- any
   outcome from 0 to 4 lineages moving is possible, with the binomial
   :math:`\binom{4}{j} (0.5)^4` giving probabilities 1/16, 4/16, 6/16, 4/16,
   1/16.

.. admonition:: Solution 4: Scaling with populations

   **Grid-based (dadi-style):** The SFS is represented on a :math:`k`-dimensional
   grid of size :math:`M^k` where :math:`M \approx n` is the grid resolution per
   dimension. The PDE solver must update all grid points at each time step.

   .. math::

      \text{Cost}_{\text{grid}} = O(M^k \times T_{\text{steps}})

   **Tensor-tree (momi2-style):** The computation visits each event in the tree
   once. At each event, the cost depends on the number of *active* populations
   (typically 1--3 at any point in the tree).

   .. math::

      \text{Cost}_{\text{tensor}} = O\!\left(\sum_{\text{events}} \prod_{\text{active pops } p} (n_p + 1)\right)

   .. code-block:: python

      import numpy as np

      print("Comparison of computational cost:")
      print(f"{'k':>3} {'n':>4} {'Grid (n^k)':>15} {'Tensor (tree)':>15} {'Ratio':>10}")
      print("-" * 52)

      for k in [2, 3, 4, 5, 6]:
          for n in [10, 20, 50]:
              # Grid: n^k entries, each updated ~100 times
              grid_cost = (n + 1) ** k * 100

              # Tensor-tree: for a balanced binary tree of k leaves,
              # there are 2k-1 nodes. At each internal node, at most 2
              # populations are active. Merge cost ~ n^2, Moran cost ~ n^2.
              # After each merge, the lineage count is at most 2n, but
              # the quasi-inverse reduces it back to n.
              tree_nodes = 2 * k - 1
              tensor_cost = tree_nodes * (2 * n + 1) ** 2

              ratio = grid_cost / tensor_cost
              print(f"{k:>3} {n:>4} {grid_cost:>15,} {tensor_cost:>15,} {ratio:>10.1f}")

   For :math:`k = 2, n = 10`, the costs are similar (grid may even be faster
   due to simpler operations). But for :math:`k \geq 4`, the tensor approach
   becomes dramatically better: the grid cost grows as :math:`n^k` while the
   tensor cost grows as :math:`k \cdot n^2`. At :math:`k = 6, n = 50`, the
   grid requires :math:`\sim 10^{12}` operations versus :math:`\sim 10^{5}` for
   the tensor tree -- a factor of :math:`10^7`.

   The crossover point is approximately :math:`k = 3`: for two populations,
   grid methods are competitive; for three or more, the tensor approach is
   strongly preferred. This is why ``momi2`` was designed specifically for
   multi-population analyses.

Next: :ref:`momi2_inference`
