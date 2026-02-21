.. _tensor_machinery:

=========================
Tensor Machinery
=========================

   *The gear train of the watch: assembling the expected SFS from a product of tensors, one demographic event at a time.*

.. epigraph::

   "The expected SFS can be written as a tensor product over the events in the
   demographic history."

   -- Kamm, Terhorst, Song, and Durbin (2017)

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

For child populations with :math:`n_1` and :math:`n_2` lineages:

.. math::

   L^{\text{anc}}_{i} = \sum_{j+k=i} L^{(1)}_{j} \cdot L^{(2)}_{k}

In practice, ``momi2`` first multiplies each tensor by binomial coefficients,
convolves via polynomial multiplication, then divides out the binomial
coefficients:

.. math::

   L^{\text{anc}}_{i} = \frac{1}{\binom{n_1+n_2}{i}} \sum_{j+k=i} \binom{n_1}{j} L^{(1)}_j \cdot \binom{n_2}{k} L^{(2)}_k

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

Next: :ref:`momi2_inference`
