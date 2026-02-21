.. _coalescent_sfs:

=========================
The Coalescent SFS
=========================

   *The dial face of the watch: reading expected allele frequencies from the shape of the genealogy.*

.. epigraph::

   "The expected SFS is a linear function of the expected coalescence times."

   -- Polanski and Kimmel (2003)

Step 1: From Genealogies to the SFS
=====================================

The site frequency spectrum counts how many SNPs have a given number of derived
alleles in the sample. Under the infinitely-many-sites mutation model, a
mutation that falls on a branch of the genealogy appears in exactly those
samples that descend from that branch. So the SFS is determined by the
**branch lengths** of the genealogy:

.. math::

   E[\text{SFS}[b]] = \frac{\theta}{2} \sum_{j=2}^{n} W_{b,j} \; E[T_{jj}]

where:

- :math:`\theta = 4 N_e \mu` is the population-scaled mutation rate
- :math:`T_{jj}` is the total time during which there are exactly :math:`j`
  lineages, each descending from exactly :math:`j` of the :math:`n` sampled
  chromosomes
- :math:`W_{b,j}` are combinatorial coefficients (the **W-matrix** of Polanski
  and Kimmel) that convert expected coalescence times into expected SFS entries

The W-matrix is the bridge between the coalescent (which gives us
:math:`E[T_{jj}]`) and the SFS (which is what we observe). Each entry
:math:`W_{b,j}` answers: "If a mutation falls on a branch with :math:`j`
descendants, what is the probability that exactly :math:`b` of my :math:`n`
sampled chromosomes carry the derived allele?"

.. admonition:: Probability Aside -- Why the W-matrix works

   Consider a genealogy with :math:`n` samples. At a moment when there are
   :math:`j` ancestral lineages, a mutation on one of those lineages will be
   inherited by some subset of the :math:`n` samples. The W-matrix entries are
   derived from the hypergeometric distribution: given :math:`j` lineages
   partitioning the :math:`n` samples, what is the probability that a random
   lineage has exactly :math:`b` descendants? Polanski and Kimmel (2003) showed
   that these coefficients satisfy a three-term recurrence, which ``momi2``
   implements in optimized Cython code.

Step 2: The W-Matrix Recurrence
================================

The W-matrix has dimensions :math:`(n-1) \times (n-1)`, with rows indexed by
SFS entry :math:`b \in \{1, \ldots, n-1\}` and columns indexed by epoch
:math:`j \in \{2, \ldots, n\}`. The recurrence (Polanski and Kimmel 2003) is:

.. math::

   W_{b,1} &= \frac{6}{n+1}

   W_{b,2} &= \frac{30(n - 2b)}{(n+1)(n+2)}

   W_{b,j} &= \frac{(2j+1)(n - 2b)}{j(n + j + 1)} W_{b,j-1}
              - \frac{(j+1)(2j+3)(n-j)}{j(2j-1)(n+j+1)} W_{b,j-2}

.. code-block:: python

   import numpy as np

   def w_matrix(n):
       """Compute the W-matrix of Polanski and Kimmel (2003).

       Returns W of shape (n-1, n-1), where W[b-1, j-2] gives the
       coefficient for SFS entry b from expected time with j lineages.
       """
       W = np.zeros((n - 1, n - 1))
       bb = np.arange(1, n)  # SFS entries 1..n-1

       W[:, 0] = 6.0 / (n + 1)
       if n > 2:
           W[:, 1] = 30.0 * (n - 2 * bb) / ((n + 1) * (n + 2))

       for col in range(2, n - 1):
           j = col + 2  # number of lineages
           W[:, col] = (
               W[:, col - 1] * (2 * j + 1) * (n - 2 * bb) / (j * (n + j + 1))
               - W[:, col - 2] * (j + 1) * (2 * j + 3) * (n - j)
                 / (j * (2 * j - 1) * (n + j + 1))
           )
       return W

.. code-block:: python

   # Verification: for a standard neutral model (constant size),
   # E[T_{jj}] = 2 / (j*(j-1)), and SFS[b] should equal theta/b.
   n = 20
   W = w_matrix(n)
   j_vals = np.arange(2, n + 1)
   E_Tjj_neutral = 2.0 / (j_vals * (j_vals - 1))
   expected_sfs = W @ E_Tjj_neutral  # should be proportional to 1/b
   bb = np.arange(1, n)
   ratio = expected_sfs / (1.0 / bb)
   assert np.allclose(ratio, ratio[0], atol=1e-10), "W-matrix verification failed"

Step 3: Expected Coalescence Times Under Constant Size
========================================================

For a population of constant size :math:`N`, the coalescent gives a simple
expression for the expected time spent with exactly :math:`j` lineages. The
rate of coalescence when there are :math:`j` lineages is
:math:`\binom{j}{2} = j(j-1)/2`, so:

.. math::

   E[T_{jj}] = \frac{1}{\binom{j}{2}} = \frac{2}{j(j-1)}

(in units of :math:`2N` generations). Plugging this into the W-matrix formula
recovers the classic neutral SFS: :math:`E[\text{SFS}[b]] \propto 1/b`.

But what if population size changes through time? This is where ``momi2``'s
machinery becomes essential.

.. code-block:: python

   def etjj_constant(n, tau, N):
       """Expected time with j lineages in an epoch of duration tau and size N.

       tau: epoch duration in generations
       N: population size (constant throughout epoch)

       Returns array of length n-1, indexed by j = 2, ..., n.
       """
       j = np.arange(2, n + 1)
       rate = j * (j - 1) / 2.0  # coalescence rate with j lineages
       scaled_time = 2.0 * tau / N  # time in coalescent units
       # expected time with j lineages, accounting for finite epoch duration
       return (1.0 - np.exp(-rate * scaled_time)) / rate

Step 4: How Demographic Events Modify Expected Branch Lengths
===============================================================

Each type of demographic event modifies the expected coalescence times in a
specific way:

**Population size change.** If the population has size :math:`N_1` in one epoch
and :math:`N_2` in the next, the coalescence rate changes proportionally. In
scaled coalescent time, the rate with :math:`j` lineages is always
:math:`\binom{j}{2}`, but the *real-time* duration of that epoch is
:math:`\tau = \int_0^T 2/N(s)\, ds` in coalescent units. A larger population
means slower coalescence (longer branches), and vice versa.

**Exponential growth.** When the population grows exponentially at rate :math:`g`,
:math:`N(s) = N_0 e^{-gs}` (backward in time). The scaled time becomes:

.. math::

   \tau = \frac{2}{N_0} \int_0^T e^{gs}\, ds = \frac{2(e^{gT} - 1)}{N_0 g}

The expected coalescence times involve exponential integrals (the :math:`\text{Ei}`
function), computed in closed form.

**Population split** (viewed backward: a merge). When two populations merge into
an ancestral population, the lineages from both child populations now share a
single common ancestor pool. The number of lineages in the ancestral population
is the sum of remaining lineages from each child. This is handled by
**convolution** of the likelihood tensors (see :ref:`tensor_machinery`).

**Admixture pulse.** A fraction :math:`f` of population :math:`A`'s ancestry
comes from population :math:`B` at some time in the past. Viewed backward, each
lineage in :math:`A` independently "jumps" to :math:`B` with probability
:math:`f`. This redistributes lineages between the two populations according to
a binomial distribution, implemented as a 3-tensor contraction.

.. code-block:: python

   def etjj_exponential(n, tau, growth_rate, N_bottom):
       """Expected time with j lineages under exponential growth.

       N_bottom: population size at the more recent end of the epoch
       growth_rate: exponential growth rate (positive = growing forward)
       tau: epoch duration in generations
       """
       from scipy.special import expi

       j = np.arange(2, n + 1)
       rate = j * (j - 1) / 2.0
       N_top = N_bottom * np.exp(-tau * growth_rate)

       if abs(growth_rate) < 1e-10:
           return etjj_constant(n, tau, N_bottom)

       # Scaled time for the epoch
       total_growth = tau * growth_rate
       scaled_time = (np.expm1(total_growth) / total_growth) * tau * 2.0 / N_bottom

       # Expected coalescence times via exponential integral
       a = rate * 2.0 / (N_bottom * growth_rate)
       result = np.zeros_like(rate)
       for idx in range(len(rate)):
           c = a[idx]
           result[idx] = (
               np.exp(-c) * (-expi(c) + expi(c * np.exp(total_growth)))
           )
       return result

Step 5: The Multi-Population SFS
=================================

For :math:`k` populations with sample sizes :math:`n_1, \ldots, n_k`, the SFS
becomes a :math:`k`-dimensional array of shape
:math:`(n_1 + 1) \times (n_2 + 1) \times \cdots \times (n_k + 1)`. Each entry
:math:`\text{SFS}[b_1, b_2, \ldots, b_k]` counts the number of SNPs where
population :math:`i` has :math:`b_i` derived alleles.

``momi2`` refers to each such multi-population index :math:`(b_1, \ldots, b_k)`
as a **configuration**. The expected value of each configuration is computed by
the tensor machinery described in :ref:`tensor_machinery`.

The multi-population extension is where ``momi2`` truly shines compared to
grid-based methods. Adding populations to ``dadi`` multiplies the grid
dimensions; adding populations to ``momi2`` adds axes to the tensors but the
computation proceeds along the event tree, visiting each event exactly once.

.. code-block:: python

   import numpy as np

   def compute_joint_sfs(genotype_matrix, pop_assignments, pop_names):
       """Compute the joint SFS from a genotype matrix.

       genotype_matrix: (n_sites, n_samples) array of 0/1
       pop_assignments: dict mapping sample index -> population name
       pop_names: list of population names (determines axis order)

       Returns: k-dimensional array of shape (n1+1, n2+1, ..., nk+1)
       """
       # group samples by population
       pop_indices = {p: [] for p in pop_names}
       for idx, pop in pop_assignments.items():
           pop_indices[pop].append(idx)

       sample_sizes = [len(pop_indices[p]) for p in pop_names]
       sfs_shape = tuple(s + 1 for s in sample_sizes)
       sfs = np.zeros(sfs_shape, dtype=int)

       for site in range(genotype_matrix.shape[0]):
           config = tuple(
               genotype_matrix[site, pop_indices[p]].sum()
               for p in pop_names
           )
           sfs[config] += 1

       return sfs

Step 6: From Branch Lengths to Summary Statistics
===================================================

A powerful feature of the coalescent formulation is that the expected SFS is
not the only quantity you can compute. Any **linear function** of the SFS can
be expressed as an inner product with the likelihood tensor. This includes:

- **f-statistics** (:math:`f_2`, :math:`f_3`, :math:`f_4`): linear combinations
  of SFS entries that measure shared drift between populations
- **Patterson's D statistic** (ABBA-BABA): a test for admixture
- **Nucleotide diversity** (:math:`\pi`): a weighted sum of SFS entries
- **Tajima's D**: a ratio of SFS-derived estimators

``momi2`` computes these by passing appropriate **weight vectors** through the
same tensor machinery used for the SFS. The same autograd gradients apply,
so you can optimize demographic models to fit any of these statistics.

.. admonition:: Probability Aside -- Why linear statistics are free

   If the expected SFS is :math:`E[\mathbf{S}]` and a statistic is
   :math:`f(\mathbf{S}) = \mathbf{w}^T \mathbf{S}` for some weight
   vector :math:`\mathbf{w}`, then
   :math:`E[f(\mathbf{S})] = \mathbf{w}^T E[\mathbf{S}]`.
   In ``momi2``'s tensor framework, the weight vector :math:`\mathbf{w}`
   is simply passed as the initial state of the likelihood tensor at the
   leaves, rather than the identity vector used for the full SFS. The
   rest of the computation proceeds identically.

Exercises
=========

.. admonition:: Exercise 1: Verify the neutral SFS

   Using the ``w_matrix`` and ``etjj_constant`` functions above, verify that
   the expected SFS under a constant population is proportional to :math:`1/b`
   for sample sizes :math:`n = 10, 50, 100`.

.. admonition:: Exercise 2: Population expansion and the SFS

   Compute the expected SFS for a population that expanded 10-fold at time
   :math:`T = 0.1 \times 2N` generations ago. Compare with the neutral
   expectation. Where does the expansion create excess entries?

.. admonition:: Exercise 3: Bottleneck signature

   Compute the expected SFS for a population that went through a bottleneck
   (size reduced to 0.1 for duration 0.05, then recovered). How does this
   differ from a simple size change?

.. admonition:: Exercise 4: Multi-population SFS dimensions

   For :math:`k` populations each of sample size :math:`n`, how many entries
   does the joint SFS have? For what values of :math:`k` and :math:`n` does
   this become impractical to store? How does ``momi2``'s approach avoid this
   problem?

Next: :ref:`moran_model`
