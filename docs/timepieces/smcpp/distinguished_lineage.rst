.. _smcpp_distinguished:

===========================
The Distinguished Lineage
===========================

   *Single out one lineage, and let the rest tell you about the population.*

The Setup
==========

Consider :math:`n` haploid lineages sampled at the present. In PSMC, :math:`n = 2` and
the single coalescence time :math:`T` between them is the hidden variable. In SMC++,
we generalize this by choosing one lineage -- the **distinguished lineage** -- and
tracking its coalescence time :math:`T` with the rest of the sample.

The remaining :math:`n - 1` lineages are the **undistinguished lineages**. They are not
individually tracked; instead, we care only about how many of them are still present
(have not yet coalesced with each other) at any given time :math:`t` as we look back
from the present.

.. admonition:: Why "distinguished"?

   The terminology comes from the original SMC++ paper (Terhorst, Kamm & Song, 2017).
   The distinguished lineage is the one whose coalescence time we treat as the hidden
   state in the HMM. It is "distinguished" only in the mathematical sense -- any lineage
   can serve this role, and in practice SMC++ cycles through different choices to form a
   composite likelihood.

   Think of it this way: in PSMC, both lineages are symmetric -- neither is special.
   In SMC++, we break this symmetry by designating one lineage as the "probe" whose
   coalescence time we track, while the others serve as "reference" lineages that
   provide context about the population.


The Lineage-Count Process
===========================

As we trace the undistinguished lineages backward in time, they coalesce with each
other. At time :math:`t = 0` (the present), there are :math:`n - 1` undistinguished
lineages. At some time in the past, there may be fewer, because some pairs have found
common ancestors.

Let :math:`J(t)` be the number of undistinguished lineages remaining at time :math:`t`.
This is a **pure-death process**: lineages can only disappear (by coalescing), never
appear. The process starts at :math:`J(0) = n - 1` and decreases over time.

The coalescence rate among :math:`j` undistinguished lineages in a population of
size :math:`N(t) = N_0 \lambda(t)` is (from :ref:`coalescent theory <coalescent_theory>`):

.. math::

   \text{Rate}(j \to j-1) = \frac{\binom{j}{2}}{\lambda(t)} = \frac{j(j-1)}{2\lambda(t)}

This is the rate at which any pair among the :math:`j` lineages finds a common ancestor.
In a small population (:math:`\lambda(t)` small), this rate is high -- coalescence
is rapid. In a large population, the rate is low.

The distinguished lineage can coalesce with any of the :math:`j` undistinguished
lineages at rate :math:`j / \lambda(t)`. This is the **hazard rate** for the hidden
variable :math:`T` -- the rate at which the distinguished lineage's coalescence time
"ticks" at time :math:`t`, given that :math:`j` undistinguished lineages are still
present.

.. code-block:: python

   import numpy as np

   def undistinguished_coalescence_rate(j, lam):
       """Rate at which j undistinguished lineages coalesce among themselves.

       Parameters
       ----------
       j : int
           Number of undistinguished lineages currently present.
       lam : float
           Relative population size lambda(t) at current time.

       Returns
       -------
       float
           Coalescence rate C(j,2) / lambda.
       """
       return j * (j - 1) / (2 * lam)

   def distinguished_coalescence_rate(j, lam):
       """Rate at which the distinguished lineage coalesces with an undistinguished one.

       Parameters
       ----------
       j : int
           Number of undistinguished lineages currently present.
       lam : float
           Relative population size lambda(t) at current time.

       Returns
       -------
       float
           Rate j / lambda.
       """
       return j / lam

   # Example: 9 undistinguished lineages (10 haplotypes total), constant pop
   lam = 1.0
   j = 9
   print(f"Undistinguished coalescence rate (j={j}): {undistinguished_coalescence_rate(j, lam):.1f}")
   print(f"Distinguished coalescence rate (j={j}):   {distinguished_coalescence_rate(j, lam):.1f}")
   print(f"Total rate out of state j={j}:            "
         f"{undistinguished_coalescence_rate(j, lam) + distinguished_coalescence_rate(j, lam):.1f}")

.. code-block:: text

   Undistinguished coalescence rate (j=9): 36.0
   Distinguished coalescence rate (j=9):   9.0
   Total rate out of state j=9:            45.0

The total rate :math:`\binom{j}{2}/\lambda + j/\lambda = \binom{j+1}{2}/\lambda` is
simply the rate at which *any* pair among all :math:`j + 1` lineages (including the
distinguished one) coalesces. This makes sense: we have :math:`j + 1` lineages total,
and the coalescence rate for :math:`j + 1` lineages is :math:`\binom{j+1}{2}/\lambda`.


Why Unphased Data Works
=========================

A crucial practical advantage of SMC++ over MSMC is that it does not require
**phased** data. Phasing is the process of determining which alleles at different
sites came from the same parental chromosome. This is a non-trivial statistical
problem, and phasing errors can bias genealogical inference.

MSMC needs phased haplotypes because it tracks specific pairs of lineages and needs
to know which lineage each allele belongs to. SMC++ avoids this requirement because
the undistinguished lineages are, by definition, interchangeable -- they are tracked
only through their count :math:`j`, not their individual identities.

When working with unphased diploid genotypes, the distinguished lineage is one of
the two alleles at each site in the focal individual. The emission probability sums
over both possible assignments (which allele is the distinguished one), weighted by
their probability. For a diploid genotype at a site:

- **Homozygous reference (0/0)**: Both alleles are ancestral. Regardless of which is
  the distinguished lineage, the observation is "ancestral." The emission probability
  is the same as in the phased case.

- **Heterozygous (0/1)**: One allele is ancestral, one derived. The distinguished
  lineage could be either one. The emission probability averages over both
  assignments.

- **Homozygous alternate (1/1)**: Both alleles are derived. Again, the observation
  is unambiguous regardless of the assignment.

.. code-block:: python

   def emission_unphased(genotype, t, theta):
       """Emission probability for an unphased diploid genotype.

       Parameters
       ----------
       genotype : int
           Number of derived alleles: 0, 1, or 2.
       t : float
           Coalescence time of the distinguished lineage.
       theta : float
           Scaled mutation rate per bin.

       Returns
       -------
       float
           P(genotype | T = t).
       """
       # Probability that the distinguished lineage carries the derived allele
       # is approximately proportional to coalescence time (under infinite-sites
       # model, each lineage mutates with rate theta/2 per unit time).
       p_derived = 1 - np.exp(-theta * t)

       if genotype == 0:
           return (1 - p_derived) ** 2
       elif genotype == 1:
           return 2 * p_derived * (1 - p_derived)
       else:  # genotype == 2
           return p_derived ** 2


The Total Coalescence Rate
============================

The distinguished lineage's instantaneous coalescence rate at time :math:`t` depends
on two things:

1. How many undistinguished lineages :math:`j` are present at time :math:`t`.
2. The population size :math:`\lambda(t)`.

Since :math:`j` is random (it depends on the coalescence history of the
undistinguished lineages), the effective coalescence rate of the distinguished
lineage is an **average** over :math:`j`:

.. math::

   h(t) = \sum_{j=1}^{n-1} \frac{j}{\lambda(t)} \, p_j(t)

where :math:`p_j(t) = P(J(t) = j \mid J(0) = n - 1)` is the probability that
:math:`j` undistinguished lineages are still present at time :math:`t`.

When :math:`n = 2` (PSMC's case), :math:`J(t) = 1` for all :math:`t` (there is only
one undistinguished lineage, and it cannot coalesce with itself), so
:math:`h(t) = 1/\lambda(t)`. This is exactly PSMC's coalescence rate. SMC++ reduces
to PSMC when :math:`n = 2`.

For :math:`n > 2`, the function :math:`h(t)` is more complex. In the recent past
(small :math:`t`), many undistinguished lineages are present, so :math:`h(t)` is large
-- the distinguished lineage has many potential partners to coalesce with. In the
distant past (large :math:`t`), most undistinguished lineages have already coalesced
with each other, so :math:`h(t)` decreases toward :math:`1/\lambda(t)`.

This is the source of SMC++'s improved resolution in the recent past: the
coalescence rate :math:`h(t)` is high when :math:`t` is small, which means
coalescence events are concentrated in the recent past, providing dense signal
about recent :math:`N(t)`.

.. code-block:: python

   def compute_h(t, p_j, lam):
       """Compute the effective coalescence rate h(t) of the distinguished lineage.

       Parameters
       ----------
       t : float
           Time (used only for lambda evaluation).
       p_j : ndarray of shape (n,)
           p_j[j] = P(J(t) = j), for j = 0, 1, ..., n-1.
       lam : float
           Relative population size lambda(t).

       Returns
       -------
       float
           The effective coalescence rate h(t).
       """
       n_minus_1 = len(p_j) - 1
       h = 0.0
       for j in range(1, n_minus_1 + 1):
           h += j / lam * p_j[j]
       return h

Computing :math:`p_j(t)` is the subject of the next chapter: the ODE system that
tracks the lineage-count process through time.


Summary
========

The distinguished lineage framework gives SMC++ three advantages over a brute-force
multi-lineage HMM:

1. **Constant state space**: The hidden state is still just the coalescence time of
   one lineage, discretized into :math:`K` intervals -- the same as PSMC. Adding more
   samples does not increase :math:`K`.

2. **Unphased data**: Because undistinguished lineages are interchangeable, no phasing
   is needed. This avoids a major source of error.

3. **Scalability**: The composite likelihood decomposes into independent HMM
   computations, one per distinguished lineage. These can be parallelized trivially.

The cost is that SMC++ does not reconstruct the full genealogy -- it infers only the
population size function :math:`\lambda(t)`. For genealogical inference, you need
SINGER or tsinfer. But for demographic inference, SMC++ provides the best resolution
per unit of computational effort.
