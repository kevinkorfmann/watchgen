.. _gamma_smc_timepiece:

====================================
Timepiece XIII: Gamma-SMC
====================================

   *Ultrafast pairwise TMRCA inference -- the same problem as PSMC, without discretizing time.*

The Mechanism at a Glance
==========================

Gamma-SMC (Schweiger & Durbin, 2023) infers the **pairwise time to the most
recent common ancestor (TMRCA)** at every position along the genome, just like
PSMC (:ref:`Timepiece I <psmc_timepiece>`). It reads the same input -- a
sequence of heterozygous and homozygous sites from a pair of haplotypes -- and
asks the same question: when did these two lineages last share an ancestor?

But Gamma-SMC answers that question in a fundamentally different way. Where PSMC
discretizes coalescence time into a finite set of intervals and runs a standard
discrete-state HMM, Gamma-SMC keeps time **continuous** and tracks the posterior
distribution of the TMRCA as a **gamma distribution** at every position. The
result is a **continuous-state HMM (CS-HMM)** whose forward and backward passes
run in :math:`O(N)` time with no matrix multiplications, no time discretization
artifacts, and no EM iteration -- just a single pass through the data.

.. math::

   \text{Input: } \text{One pair of haplotypes} \quad \rightarrow \quad \text{a sequence of het/hom/missing observations}

.. math::

   \text{Output: } \text{Gamma}(\alpha_i, \beta_i) \quad \text{(posterior TMRCA distribution at each position } i\text{)}

The key insight is a happy conjugacy: if the prior on the TMRCA is a gamma
distribution and the emission model is Poisson, then the posterior after
observing a heterozygous or homozygous site is **again a gamma distribution**.
The transition step (accounting for recombination) breaks this conjugacy -- but
empirically, the post-transition distribution is *very close* to gamma. By
approximating it as gamma (via a precomputed flow field), the entire forward
pass reduces to a sequence of table lookups and parameter updates.

If PSMC is a mechanical watch with discrete gear teeth, Gamma-SMC is a
smooth-running quartz movement: no teeth to count, no intervals to choose.
The gamma distribution glides continuously through the space of possible
TMRCAs, updating its shape and rate parameters at each genomic position.

.. admonition:: Primary Reference

   :cite:`gamma_smc`

The four gears of Gamma-SMC:

1. **The Gamma Approximation** (the escapement) -- Poisson-gamma conjugacy makes
   the emission step exact and :math:`O(1)`: observing a heterozygous site
   transforms :math:`\text{Gamma}(\alpha, \beta)` into
   :math:`\text{Gamma}(\alpha + 1, \beta + \theta)`. The transition step is
   approximated by projecting back onto the gamma family. This is the
   fundamental tick of the mechanism -- the mathematical insight that makes
   everything else possible.

2. **The Flow Field** (the gear train) -- A precomputed two-dimensional vector
   field that maps gamma parameters :math:`(\alpha, \beta)` through one SMC
   transition step. The flow field is evaluated over a grid of mean and
   coefficient of variation values, and it is **independent of all model
   parameters** (:math:`\theta`, :math:`\rho`, :math:`N_e`). It is computed
   once, before any analysis, and reused for every dataset.

3. **The Forward-Backward CS-HMM** (the mainspring) -- The forward pass sweeps
   along the genome, updating the gamma parameters at each position via the
   flow field (transition) and conjugate update (emission). The backward pass
   is simply the forward algorithm run on the **reversed sequence**. The
   forward and backward gamma approximations are combined via a simple formula:
   :math:`\text{Gamma}(\alpha + \alpha' - 1, \beta + \beta' - 1)`. This
   combination yields the full posterior at each site.

4. **Segmentation and Caching** (the regulator) -- Long stretches of
   homozygous or missing sites are handled by precomputed **cached lookups**
   that skip many positions at once. An **entropy clipping** mechanism prevents
   the gamma approximation from drifting into invalid parameter regions. These
   are the practical engineering tricks that make Gamma-SMC ultrafast.

These gears mesh together into a single-pass inference machine:

.. code-block:: text

   VCF + BED masks (het/hom/missing along genome)
                     |
                     v
           +-----------------------+
           |  PRECOMPUTE FLOW      |
           |  FIELD                |
           |                       |
           |  Grid of (mu, CV)     |
           |  SMC transition step  |
           |  -> new (mu', CV')    |
           |  [parameter-free]     |
           +-----------------------+
                     |
                     v
           +-----------------------+
           |  SEGMENT & CACHE      |
           |                       |
           |  Group consecutive    |
           |  hom/missing sites    |
           |  Precompute multi-    |
           |  step flow lookups    |
           +-----------------------+
                     |
                     v
           +-----------------------+
           |  FORWARD PASS         |
           |                       |
           |  For each segment:    |
           |    cache lookup (miss)|
           |    cache lookup (hom) |
           |    emission update    |
           |  Record (alpha, beta) |
           |  at output positions  |
           +-----------------------+
                     |
                     v
           +-----------------------+
           |  BACKWARD PASS        |
           |                       |
           |  Forward on reversed  |
           |  sequence + one flow  |
           |  field step           |
           +-----------------------+
                     |
                     v
           +-----------------------+
           |  COMBINE              |
           |                       |
           |  Gamma(a+a'-1, b+b'-1)|
           |  at each output site  |
           +-----------------------+
                     |
                     v
           Posterior TMRCA distribution
           at each output position

PSMC vs. Gamma-SMC
====================

Gamma-SMC solves the same problem as PSMC, but the internal mechanism is
different in almost every respect:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Aspect
     - PSMC
     - Gamma-SMC
   * - **Hidden state**
     - Discrete time intervals :math:`[t_k, t_{k+1})`
     - Continuous TMRCA :math:`t \in (0, \infty)`
   * - **Posterior representation**
     - Probability vector over :math:`n` intervals
     - :math:`\text{Gamma}(\alpha, \beta)` -- two parameters
   * - **Transition step**
     - Matrix-vector multiply :math:`O(n^2)`
     - Flow field lookup + interpolation :math:`O(1)`
   * - **Emission step**
     - Elementwise multiply :math:`O(n)`
     - Conjugate update :math:`O(1)`
   * - **Time discretization**
     - Required (introduces artifacts)
     - None (continuous throughout)
   * - **Parameter estimation**
     - EM over many iterations
     - :math:`\theta` estimated from data; flow field is parameter-free
   * - **Output**
     - :math:`\hat{N}(t)` (demographic history)
     - :math:`\text{Gamma}(\alpha_i, \beta_i)` (per-site TMRCA posterior)
   * - **Demographic model**
     - Piecewise constant :math:`N(t)` (estimated)
     - Constant :math:`N_e` (assumed)

The constant-:math:`N_e` assumption is Gamma-SMC's main limitation compared to
PSMC: it does not estimate a demographic history. Instead, it provides
**per-site TMRCA posteriors** under a constant-population model, which can then
be used as input to downstream analyses (e.g., detecting selection, estimating
local ancestry, or feeding into demographic inference pipelines).

.. admonition:: Prerequisites for this Timepiece

   Before starting Gamma-SMC, you should have worked through:

   - :ref:`Coalescent Theory <coalescent_theory>` -- the exponential distribution
     of coalescence times and coalescent time units
   - :ref:`Hidden Markov Models <hmms>` -- the forward-backward algorithm
   - :ref:`The SMC <smc>` -- the SMC/SMC' transition density for constant
     population size

   The :ref:`PSMC Timepiece <psmc_timepiece>` is helpful but not strictly
   required. Gamma-SMC solves the same problem with a different mechanism, so
   familiarity with PSMC will provide useful context but is not a prerequisite
   for the derivations that follow.

Chapters
========

.. toctree::
   :maxdepth: 2

   overview
   gamma_approximation
   flow_field
   forward_backward
   segmentation_and_caching
   demo
