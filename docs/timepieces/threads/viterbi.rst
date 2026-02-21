.. _viterbi_threads:

=========================================
Memory-Efficient Viterbi Inference
=========================================

   *The classical machine works, but it does not fit in the case. Redesign it.*

The second step in the Threads pipeline takes the candidate matches from the
PBWT pre-filter and runs a **memory-efficient Viterbi algorithm** under the
Li-Stephens model to find the optimal threading path for each sample.


The Classical Viterbi Limitation
==================================

Given a reference panel :math:`H` of :math:`N` haplotypes over :math:`M` sites
and a query haplotype :math:`g`, the Li-Stephens model assigns a probability
:math:`P(\pi)` to each path :math:`\pi \in \{1, \ldots, N\}^M` through the
panel. A **Viterbi path** is a path of maximum probability.

The classical Viterbi algorithm finds this path in :math:`O(NM)` time by
constructing a full :math:`N \times M` probability matrix, then performing a
traceback. For biobank-scale data, this matrix is computationally prohibitive
-- even after PBWT pre-filtering reduces :math:`N` to :math:`L`, the memory
requirement for long genomic tracts remains high.


Two Key Observations
======================

The Threads-Viterbi algorithm exploits two properties of the Li-Stephens model:

**Observation 1: Recombination events are rare.** Viterbi paths consist of few
but long segments. In the 1000 Genomes Project (2,251 sequences), the average
segment length exceeds 200 kilobases. In UK Biobank array data
(:math:`N = 337{,}464`), segments average well over a megabase. This means a
complete Viterbi path can be stored compactly by recording only the segment
breakpoints and threading targets.

**Observation 2: Recombination is symmetric.** Under the Li-Stephens model:

.. math::

   p(\pi_{i+1} = \beta \mid \pi_i = \alpha, \text{recombination between } i \text{ and } i+1) = \frac{1}{N}

for any states :math:`\alpha, \beta`. This symmetry dramatically reduces the
search space for possible Viterbi paths, as formalized in Proposition 1.


Proposition 1
===============

**Proposition 1.** *Suppose* :math:`\pi^{(i)}` *is a Viterbi path for the
subset of the panel* :math:`H` *containing sites 1 through* :math:`i`. *If
there exists a Viterbi path through* :math:`H` *that recombines between sites*
:math:`i` *and* :math:`i + 1`, *then there exists a Viterbi path* :math:`\pi`
*through* :math:`H` *satisfying* :math:`\pi_i = \pi^{(i)}_i`.

In plain language: at site :math:`i`, we only need to consider recombination
from the sequence of highest probability given all observations up to
:math:`i`. This is the property that makes the branch-and-bound strategy
correct.


The Segment Set
=================

The algorithm maintains a set :math:`\Omega` of **path segments**, each
consisting of:

- A start site :math:`m_\omega \in \{1, \ldots, M\}`
- A threading target :math:`n_\omega \in \{1, \ldots, N\}`
- If :math:`m_\omega > 0`, a traceback segment :math:`\omega' \in \Omega` with
  :math:`m_{\omega'} < m_\omega`

A full path through :math:`H` is constructed by starting at any segment and
following traceback pointers until a segment with :math:`m_\omega = 0` is
reached. The penalty (negative log-likelihood) of a path ending at
:math:`\omega` is denoted :math:`s(\omega)`.

The set :math:`\Omega` contains exactly :math:`N` **active segments**
:math:`\omega_1, \ldots, \omega_N`, one per reference haplotype. The segment
set is **complete** if each :math:`P(\omega_n)` is the Li-Stephens-optimal path
ending at haplotype :math:`n`. When the set is complete, the active segment
with minimum penalty gives a Viterbi path.


The Branch Step (Theorem 1)
=============================

Given a complete segment set :math:`\Omega^{(m)}` for sites :math:`1` through
:math:`m`, the branch step constructs :math:`\Omega^{(m+1)}` that is complete
for sites :math:`1` through :math:`m + 1`.

Let :math:`\omega'` be the active segment with minimum penalty (the current
Viterbi path), and let :math:`\rho` and :math:`\rho_c` be the penalties for
recombination and no-recombination respectively.

For each active segment :math:`\omega_n`, if continuing without recombination
is worse than recombining from the best path:

.. math::

   s(\omega_n, m) + \rho_c > s(\omega', m) + \rho

then a **new segment** :math:`\omega(m+1, n, \omega')` is created, representing
a recombination from the best path to haplotype :math:`n` at site :math:`m+1`.

The new active segment for haplotype :math:`n` becomes whichever option has
lower penalty:

.. math::

   s_{\text{new}} = \min\{s(\omega_n, m) + \rho_c,\; s(\omega', m) + \rho\} + \mu_n

where :math:`\mu_n` is the match/mismatch penalty at site :math:`m + 1`.

**Complexity per site:** The branch step adds at most :math:`N` new segments,
giving :math:`O(NM)` total segments across all sites. In practice, new
segments are created only at inferred recombination events, which are rare.


The Bound Step (Theorem 2)
============================

The bound step prunes the segment set without losing completeness.

**Theorem 2.** *Let* :math:`\Omega` *be a complete segment set with active
segments* :math:`\omega_1, \ldots, \omega_N`. *Define*
:math:`\Omega^* \subseteq \Omega` *as the union of all traceback paths from
the active segments:*
:math:`\Omega^* = \bigcup_{n=1}^{N} P(\omega_n)`. *Then* :math:`\Omega^*`
*is also a complete segment set.*

The bound step simply discards any segment that is not on a traceback path from
an active segment -- these segments are **undercut** and can never be part of
an optimal path.

Threads applies the bound step at regular intervals using a heuristic threshold:

- Initialize with :math:`B_0 = 10 \cdot N`
- If :math:`|\Omega^{(m)}| > B_0`, prune to :math:`\Omega^*`
- If the next pruning trigger occurs within 30 sites, double the threshold to
  :math:`B_1 = 2B_0`; otherwise reset to :math:`B_0`

This balances pruning frequency against the risk of memory spikes from rapid
segment accumulation.


Traceback
===========

After processing all :math:`M` sites, the final Viterbi path is recovered by:

1. Identifying the active segment :math:`\omega^*` with minimum penalty
2. Following traceback pointers from :math:`\omega^*` until reaching a segment
   with start site 0

The result is a piecewise-constant path through the reference panel: a sequence
of segments, each specifying a threading target and a genomic interval.


Parallelism
=============

A critical property of the Threads-Viterbi algorithm: the :math:`N` Viterbi
instances (one per sample) are **completely independent**. The output of each
HMM does not depend on any other. This means all :math:`N` instances can run
in parallel, divided evenly among available CPU cores.

Given :math:`L` haplotype matches per sample and :math:`N_{\text{CPU}}` cores:

- **Time:** :math:`O(MLN / N_{\text{CPU}})`
- **Memory:** :math:`O(LN)` average

The genotype data is streamed from disk once per core, and neither the full
genotype matrix nor any :math:`N \times M` probability matrix is ever stored in
memory.


Complexity Summary
====================

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Property
     - Classical Viterbi
     - Threads-Viterbi
   * - Time (per sample)
     - :math:`O(NM)`
     - :math:`O(NM)` (same)
   * - Memory (per sample)
     - :math:`O(NM)`
     - :math:`O(N)` average
   * - Total (all samples)
     - :math:`O(MN^2)` time + memory
     - :math:`O(MLN/N_{\text{CPU}})` time, :math:`O(LN)` memory
   * - Parallelism
     - Not straightforward
     - Embarrassingly parallel
