.. _smcpp_timepiece:

====================
Timepiece II: SMC++
====================

   *Link a distinguished pair to the frequency spectrum of a large sample.*

SMC++ combines a pairwise coalescent HMM with allele-frequency information from
additional, unlabeled haplotypes :cite:`smcpp`. Its hidden state is the
discretized TMRCA of a **distinguished pair**, just as in PSMC. The remaining
haplotypes do not alter that pair's transition process. Instead, they enrich the
emission at each site from a binary difference indicator to a
**conditioned sample-frequency spectrum** (CSFS) observation.

That distinction is the central mechanism of SMC++ and the organizing principle
of this timepiece:

.. code-block:: text

   unphased genotypes
          |
          v
   choose two distinguished haplotypes
          |
          +-----------------------------+
          | hidden state: their TMRCA   |
          | transitions: two-locus CSC  |
          +-----------------------------+
                         |
   remaining haplotypes  |  observed (a, b)
          |              v
          +-----> conditioned SFS emission
                         |
                         v
               forward-backward / EM
                         |
                         v
             regularized size history

Here :math:`a\in\{0,1,2\}` is the derived-allele count in the
distinguished pair and :math:`b` is the count in the undistinguished sample.
Unphased diploid data work because swapping the two alleles within a diploid
does not change :math:`a` or :math:`b`.

The mini implementation follows the original statistical objects but uses a
partition-state CSFS calculation whose cost grows exponentially with sample
size. Production SMC++ replaces that enumeration with exact matrix identities,
automatic differentiation, locus skipping, and regularized optimization.

.. admonition:: Primary ground truth

   The derivations in this timepiece follow Terhorst, Kamm, and Song
   :cite:`smcpp` and the corresponding ``popgenmethods/smcpp`` source. In
   particular, ``src/conditioned_sfs.cpp`` defines the emission transform and
   ``src/transition.cpp`` defines the continuous-time transition kernel.

.. admonition:: Prerequisites

   Read :ref:`PSMC <psmc_timepiece>`, :ref:`Coalescent Theory
   <coalescent_theory>`, :ref:`The SMC <smc>`, and :ref:`Frequency Spectra
   <the_frequency_spectrum>` first. SMC++ reuses the pairwise hidden state but
   adds a conditional frequency-spectrum calculation.

.. toctree::
   :maxdepth: 2

   overview
   distinguished_lineage
   ode_system
   continuous_hmm
   population_splits
   demo
