.. _lshmm_timepiece:

====================================
Timepiece III: The Li & Stephens HMM
====================================

The Li and Stephens model is a tractable approximation to the coalescent with
recombination. It represents one haplotype as an imperfect mosaic of a panel
of other haplotypes and evaluates that conditional model with a hidden Markov
model (HMM). The hidden copying path is a statistical device: a change of state
is not proof of a particular historical recombination, and a mismatch is not
proof of a mutation.

The original paper introduced a product of approximate conditionals (PAC) for
a sample of haplotypes and used it to estimate recombination rates and identify
hotspots :cite:`lshmm`. Later imputation, phasing, and genealogy methods reuse
the copying-model structure for different inferential tasks.

.. admonition:: Sources and scope

   The equations here are checked against Li and Stephens (2003), especially
   Appendix A. Executable haploid results are checked against the modern
   `lshmm reference implementation <https://github.com/astheeggeggs/lshmm>`_
   (release 0.0.8). That package is a later research implementation, not the
   historical Hotspotter program described by the paper. The diploid section
   derives a commonly used product-of-two-copying-processes extension; it is
   not part of the 2003 derivation.

.. toctree::
   :maxdepth: 2

   overview
   copying_model
   haploid_algorithms
   diploid
   demo
