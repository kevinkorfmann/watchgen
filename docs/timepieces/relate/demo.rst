.. _relate_demo:

=========================
Verified Relate exercises
=========================

The two figures in this Timepiece are generated from
``watchgen.mini_relate``. They validate internal mathematical targets; they are
not labelled as Relate executable output and do not establish whole-program
parity.

.. figure:: /_static/figures/fig_demo_relate.png
   :width: 100%
   :align: center

   Independent checks of directional mutation counts, mutual-minimum tree
   construction, fixed-order interval sampling, and event/exposure rate estimates.

Running the mini
================

.. code-block:: console

   python -m watchgen.mini_relate

The demo prints a topology-only Newick tree, posterior mean coalescent intervals
from the restricted sampler, its acceptance fraction, and the pairwise
coalescence-rate example. A seeded run is reproducible, but MCMC equality to a
single stored vector is not a scientific parity criterion.

Production executable validation
================================

The audited upstream commit was configured and compiled with CMake. Its bundled
C++ test executable passed all 239 assertions in 9 test cases. The supplied
``example/`` data were then run through ``scripts/RelateParallel/RelateParallel.sh``
with the repository's documented settings: :math:`N_e=30{,}000`, mutation rate
``1.25e-8``, sample ages, one thread, and seed 1.

The run produced 8,947 local trees for 11 haplotypes and 130,862 mutation rows.
Every ``.anc`` tree contained the required :math:`2N-1=21` node records, all
reported branch lengths were nonnegative, and tree start indices increased from
0 through 130,858. Mutation positions increased from 745 through 249,215,937 and
all tree indices lay in ``[0, 8947)``. The production diagnostics reported 287
non-mapping SNPs and 26 flipped SNPs, which also matched the parsed ``.mut`` file.

These checks include:

* every local tree has :math:`2N-1` nodes for :math:`N` haplotypes;
* parent times exceed child times and paths are ultrametric within numerical
  tolerance;
* mutation positions are ordered and mapped branch identifiers exist;
* adjacent-tree coordinates cover the requested region without reversal; and
* rerunning with the same supported seed and build gives the expected level of
  reproducibility.

Those checks validate file and tree invariants. Distributional or accuracy parity
requires simulations with known genealogies and cannot be inferred from successful
execution alone.

Automated chapter checks
========================

The focused tests independently verify all four emission cases, dense
forward--backward limiting cases, row rescaling, directional mutation counts,
mutual-minimum and fallback merges, weighted cluster distances, exact mapping and
flipping, compatible event ranks, ultrametric branch construction, the
choose-two coalescent density, Poisson likelihood, seeded MCMC behavior, and the
event/exposure estimator. Documentation gates require the audited source commit
``b54ede259cbb0be095bc9c9a8bd18cdaf7e88b74`` and reject the removed false-parity
APIs and claims.
