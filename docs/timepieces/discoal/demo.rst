.. _discoal_demo:

==========================
Verified discoal exercises
==========================

The demo now separates production-software validation from the pedagogical
single-locus calculation. The previous figure said it used data simulated with
msprime although its script used only the mini, compared a TMRCA quantity with
diversity, and overlaid an unsupported rational curve as a classical diversity law.
Those labels and the unsupported curve have been removed.

.. figure:: /_static/figures/fig_demo_discoal.png
   :width: 100%
   :align: center

   Independent validation targets: the exact deterministic C curve, the pairwise
   TMRCA shift at the selected site, and convergence as the trajectory grid is
   refined.

Production command
==================

After compiling paper-era upstream discoal at commit ``82971bf``, the executable
validation used this reproducible hard-sweep command:

.. code-block:: console

   ./discoal 20 50 1000 -t 10 -r 20 -ws 0.05 -a 1000 -x 0.5 -d 23456 78901

Here 20 is the number of haploid samples, 50 the number of replicates, and 1000
the number of discrete sites. ``-t 10`` and ``-r 20`` are whole-locus values of
:math:`4N\mu_L` and :math:`4Nr_L`. ``-ws 0.05`` places a stochastic sweep endpoint
0.05 units of :math:`4N` generations ago, ``-a 1000`` sets :math:`2Ns`, and
``-x 0.5`` places the selected site
at the center. The ``-d`` values are discoal's two random seeds.

All 50 outputs passed the structural checks below, and their mean pairwise
diversity in the central fifth of the locus was 0.117 mutations per pair, versus
0.213 and 0.278 in the two flanking fifths. This small run is a smoke test rather
than a benchmark. Validation should inspect more than successful execution. Every replicate must
contain 20 equal-length haplotypes, positions must be ordered within the unit
interval, and the reported number of segregating sites must equal each haplotype
length. Across many replicates, neutral simulations should recover the expected
coalescent summaries, while linked sweep simulations should show a reduction in
pairwise diversity near the selected site. These are distributional checks, not a
claim that one random seed has a prescribed answer.

As an independent neutral check, 1000 replicates with ``-t 10`` and no
recombination or sweep gave a mean of 35.508 segregating sites, close to the
standard-coalescent expectation :math:`10H_{19}=35.477`. Exact random output and
performance are version-specific, which is why the commit is part of the command's
provenance.

Source-guided mini
==================

The module demo can be run with

.. code-block:: console

   python -m watchgen.mini_discoal

It constructs ``mini_discoal.deterministic_trajectory`` with a small sweep
effective size, checks ``mini_discoal.escape_probability`` on that explicit path,
and estimates ``mini_discoal.pairwise_diversity_profile``. The profile uses two
samples, so mean TMRCA divided by :math:`2N` is a valid relative pairwise-diversity
proxy. Each position is nevertheless simulated independently; this is not a
chromosome-scale ARG simulator.

Automated checks
================

The dedicated tests verify the deterministic C formula at fixed times, the
conditional jump endpoints and neutral standing phase, exact rate scaling,
rejection of a coarse event grid, conservation of lineages, the :math:`2N`
neutral pairwise-TMRCA expectation, and haploid sample parity in the msprime
converter. The documentation tests also reject the former whole-program parity
claims and require the audited upstream commit
``7d0955f4107053c135d2086790b0426457147a8e``, the paper-era executable commit
``82971bf``, and primary references
:cite:`discoal,braverman_1995,coop_griffiths_2004,msprime2`.

The implementation used by these exercises is kept in one place:

.. literalinclude:: ../../../watchgen/mini_discoal.py
   :language: python
   :lines: 1-24
