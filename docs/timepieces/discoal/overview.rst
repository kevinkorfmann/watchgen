.. _discoal_overview:

=====================
Overview of discoal
=====================

``discoal`` generates samples under coalescent models with recombination,
demographic changes, population structure, and selection :cite:`discoal`. A sweep
is not simulated by assigning a shorter neutral tree after the fact. The simulator
conditions ancestry on the frequency path of the selected allele and tracks which
ancestral segments reside on the beneficial or wild-type background.

.. figure:: /_static/figures/fig_mini_discoal.png
   :width: 100%
   :align: center

   Source-guided checks for the teaching kernel. The panels show the deterministic
   production curve, stochastic conditional trajectories, the four single-locus
   event probabilities, and an independently simulated pairwise-diversity proxy.
   The last panel uses independent marginal loci and is not an ARG simulation.

The production algorithm
========================

At a high level, a sweep simulation contains a neutral phase from the present back
to the sweep endpoint, a sweep phase, and an older neutral phase. During the sweep
phase, the program advances on a fine time grid. It updates the allele frequency,
calculates coalescence, crossover, gene-conversion, and recurrent-adaptive-mutation
probabilities, and performs a possible event. For an in-locus crossover, an active
ancestor is split at a discrete breakpoint. One child segment retains the selected
background and the other receives a background according to the current allele
frequency. This ancestry bookkeeping is why a two-counter single-locus model is
not a translation of the complete program.

The simulator accepts a haploid sample size, a number of replicates, and a number
of discrete sites. Its main scaled parameters are

.. math::

   \theta = 4N_0\mu_L, \qquad
   \rho = 4N_0r_L, \qquad
   \alpha = 2N_0s,

where :math:`\mu_L` and :math:`r_L` are whole-locus mutation and crossover rates.
The ``-t`` and ``-r`` options therefore do not take per-base rates. The selected
site is given as a relative coordinate with ``-x``. Sweep and demographic times
entered at the command line are in units of :math:`4N_0` generations; the C event
loop converts them to its internal :math:`2N_0` clock.

The output is normally an ``ms``-compatible haplotype matrix. With ``-T``, discoal
instead prints marginal Newick trees and their spans. The program can also include
the selected SNP in partial-sweep output; ``-h`` suppresses that site.

Selection models and limits
===========================

The flags ``-wd``, ``-ws``, and ``-wn`` request deterministic selection,
stochastic selection, and neutral fixation, respectively. The ``-f`` option adds
a neutral phase below the frequency at which selection began, ``-uA`` allows
recurrent mutation toward the adaptive class, and ``-c`` stops a partial sweep at
an intermediate frequency. Recurrent hitchhiking is available with ``-R`` for
sweeps in the locus and ``-L`` for sweeps to its left. These options are described
in the original manual and program paper :cite:`discoal`.

The program also implements crossing over, gene conversion, instantaneous size
changes, multiple populations, migration, population merges, admixture, and ancient
samples. Selection is restricted to population 0, and migration into or out of
that population is disabled during its sweep phase because discoal does not follow
selected-allele trajectories in several populations simultaneously.

What is approximate
===================

The trajectory-plus-structured-coalescent construction is an approximation to a
selected Wright--Fisher population. It assumes that conditioning on the selected
allele's path captures the relevant effect of selection on linked ancestry. The
trajectory is discretized, and the event loop treats each small interval as having
at most one event. Finer steps reduce discretization error but increase runtime.
The C code uses a default step scalar of 40 and a default sweep effective size of
one million; together they give an internal step of
:math:`1/(40N_{\mathrm{sweep}})` in :math:`2N` time units.

The structured coalescent is especially useful for strong, recent sweeps, but it
does not model individual genomes or selected-locus interference. Forward
simulators such as SLiM are needed when diploid fitness, multiple linked selected
alleles, or explicit pedigrees are central.

The teaching boundary
=====================

The rest of this Timepiece derives the single-locus rates and reproduces the
trajectory law. ``mini_discoal`` operates at a fixed recombination distance from
the selected site, so recombination appears as switching backgrounds. In the full
program, recombination inside the simulated region splits ancestral material. The
distinction is methodological, not merely an implementation detail.
