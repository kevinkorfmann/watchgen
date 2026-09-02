.. _clues_emissions:

======================
Emission Probabilities
======================

Ancient observations
====================

At derived frequency :math:`x`, Hardy--Weinberg genotype probabilities are

.. math::

   P(AA,AD,DD\mid x)=((1-x)^2,\;2x(1-x),\;x^2).

For sequencing reads :math:`R`, CLUES2 multiplies these prior genotype
probabilities by the supplied genotype likelihoods :math:`P(R\mid g)` and sums
over :math:`g`.  A haploid sample analogously uses probabilities
:math:`(1-x,x)`.  Likelihoods—not normalized genotype posteriors—are the expected
input values.

Genealogy emissions
===================

Derived lineages occupy fraction :math:`x` of the population; ancestral
lineages occupy :math:`1-x`.  Conditional on a frequency history, their
within-class coalescence waiting times contribute exponential survival and
event terms.  The CLUES2 source kernel uses internal diploid :math:`N_e` after
the public haploid-size conversion.

The implementation deliberately drops **frequency-independent event-rate
constants** from each coalescence event.  For a derived-class event it retains
the :math:`-\log x` term and the frequency-dependent survival hazard.  The
result is therefore an emission score proportional to the time density, not a
normalized absolute density.  The omitted constants cancel in likelihood
ratios over selection for the same genealogy.

.. code-block:: python

   import numpy as np
   from watchgen.mini_clues import log_coalescent_density

   score = log_coalescent_density(
       np.array([0.25]), n_lineages=3,
       epoch_start=0, epoch_end=1,
       frequency=0.3, n_haploid=200,
   )
   assert np.isclose(score, 1.1789728043259362)

The value above is a direct fixture from ``hmm_utils.py`` at ``b20dc5d``.
Calling it an absolute log probability would be methodologically incorrect.

Allele-class assumptions
========================

Tree converters assign sampled lineages and coalescences to derived or
ancestral classes under an infinite-sites interpretation.  Polarization errors,
leaf flips needed to satisfy that assumption, and an incorrect focal position
change the emissions before the HMM begins; they are not repaired by increasing
the number of frequency bins.
