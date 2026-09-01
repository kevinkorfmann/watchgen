.. _relate_population_size:

========================================
Coalescence rates and population history
========================================

Relate estimates piecewise-constant pairwise coalescence rates from dated local
trees. Fix a pair of haplotypes and let :math:`t_z` be their TMRCA in local tree
:math:`z`. Epoch :math:`e` spans :math:`[T_e,T_{e+1})` and has rate
:math:`\lambda_e`. Each observed TMRCA contributes one event in its ending epoch
and survival exposure in every epoch it traverses.

If :math:`n_e` pairwise TMRCAs fall in epoch :math:`e`, the maximum-likelihood
estimate from the Supplementary Note is

.. math::

   \widehat\lambda_e=
   \frac{n_e}
   {\displaystyle
      \sum_{z:e=e_z}(t_z-T_e)
      +\sum_{z:e<e_z}(T_{e+1}-T_e)}.

This is simply events divided by exposure. Relate calculates it for each pair and
then averages over pairs for a population-wide curve. The authors explicitly note
that this average is not the panmictic population-wide MLE. Its flexibility also
allows within- and between-population coalescence-rate summaries :cite:`relate`.

Units and effective size
========================

The inverse of a pairwise coalescence hazard has the units of a population-size
time scale. Whether it is labelled :math:`N_e` or :math:`2N_e` depends on whether
time and rate use haploid, diploid-generation, or coalescent units. A chapter must
state that convention before converting :math:`1/\lambda_e` to effective size.
The mini returns rates, event counts, and exposure directly and does not silently
apply a factor of two.

The iterative production procedure
==================================

Relate first dates trees with a constant-size prior. It then repeats the following
cycle:

1. estimate population-wide coalescence rates from current pairwise TMRCAs;
2. estimate mutation rate through time as mutations divided by branch exposure;
3. rescale coalescence rates by the ratio of a fixed mutation rate to that
   estimated temporal rate, a heuristic intended to accelerate convergence;
4. re-estimate branch lengths under the resulting variable-size coalescent prior.

The paper implementation uses five cycles, initially restricting the calculation
to trees with at least :math:`N` mapped mutations, and then performs a final dating
pass over all trees. This is an alternating plug-in procedure. The former chapter's
generic E-step/M-step equations for latent epoch allocations were not Relate's
published population-history method.

Teaching calculation
====================

For TMRCAs ``[0.2, 0.4, 1.2, 1.8]`` and boundaries
``[0, 0.5, 1, 2]``, the event counts are ``[2, 0, 2]`` and exposures are
``[1.6, 1.0, 1.0]``. The resulting rates are ``[1.25, 0, 2]``. The zero in the
middle is the unsmoothed MLE for an epoch with exposure but no event; production
analyses need sufficient genome-wide information and suitable epoch choices.

.. literalinclude:: ../../../watchgen/mini_relate.py
   :language: python
   :pyobject: piecewise_coalescence_rate_mle
