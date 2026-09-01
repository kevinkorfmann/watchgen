.. _lshmm_overview:

========
Overview
========

Suppose :math:`H=(h_1,\ldots,h_k)` is a panel of :math:`k` haplotypes and
:math:`s` is a query observed at :math:`m` sites. At site :math:`\ell`, the
latent state :math:`Z_\ell\in\{1,\ldots,k\}` names the panel haplotype from
which the model emits the query allele. Recombination controls how persistent
that label is along the chromosome, while an error-or-mutation parameter allows
the emitted allele to differ from the selected template.

This language must be interpreted carefully. The panel sequences are not
literal ancestors in general, and the decoded path is not a reconstructed
ancestral recombination graph. A recombination event in the model redraws a
template and may redraw the same one; conversely, a decoded state change is only
evidence favoured by this approximation. Emission mismatches can absorb
mutation, genotyping error, model misspecification, or ancestry absent from the
panel.

The original PAC model
======================

Li and Stephens did not define only a fixed-panel query HMM. For an ordering
:math:`h_1,\ldots,h_n` of a sample, their approximate likelihood is

.. math::

   \widehat{\pi}(h_1,\ldots,h_n)
   = \widehat{\pi}(h_1)\prod_{k=1}^{n-1}
     \widehat{\pi}(h_{k+1}\mid h_1,\ldots,h_k).

Each conditional factor is evaluated by the copying HMM. The product depends on
haplotype order, so the paper averages or otherwise combines results across
orders. Calling a single conditional HMM “the Li--Stephens likelihood” hides
this important qualification.

What the HMM returns
====================

The forward recursion returns the conditional data likelihood and filtered
state probabilities. Forward--backward returns marginal posterior copying
probabilities using all sites. Viterbi returns one highest-probability state
sequence. Those quantities support downstream imputation and phasing models,
but local ancestry requires population labels and additional assumptions; it
does not follow automatically from a vanilla copying path.

The next section states the transition and emission laws exactly. The
:ref:`haploid_algorithms` section then shows why their special structure reduces
each HMM pass from :math:`O(mk^2)` to :math:`O(mk)`.
