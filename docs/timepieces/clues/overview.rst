.. _clues_overview:

========
Overview
========

CLUES2 accepts one or more of three evidence sources: coalescence times from
sampled local trees, ancient diploid genotype likelihoods, and ancient haploid
likelihoods.  It conditions the present on the observed modern derived-allele
frequency.  All times are in generations.

The workflow is:

#. convert Relate or SINGER trees into derived- and ancestral-class
   coalescence times, or supply ancient likelihoods;
#. discretize frequency with Beta(1/2, 1/2) quantiles;
#. approximate one-generation backward transitions with a Gaussian;
#. combine transition probabilities with genealogy and ancient-data emissions;
#. use forward/backward recursions to integrate over frequency histories;
#. if multiple trees were sampled, form an importance estimate from their
   selected-versus-neutral likelihood ratios; and
#. optimize one or more epoch-specific selection coefficients.

What the likelihood means
=========================

Let :math:`X` be the frequency trajectory and :math:`G` a local genealogy.
For fixed :math:`G`, the HMM sums over discretized :math:`X`.  With posterior
genealogy samples, importance sampling approximates the remaining integral over
:math:`G`.  Both the Gaussian transition and finite frequency grid are
approximations; tree inference adds another approximation.  “Approximate full
likelihood” is therefore the accurate term.

Original CLUES versus CLUES2
============================

The 2019 method sampled ARGs with ARGweaver and derived a conditional
importance estimator.  CLUES2 retained the HMM/importance architecture while
adding ancient observations and more efficient sparse recursions.  Its supplied
converters support Relate and SINGER.  A SINGER tree sequence is not itself an
``inference.py`` input: ``SingerToCLUES.py`` first extracts locus-specific
coalescence times and allele classes.

Production analyses should cite both the 2019 method and the CLUES2 paper,
record the source revision, state allele polarization, and report the frequency
grid, population-size convention, dominance coefficient, time cutoff, and
selection epochs.
