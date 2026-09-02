.. _phlash_composite_likelihood:

========================
The Composite Objective
========================

For diploid genomes :math:`g_1,\ldots,g_n`, PHLASH approximates their joint
sequence likelihood by a product of marginal PSMC likelihoods,

.. math::

   P(g_1,\ldots,g_n\mid\eta) \approx
   \prod_{i=1}^{n}P_{\mathrm{PSMC}}(g_i\mid\eta).

The factors are dependent because the genomes come from the same population.
Calling their product a *composite likelihood* records that dependence; it must
not be interpreted as an exact factorization.  This distinction is especially
important when interpreting posterior spread.

The released AFS term
=====================

Let :math:`a` be observed AFS counts and let :math:`e_\eta` be expected total
branch lengths under the history.  The source normalizes the expectation,
optionally folds and bins both vectors with a linear transform :math:`T`, then
computes

.. math::

   \ell_{\mathrm{AFS}}(\eta)
   = \sum_k (Ta)_k\log\left[T\left(
       \frac{e_\eta}{\sum_j e_{\eta,j}}
   \right)\right]_k.

This is the parameter-dependent part of a multinomial/Poisson-random-field-style
score.  The previous chapter implementation instead used independent Poisson
means without normalizing the expected spectrum; that was not source parity.

In ``phlash.model.log_density`` the implemented target is

.. math::

   c_0\log p(\vartheta)
   + c_1\sum_{i\in B}\ell_{\mathrm{PSMC},i}(\vartheta)
   + c_2\ell_{\mathrm{AFS}}(\vartheta).

For a minibatch of :math:`S` chunks from :math:`N`, ``fit`` uses
:math:`c=(1,N/S,1)`, so the sequence score is scaled to estimate the whole-data
sum.  The AFS term is not replicated per diploid HMM.

Prior and parameterization
==========================

The supplement describes independent standard-normal priors on log rates and
log-normal endpoint priors.  The released source additionally supports quadratic
regularizers: ``alpha`` penalizes adjacent differences in log rates and ``beta``
penalizes the flattened transformed parameters.  Both default to zero.  It also
constrains :math:`\rho/\theta` to :math:`[0.1,10]` by a logistic transform.
