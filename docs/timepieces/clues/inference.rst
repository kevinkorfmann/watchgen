.. _clues_inference:

===================================
HMM, Importance Sampling, and Tests
===================================

Conditioning on modern frequency
================================

The recursion starts with all mass at the grid point nearest the observed
modern derived-allele frequency.  It then moves backward one generation at a
time, multiplying transition probabilities by emissions from coalescences and
ancient samples.  Log-space normalization prevents underflow.

For a generic transition matrix :math:`P` and log emission :math:`e_t(k)`,

.. math::

   \log \alpha_t(k)=e_t(k)+
   \operatorname{logsumexp}_j[\log\alpha_{t-1}(j)+\log P_{jk}].

The teaching implementation tests this recursion against enumeration of every
state path in a small HMM.  CLUES2 adds time-varying population sizes,
selection epochs, changing lineage counts, sparse bands, and approximation
caches.

Importance sampling over trees
==============================

For genealogy sample :math:`G_m`, let :math:`\ell_s(G_m)` be its conditional
log likelihood at selection coefficient :math:`s`.  CLUES2 first computes the
neutral value :math:`\ell_0(G_m)` as the importance denominator.  Its estimator
is

.. math::

   \log \widehat R(s)=
   \operatorname{logsumexp}_{m=1}^M
      [\ell_s(G_m)-\ell_0(G_m)]-\log M.

In words: take ``ell_s(G_m) - ell_0(G_m)`` for every aligned tree sample, then a
**log-mean-exp**.  Averaging log likelihoods, reversing the ratio, or applying
unmatched weights gives a different estimator.

Optimization and likelihood-ratio output
========================================

For one epoch, current ``inference.py`` uses bracketed Brent searches, including
a retry for small samples and a wider fallback.  Multiple epochs use
Nelder--Mead.  A bounded scalar optimizer in the teaching module illustrates
the objective but is not presented as optimizer parity.

The output column named ``logLR`` is
:math:`\Delta\ell=\ell(\hat s)-\ell(0)`.  For the chi-square tail calculation,
CLUES2 uses the conventional statistic

.. math::

   \Lambda=2\Delta\ell

with degrees of freedom equal to the number of fitted selection epochs.  The
CLUES2 paper validates this calibration by simulation for its stated settings;
that does not guarantee calibration under every boundary, misspecification, or
data-dependent epoch choice.

Trajectory reconstruction
=========================

Forward and backward messages yield a posterior over frequency bins at each
generation conditional on fixed selection parameters and a genealogy.  CLUES2
then integrates over fitted-selection uncertainty with Monte Carlo and, when
applicable, over importance-sampled trees.  Increasing ``--integration_points``
reduces Monte Carlo variance but does not remove grid, transition, genealogy, or
model approximation error.
