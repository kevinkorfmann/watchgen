.. _phlash_composite_likelihood:

================================
The Composite Likelihood
================================

   *The mainspring: two sources of energy, wound into a single spring.*

This chapter derives the two components of PHLASH's composite likelihood --
the site frequency spectrum (SFS) likelihood and the coalescent HMM
likelihood -- and explains how they are combined into a single objective
function for inference.

The composite likelihood is the mainspring of the PHLASH watch. In horology,
the mainspring stores the energy that powers the entire mechanism. In PHLASH,
the composite likelihood provides the statistical "energy" -- the information
from the data -- that drives the posterior sampling. Without it, the gears
have nothing to turn.


Two Views of the Same History
==============================

A demographic history :math:`\eta(t)` -- the population size as a function of
time -- leaves its fingerprints in the genome in two distinct ways:

1. **The site frequency spectrum** captures how allele frequencies are
   distributed across many individuals. A population bottleneck, for example,
   produces an excess of rare variants (because most variation arose after the
   recovery). The SFS is a summary statistic computed from the full sample:
   it counts, for each possible frequency :math:`k/n`, how many polymorphic
   sites have exactly :math:`k` copies of the derived allele in :math:`n`
   sampled chromosomes.

2. **The pairwise coalescent HMM** captures how the coalescence time varies
   along the genome for a single diploid individual (or a pair of
   haplotypes). This is exactly the PSMC model from
   :ref:`Timepiece I <psmc_timepiece>`: heterozygous sites mark regions of
   deep coalescence, homozygous stretches mark regions of recent coalescence,
   and the transitions between them reflect recombination.

These two views are **complementary**. The SFS aggregates information across
many individuals at each site, providing a snapshot of the frequency
distribution. The coalescent HMM traces correlations *along the genome* for a
single pair, using the linkage structure to infer how coalescence times change
from position to position. Neither view alone captures all the information in
the data; together, they constrain the demographic history more tightly than
either one alone.

.. admonition:: Biology Aside -- Why two data sources are better than one

   Consider trying to date a population bottleneck. The SFS tells you that
   there is an excess of rare variants (consistent with recent recovery from
   a small population), but it does not pinpoint *when* the bottleneck
   occurred -- many different bottleneck timings can produce similar SFS
   shapes. The coalescent HMM adds timing information: a bottleneck forces
   coalescence times into a narrow band, which creates long stretches of
   homozygosity bounded by heterozygous transitions at characteristic
   distances. The spacing of these transitions depends on when the bottleneck
   occurred and how severe it was. By combining both signals, PHLASH can
   simultaneously estimate the timing, severity, and duration of demographic
   events with far more precision than either signal alone.


Part 1: The SFS Likelihood
============================

The site frequency spectrum was introduced in detail in
:ref:`The Frequency Spectrum <the_frequency_spectrum>` (from the moments
Timepiece). Here we summarize the key result and show how PHLASH uses it.

Given :math:`n` sampled haploid chromosomes and a demographic history
:math:`\eta(t)`, the expected SFS is a vector
:math:`\boldsymbol{\xi} = (\xi_1, \ldots, \xi_{n-1})` where :math:`\xi_k` is
the expected number of segregating sites with :math:`k` derived alleles.

Each entry of the expected SFS can be written as an integral over coalescent
branch lengths:

.. math::

   \xi_k = L\mu \sum_{j=2}^{n} \binom{n - k - 1}{j - 2} \binom{k - 1}{0}
   \cdot \frac{1}{\binom{n-1}{j-1}} \cdot \mathbb{E}\left[T_j\right]

where :math:`T_j` is the total branch length when there are :math:`j`
lineages, which depends on the demographic history. For piecewise-constant
:math:`\eta(t)`, these expected branch lengths have closed-form expressions
involving the coalescence rates :math:`\binom{j}{2}/\eta_k` in each time
interval.

Given the observed SFS :math:`\mathbf{D} = (D_1, \ldots, D_{n-1})` (the
actual counts from the data) and the expected SFS
:math:`\boldsymbol{\xi}(\eta)` (computed under a candidate history
:math:`\eta`), the SFS log-likelihood under a Poisson model is:

.. math::

   \ell_{\text{SFS}}(\eta) = \sum_{k=1}^{n-1} \left[
   D_k \log \xi_k(\eta) - \xi_k(\eta)
   \right] + \text{const}

This is a standard Poisson log-likelihood: each SFS entry :math:`D_k` is
modeled as an independent Poisson random variable with mean :math:`\xi_k`.
The constant term (involving :math:`\log D_k!`) does not depend on
:math:`\eta` and can be dropped from optimization.

.. code-block:: python

   import numpy as np

   def sfs_log_likelihood(observed_sfs, expected_sfs):
       """Poisson log-likelihood of the observed SFS given expected SFS.

       Parameters
       ----------
       observed_sfs : ndarray, shape (n-1,)
           Observed SFS counts D_k for k = 1, ..., n-1.
       expected_sfs : ndarray, shape (n-1,)
           Expected SFS entries xi_k under the demographic model.

       Returns
       -------
       ll : float
           Poisson log-likelihood (up to a constant).
       """
       # Avoid log(0) for zero-expected entries
       xi = np.maximum(expected_sfs, 1e-300)
       return np.sum(observed_sfs * np.log(xi) - xi)

   def expected_sfs_constant(n, theta, N_e=1.0):
       """Expected SFS under constant population size.

       Under the standard coalescent with constant N_e, the expected
       number of segregating sites at frequency k/n is proportional
       to 1/k (Watterson's result).

       Parameters
       ----------
       n : int
           Number of haploid chromosomes.
       theta : float
           Population-scaled mutation rate (4 * N_e * mu * L).
       N_e : float
           Effective population size (default 1.0 in coalescent units).

       Returns
       -------
       xi : ndarray, shape (n-1,)
           Expected SFS.
       """
       k = np.arange(1, n)
       return theta / k

   # Demonstrate: constant-size SFS likelihood
   n = 20
   theta = 100.0  # realistic total theta for a genomic region
   xi_expected = expected_sfs_constant(n, theta)
   # Simulate observed SFS by Poisson sampling
   np.random.seed(42)
   D_observed = np.random.poisson(xi_expected)
   ll = sfs_log_likelihood(D_observed, xi_expected)
   print(f"Sample size n = {n}, theta = {theta}")
   print(f"Expected SFS (first 5): {xi_expected[:5].round(2)}")
   print(f"Observed SFS (first 5): {D_observed[:5]}")
   print(f"SFS log-likelihood: {ll:.2f}")

.. admonition:: Why Poisson?

   The Poisson model for the SFS arises naturally from the infinite-sites
   mutation model: mutations occur as a Poisson process on the branches of
   the genealogy, so the number of sites with any particular frequency is
   Poisson-distributed with a mean proportional to the expected branch
   length carrying that frequency. This is the same justification used by
   ``moments`` and ``momi2``.


Part 2: The Coalescent HMM Likelihood
=======================================

The coalescent HMM likelihood is inherited directly from PSMC. For each
diploid individual (or pair of haplotypes), PHLASH runs the same forward
algorithm described in :ref:`psmc_hmm`:

1. **Discretize time** into :math:`M` intervals :math:`[t_k, t_{k+1})`.

2. **Build the transition matrix** :math:`p_{kl}` encoding how the TMRCA
   changes between adjacent genomic positions under the SMC approximation
   (see :ref:`psmc_continuous` and :ref:`psmc_discretization`).

3. **Build emission probabilities** :math:`e_k(x)`: the probability of
   observing het/hom at a genomic bin given coalescence time in interval
   :math:`k`.

4. **Run the forward algorithm** to compute the log-likelihood
   :math:`\ell_{\text{HMM}}(\eta)` -- the probability of observing the
   entire heterozygosity sequence under the demographic model.

If the data contain :math:`P` diploid individuals (or pairs), the total
coalescent HMM log-likelihood is the sum over pairs:

.. math::

   \ell_{\text{HMM}}(\eta) = \sum_{p=1}^{P} \ell_{\text{HMM}}^{(p)}(\eta)

Each individual provides an independent view of the demographic history,
because the pairwise genealogy at each position depends on :math:`\eta(t)`
but is conditionally independent across unrelated individuals.

.. admonition:: Connection to PSMC

   If you set :math:`P = 1` (one diploid individual) and drop the SFS
   component, PHLASH's coalescent HMM likelihood reduces to exactly the
   PSMC likelihood. In this sense, PSMC is a special case of PHLASH -- a
   watch with only one of its two mainsprings wound.


Part 3: The Composite Likelihood
==================================

PHLASH combines the two likelihoods into a single composite log-likelihood:

.. math::

   \ell_{\text{comp}}(\eta) = \ell_{\text{SFS}}(\eta)
   + \ell_{\text{HMM}}(\eta)

On the log scale, combination means addition: the composite log-likelihood is
the sum of the two individual log-likelihoods.

.. admonition:: Why "composite" and not just "likelihood"?

   A true joint likelihood would require modeling the dependence between the
   SFS and the pairwise coalescent HMMs -- the SFS summarizes the marginal
   genealogy across all samples, while the coalescent HMM describes the full
   genealogy along the genome for specific pairs. These are not independent
   pieces of information; they share the same underlying genealogical process.

   A **composite likelihood** intentionally ignores this dependence. It
   treats the two likelihood components as if they were independent, even
   though they are not. This is a well-studied technique in statistics:
   composite likelihoods produce consistent parameter estimates (they
   converge to the truth as data grow), but the standard errors from a
   composite likelihood are not the true standard errors.

   In PHLASH, this is not a problem because the uncertainty quantification
   comes from the posterior (via SVGD with a proper prior), not from the
   curvature of the composite likelihood. The composite likelihood serves
   as an efficient **scoring function** -- a way to compare candidate
   histories -- rather than a probabilistic model of the data-generating
   process.


Part 4: The Prior
==================

PHLASH places a **smoothness prior** on the log-transformed demographic
history. Let :math:`\boldsymbol{h} = \log \boldsymbol{\eta}` (the log
population sizes). The prior is a Gaussian:

.. math::

   p(\boldsymbol{h}) \propto \exp\left(
   -\frac{1}{2} \boldsymbol{h}^\top \Sigma^{-1} \boldsymbol{h}
   \right)

where :math:`\Sigma` is a covariance matrix that encodes smoothness: nearby
time intervals are correlated, so the population size cannot jump wildly
between adjacent epochs without incurring a penalty.

The log-posterior that PHLASH targets is therefore:

.. math::

   \log p(\boldsymbol{h} \mid \text{data}) \propto
   \ell_{\text{comp}}(\boldsymbol{h}) + \log p(\boldsymbol{h})

The gradient of this log-posterior with respect to :math:`\boldsymbol{h}` is
what the score function algorithm computes (for the likelihood part) and what
SVGD uses to update its particles (for the full posterior).

.. code-block:: python

   def composite_log_likelihood(observed_sfs, expected_sfs, hmm_log_likelihoods):
       """Compute the composite log-likelihood (SFS + coalescent HMM).

       Parameters
       ----------
       observed_sfs : ndarray
           Observed SFS counts.
       expected_sfs : ndarray
           Expected SFS under the candidate history.
       hmm_log_likelihoods : list of float
           Log-likelihood from each pairwise coalescent HMM.

       Returns
       -------
       ll : float
           Composite log-likelihood.
       """
       ll_sfs = sfs_log_likelihood(observed_sfs, expected_sfs)
       ll_hmm = sum(hmm_log_likelihoods)
       return ll_sfs + ll_hmm

   def smoothness_prior_logpdf(h, sigma=1.0):
       """Log-density of the Gaussian smoothness prior on log-eta.

       Penalizes large differences between adjacent time intervals.

       Parameters
       ----------
       h : ndarray, shape (M,)
           Log population sizes (h = log eta).
       sigma : float
           Smoothness scale.

       Returns
       -------
       lp : float
           Log prior density (up to a constant).
       """
       diffs = np.diff(h)
       return -0.5 * np.sum(diffs**2) / sigma**2

   # Demonstrate the composite log-posterior
   M = 32  # number of time intervals
   h_true = np.zeros(M)  # log(eta) = 0 means eta = 1 (constant size)
   h_true[10:20] = -1.0  # a bottleneck: eta = exp(-1) ~ 0.37

   lp_prior = smoothness_prior_logpdf(h_true, sigma=1.0)
   ll_sfs = sfs_log_likelihood(D_observed, xi_expected)
   # Placeholder HMM log-likelihoods for 5 pairs
   ll_hmms = [-500.0, -480.0, -510.0, -490.0, -505.0]
   ll_comp = composite_log_likelihood(D_observed, xi_expected, ll_hmms)
   log_posterior = ll_comp + lp_prior

   print(f"SFS log-likelihood:       {ll_sfs:.2f}")
   print(f"HMM log-likelihood (sum): {sum(ll_hmms):.2f}")
   print(f"Composite log-likelihood: {ll_comp:.2f}")
   print(f"Prior log-density:        {lp_prior:.2f}")
   print(f"Log-posterior:            {log_posterior:.2f}")


What Comes Next
================

The composite likelihood tells PHLASH *how well* a candidate history explains
the data, but computing it requires choosing a time discretization. The
:ref:`next chapter <phlash_random_discretization>` explains how PHLASH
randomizes the discretization to cancel the bias that arises from any fixed
grid. This is the tourbillon -- the mechanism that keeps the watch accurate
regardless of its orientation.
