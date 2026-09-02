.. _dating_threads:

=====================
Dating Path Segments
=====================

   *The mechanism tells you which gear meshes where. Now calibrate the clock.*

The third and final step in the Threads pipeline assigns **coalescence times**
to each of the segments inferred by the Viterbi algorithm. These dated segments
constitute the threading instructions needed to assemble the ARG.


The IBD Segment Model
=======================

Threads models each Viterbi segment as an **identical-by-descent (IBD)
region** shared between the target sample and its inferred closest cousin. An
IBD segment is a maximal genomic region where two sequences share a constant
most recent common ancestor.

Each IBD segment has:

- A **length** :math:`l_{\text{cM}}` in centimorgans and :math:`l_{\text{bp}}`
  in base pairs
- A **height** (age) :math:`t`, measured in generations, equal to the age of
  the most recent common ancestor
- A number of **heterozygous sites** :math:`m` (mutations distinguishing the
  two copies)

The key modeling assumptions:

- Mutations follow a Poisson process with rate :math:`\mu = 2 \cdot c \cdot l_{\text{bp}}`,
  where :math:`c` is the per-base mutation rate
- Under the SMC model, the segment length follows an exponential distribution
  with rate :math:`\rho = 2 \cdot 0.01 \cdot l_{\text{cM}}`
- Segments are independent of each other

.. note::

   Viterbi segments do not always coincide with true IBD segments. The Viterbi
   path may both overestimate and underestimate segment lengths. Simulations
   show that coincidence with true IBD remains stable at approximately 40%,
   while overestimation tends to 0 as :math:`N` increases. The net effect is a
   tendency to underestimate segment lengths and thus overestimate their ages.


Maximum Likelihood Estimators
===============================

We first derive estimators that ignore the coalescent prior, using only the
recombination and mutation likelihoods.

**From recombination only.** Conditional on age :math:`t`, IBD length in
Morgans follows an exponential distribution with rate :math:`2t`. With
:math:`\rho` defined as twice the observed genetic length, the likelihood is:

.. math::

   \ell(\rho \mid t) = 2t \, e^{-t\rho}

To find the MLE, take the log-likelihood and differentiate:

.. math::

   \log\ell = \log(2t) - t\rho = \log 2 + \log t - t\rho

.. math::

   \frac{d\log\ell}{dt} = \frac{1}{t} - \rho = 0 \quad\Longrightarrow\quad \hat{t} = \frac{1}{\rho}

Longer IBD segments imply more recent shared ancestry (:math:`\hat{t}` is small
when :math:`\rho` is large).

**From recombination and mutations.** Adding the Poisson mutation model with
:math:`m` heterozygous sites:

.. math::

   \ell(\rho, m \mid t, c) = 2t \, e^{-t\rho} \cdot \frac{(t\mu)^m}{m!} \, e^{-t\mu}

Taking the log-likelihood:

.. math::

   \log\ell = \log 2 + \log t - t\rho + m\log(t\mu) - t\mu - \log(m!)

Collecting the :math:`t`-dependent terms: :math:`(m+1)\log t - t(\rho + \mu) + \text{const}`.
Differentiating:

.. math::

   \frac{d\log\ell}{dt} = \frac{m+1}{t} - (\rho + \mu) = 0
   \quad\Longrightarrow\quad \hat{t} = \frac{m + 1}{\rho + \mu}

The numerator :math:`m + 1` counts the :math:`m` observed mutations plus the
one "count" from the recombination boundary. The denominator :math:`\rho + \mu`
is the total rate at which events (both mutations and recombination) accumulate
with time. This is the natural estimator: total event count divided by total rate.

.. code-block:: python

   import numpy as np

   def mle_age_recomb(rho):
       """MLE of segment age from recombination only.

       Parameters
       ----------
       rho : float
           Recombination measure: 2 * 0.01 * l_cM.

       Returns
       -------
       t_hat : float
           Maximum likelihood age estimate (in generations).
       """
       return 1.0 / rho

   def mle_age_full(rho, mu, m):
       """MLE of segment age from recombination and mutations.

       Parameters
       ----------
       rho : float
           Recombination measure.
       mu : float
           Mutation measure: 2 * c * l_bp.
       m : int
           Number of heterozygous sites in the segment.

       Returns
       -------
       t_hat : float
           Maximum likelihood age estimate.
       """
       return (m + 1) / (rho + mu)

   # Demonstrate MLE estimators
   l_cM = 1.0    # 1 centimorgan segment
   l_bp = 1e6    # ~1 Mb
   c = 1.25e-8   # per-base mutation rate
   rho = 2 * 0.01 * l_cM
   mu = 2 * c * l_bp

   print(f"Segment: {l_cM} cM, {l_bp/1e6:.0f} Mb")
   print(f"  rho = {rho:.4f}, mu = {mu:.5f}")
   t_recomb = mle_age_recomb(rho)
   print(f"  MLE (recomb only): {t_recomb:.0f} generations")
   for m in [0, 1, 3, 10]:
       t_full = mle_age_full(rho, mu, m)
       print(f"  MLE (recomb + {m} hets): {t_full:.0f} generations")


Bayesian Estimators
=====================

We now place an exponential prior :math:`\pi(t) \sim \text{Exp}(\gamma)` on the
segment age, where :math:`\gamma` is the coalescence rate. This prior models
the coalescence process: under a constant population of size :math:`N_e`, the
coalescence rate is :math:`\gamma = 1/N_e`.

**From recombination only.** The posterior is:

.. math::

   p(t \mid \rho) \propto \pi(t) \, p(\rho \mid t) = 2\gamma \, t \, e^{-t(\rho + \gamma)}

This is an Erlang-2 distribution with rate :math:`\rho + \gamma` and mean:

.. math::

   E[t \mid \rho] = \frac{2}{\rho + \gamma}

**From recombination and mutations.** Including the mutation likelihood:

.. math::

   p(t \mid \rho, m) \propto 2\gamma \, \frac{\mu^m}{m!} \, t^{m+1} \, e^{-t(\rho + \mu + \gamma)}

This is an Erlang-:math:`(m+2)` distribution with rate
:math:`\rho + \mu + \gamma` and mean:

.. math::

   E[t \mid \rho, m] = \frac{m + 2}{\rho + \mu + \gamma}

Note the difference from the maximum likelihood estimator: the numerator gains
an extra count (from the prior), and the denominator includes the coalescence
rate :math:`\gamma`.

.. code-block:: python

   def bayes_age_recomb(rho, gamma):
       """Bayesian posterior mean age from recombination only.

       Uses an Exp(gamma) prior on age.

       Parameters
       ----------
       rho : float
           Recombination measure.
       gamma : float
           Coalescence rate (1 / N_e).

       Returns
       -------
       t_hat : float
           Posterior mean age.
       """
       return 2.0 / (rho + gamma)

   def bayes_age_full(rho, mu, m, gamma):
       """Bayesian posterior mean age from recombination and mutations.

       Parameters
       ----------
       rho : float
           Recombination measure.
       mu : float
           Mutation measure.
       m : int
           Number of heterozygous sites.
       gamma : float
           Coalescence rate.

       Returns
       -------
       t_hat : float
           Posterior mean age.
       """
       return (m + 2) / (rho + mu + gamma)

   # Demonstrate Bayesian vs. MLE estimators
   N_e = 10000
   gamma = 1.0 / N_e
   print(f"\nBayesian estimators (N_e = {N_e}):")
   print(f"  gamma = {gamma:.6f}")
   t_bayes_r = bayes_age_recomb(rho, gamma)
   print(f"  Bayes (recomb only): {t_bayes_r:.0f} generations")
   for m in [0, 1, 3, 10]:
       t_mle = mle_age_full(rho, mu, m)
       t_bayes = bayes_age_full(rho, mu, m, gamma)
       print(f"  m={m:2d}: MLE = {t_mle:6.0f}, "
             f"Bayes = {t_bayes:6.0f} generations")
   print("(Prior pulls estimates toward the coalescent expectation)")


Piecewise-Constant Demographic Models
========================================

Real populations do not have constant effective size. Threads accommodates
**piecewise-constant demographic models** where the effective population size
:math:`N_e(t)` changes at discrete time boundaries.

Define time intervals :math:`0 = T_0 < T_1 < \cdots < T_K = \infty` with
constant effective sizes :math:`N_e^{(k)}` and coalescence rates
:math:`\gamma_k = 1/N_e^{(k)}` in each interval :math:`[T_k, T_{k+1})`. Write
:math:`\Delta_k = T_{k+1} - T_k`.

The prior becomes piecewise exponential:

.. math::

   \pi(t) = \gamma_k \exp\left(-\sum_{j=0}^{k-1} \Delta_j \gamma_j - (t - T_k)\gamma_k\right) \quad \text{for } t \in [T_k, T_{k+1})

**Recombination-only estimator.** The posterior expectation is:

.. math::

   E[t \mid \rho] = \frac{\sum_{k=0}^{K-1} \gamma_k \, e^{-\sum_{j=0}^{k-1}\Delta_j\gamma_j + T_k\gamma_k} \cdot \frac{2}{(\rho+\gamma_k)^3} \left[P(3, (\rho+\gamma_k)T_{k+1}) - P(3, (\rho+\gamma_k)T_k)\right]}{\sum_{k=0}^{K-1} \gamma_k \, e^{-\sum_{j=0}^{k-1}\Delta_j\gamma_j + T_k\gamma_k} \cdot \frac{1}{(\rho+\gamma_k)^2} \left[P(2, (\rho+\gamma_k)T_{k+1}) - P(2, (\rho+\gamma_k)T_k)\right]}

where :math:`P(a, z) = \gamma(a, z) / \Gamma(a)` is the regularized lower
incomplete gamma function.

This estimator is used for inference from sparse data such as genotyping
arrays, where mutation information is unreliable.

**Full estimator with mutations.** Adding :math:`m` heterozygous sites and
writing :math:`\lambda_k = \rho + \mu + \gamma_k`:

.. math::

   E[t \mid \rho, m] = \frac{\sum_{k=0}^{K-1} \gamma_k \, e^{-\sum_{j=0}^{k-1}\Delta_j\gamma_j + T_k\gamma_k} \cdot \frac{\mu^m \cdot (m+2)}{\lambda_k^{m+3}} \left[P(m+3, \lambda_k T_{k+1}) - P(m+3, \lambda_k T_k)\right]}{\sum_{k=0}^{K-1} \gamma_k \, e^{-\sum_{j=0}^{k-1}\Delta_j\gamma_j + T_k\gamma_k} \cdot \frac{\mu^m}{\lambda_k^{m+2}} \left[P(m+2, \lambda_k T_{k+1}) - P(m+2, \lambda_k T_k)\right]}

This is the estimator used by Threads for coalescence time inference in
whole-genome sequencing data. The piecewise-constant demographic model is
specified as a ``.demo`` file listing the time boundaries and effective
population sizes.

.. note:: Production decision rule

   Current ``ThreadsFastLS::date_segment`` uses the recombination-only
   piecewise formula for sparse-array mode. For sequence mode it evaluates the
   full piecewise formula through :math:`m=15`, then uses a constant-rate
   shortcut for larger mutation counts. The teaching function
   ``watchgen.mini_threads.threads_date_segment`` mirrors that dispatch;
   the standalone MLEs above are derivations, not production defaults.

.. code-block:: python

   from watchgen.mini_threads import bayesian_full, threads_date_segment

   l_cM = 1.0
   l_bp = 1_000_000
   c = 1.25e-8
   rho = 2 * 0.01 * l_cM
   mu = 2 * c * l_bp

   # Demonstrate with a bottleneck demography
   # Recent: N_e=10000 (0-1000 gen), Bottleneck: N_e=1000 (1000-2000),
   # Ancient: N_e=20000 (2000+)
   T_bounds = [0, 1000, 2000]  # final epoch extends to infinity
   N_e_values = [10000, 1000, 20000]

   print("\nPiecewise-constant demography:")
   for k in range(len(N_e_values)):
       upper = T_bounds[k + 1] if k + 1 < len(T_bounds) else float("inf")
       print(f"  [{T_bounds[k]:.0f}, {upper}): N_e = {N_e_values[k]}")

   print(f"\nSegment: {l_cM} cM, {m} hets")
   for m in [0, 1, 5]:
       t_const = bayesian_full(m, rho, mu, 1.0/10000)
       t_pw = threads_date_segment(
           m, l_cM, l_bp, c, T_bounds, N_e_values, sparse=False)
       print(f"  m={m}: constant N_e -> {t_const:.0f}, "
             f"piecewise -> {t_pw:.0f} generations")
