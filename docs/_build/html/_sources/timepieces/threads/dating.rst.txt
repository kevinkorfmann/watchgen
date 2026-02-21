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

**From recombination only.** Under the SMC model, the length of an IBD segment
follows an exponential distribution with rate :math:`1/(2t)`. The likelihood is:

.. math::

   \ell(\rho \mid t) = 2t \, e^{-t\rho}

This has a maximum at :math:`\hat{t} = 1/\rho`.

**From recombination and mutations.** Adding the Poisson mutation model with
:math:`m` heterozygous sites:

.. math::

   \ell(\rho, m \mid t, c) = 2t \, e^{-t\rho} \cdot \frac{(t\mu)^m}{m!} \, e^{-t\mu}

Differentiating with respect to :math:`t`, the maximum likelihood estimator is:

.. math::

   \hat{t} = \frac{m + 1}{\rho + \mu}


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
