.. _mcmc:

===========================
Markov Chain Monte Carlo
===========================

   *"When you can't find the answer, let randomness find it for you -- one carefully guided step at a time."*

A master watchmaker, confronted with a broken mechanism of unknown design, cannot
simply enumerate every possible gear arrangement to find the one that matches the
symptoms. The space of possibilities is too vast. Instead, the watchmaker makes an
educated guess, tries a small modification, checks whether the watch runs better, and
repeats. The analogy is useful for exploration, but MCMC targets a distribution;
it does not generally converge on one "true" mechanism.

Markov Chain Monte Carlo (MCMC) is the mathematical formalization of this strategy.
In population genetics, we face exactly the same challenge: given observed sequence
data :math:`D`, we want to infer the genealogical history :math:`G` -- the ancestral
recombination graph, the coalescence times, the population sizes. The posterior
distribution :math:`P(G \mid D)` lives in an astronomically large space, and direct
computation is impossible. MCMC constructs a random walk through this space, visiting
states in proportion to their posterior probability, and thereby producing samples
from the distribution we cannot compute directly.

This chapter develops MCMC from the ground up, building toward the specific forms
used by ARGweaver (see :ref:`argweaver_mcmc`), SINGER (see :ref:`sgpr`), and PHLASH
(see :ref:`phlash_svgd`). If you have not yet read the chapter on :ref:`hmms`, the
section on Markov chains here will provide the necessary foundation. If you have,
the connection between HMMs and MCMC will become clear: both exploit the Markov
property, but in very different ways.


The Big Idea: Why Sample?
===========================

In Bayesian inference, the goal is to compute the **posterior distribution**:

.. math::

   P(G \mid D) = \frac{P(D \mid G) \, P(G)}{P(D)}

where :math:`G` represents the unknown quantity (a genealogy, a demographic model,
a set of parameters) and :math:`D` is the observed data (genotype sequences,
allele frequencies, variant calls).

The posterior tells us everything we want to know: which genealogies are consistent
with the data, how certain we are about each one, and what the range of plausible
parameter values is. But computing it requires evaluating the normalizing constant
:math:`P(D) = \int P(D \mid G) P(G) \, dG`, which sums (or integrates) over every
possible value of :math:`G`. For an ARG with thousands of branches, millions of
genomic positions, and continuous coalescence times, this integral is intractable.

MCMC sidesteps this problem entirely. Instead of computing the posterior, it
**samples** from it. The algorithm constructs a Markov chain -- a sequence of
states :math:`G^{(0)}, G^{(1)}, G^{(2)}, \ldots` -- that converges to the posterior
distribution. After warmup, states are still generally **correlated**, but ergodic
averages can estimate posterior expectations. We can use the samples to estimate
any quantity of interest: posterior means, credible intervals, marginal distributions.

Like a blind watchmaker exploring the space of possible mechanisms -- unable to see
the full blueprint, but able to feel whether each small adjustment improves the
mechanism or not -- MCMC navigates the posterior landscape one step at a time,
gradually concentrating its exploration on the regions that matter most.


Bayesian Inference in 60 Seconds
==================================

Before building the MCMC machinery, let us establish the Bayesian framework that
motivates it.

.. admonition:: Probability Aside -- Bayes' theorem

   **Bayes' theorem** relates the posterior probability of a hypothesis :math:`H`
   given data :math:`D` to the likelihood and prior:

   .. math::

      P(H \mid D) = \frac{P(D \mid H) \, P(H)}{P(D)}

   The four components are:

   - :math:`P(H)` -- the **prior**: our belief about :math:`H` before seeing data.
   - :math:`P(D \mid H)` -- the **likelihood**: how probable the data is if :math:`H`
     is true.
   - :math:`P(H \mid D)` -- the **posterior**: our updated belief after seeing data.
   - :math:`P(D)` -- the **evidence** (or marginal likelihood): the total probability
     of the data under all hypotheses. This is the normalizing constant that makes the
     posterior sum to 1.

   The critical insight: :math:`P(D)` does not depend on :math:`H`. It is the same
   for every hypothesis. This means we can write:

   .. math::

      P(H \mid D) \propto P(D \mid H) \, P(H)

   The posterior is *proportional to* the likelihood times the prior. MCMC exploits
   this: it only needs the unnormalized posterior, never the evidence.

Let us make this concrete with an example where the posterior is known exactly, so
we can verify our MCMC results later.

**The Beta-Binomial model.** Suppose we observe :math:`k` successes in :math:`n`
trials (say, :math:`k = 7` derived alleles out of :math:`n = 20` sites), and we
want to infer the success probability :math:`\theta`.

- **Prior**: :math:`\theta \sim \text{Beta}(\alpha, \beta)` with density
  :math:`p(\theta) \propto \theta^{\alpha - 1}(1 - \theta)^{\beta - 1}`.
- **Likelihood**: :math:`P(k \mid \theta, n) = \binom{n}{k} \theta^k (1 - \theta)^{n-k}`.
- **Posterior**: By Bayes' theorem, the posterior is also a Beta distribution:
  :math:`\theta \mid k \sim \text{Beta}(\alpha + k, \beta + n - k)`.

This is a **conjugate** model: the prior and posterior belong to the same family.
This is the exception, not the rule -- for most real problems (including ARG
inference), no conjugate form exists, and we must resort to MCMC.

.. code-block:: python

   import numpy as np

   def beta_binomial_demo():
       """Demonstrate exact Bayesian inference with a conjugate model.

       We use this as a ground truth to verify MCMC results later.

       Returns
       -------
       alpha_post : float
           Posterior alpha parameter.
       beta_post : float
           Posterior beta parameter.
       """
       # Observed data: 7 derived alleles out of 20 sites
       n, k = 20, 7

       # Prior: Beta(2, 2) -- a gentle prior favoring values near 0.5
       alpha_prior, beta_prior = 2, 2

       # Posterior: Beta(alpha + k, beta + n - k)
       alpha_post = alpha_prior + k       # 2 + 7 = 9
       beta_post = beta_prior + (n - k)   # 2 + 13 = 15

       # The posterior mean is alpha / (alpha + beta)
       post_mean = alpha_post / (alpha_post + beta_post)
       post_var = (alpha_post * beta_post) / (
           (alpha_post + beta_post)**2 * (alpha_post + beta_post + 1)
       )

       print(f"Prior: Beta({alpha_prior}, {beta_prior})")
       print(f"Data: {k} successes in {n} trials")
       print(f"Posterior: Beta({alpha_post}, {beta_post})")
       print(f"Posterior mean: {post_mean:.4f}")
       print(f"Posterior std:  {np.sqrt(post_var):.4f}")

       return alpha_post, beta_post

   alpha_post, beta_post = beta_binomial_demo()


Markov Chains
===============

MCMC works by constructing a Markov chain whose stationary distribution is the
target posterior. Before building the full algorithm, we need to understand Markov
chains themselves.

A **Markov chain** is a sequence of random variables :math:`X_0, X_1, X_2, \ldots`
where the distribution of :math:`X_{t+1}` depends only on :math:`X_t`, not on
earlier values. This is the same Markov property that drives the HMMs in
:ref:`hmms` -- but here the chain operates in *algorithm time* (MCMC iterations)
rather than along the genome.

Formally, a Markov chain on a state space :math:`\mathcal{S}` is defined by a
**transition kernel** :math:`T(x, y)`: the probability (or probability density) of
moving from state :math:`x` to state :math:`y` in one step.

For a finite state space :math:`\{1, 2, \ldots, K\}`, the transition kernel is a
matrix :math:`T_{ij} = P(X_{t+1} = j \mid X_t = i)`, exactly like the transition
matrix of an HMM. Each row sums to 1, meaning the chain must go somewhere at each
step.

Stationary Distribution
-------------------------

A probability distribution :math:`\pi` over the state space is **stationary** for
the chain if it is unchanged by one step of the transition:

.. math::

   \pi_j = \sum_{i} \pi_i \, T_{ij} \quad \text{for all } j

In matrix notation: :math:`\pi T = \pi`. If the chain starts in distribution
:math:`\pi`, it stays in distribution :math:`\pi` forever.

For the finite chains considered here, irreducibility and aperiodicity guarantee
convergence to the unique stationary distribution regardless of the starting state:

- **Irreducible**: every state can be reached from every other state (eventually).
- **Aperiodic**: the chain does not get trapped in deterministic cycles.

For such a finite chain, no matter where we start, the distribution of :math:`X_t`
converges to :math:`\pi` as :math:`t \to \infty`. This is the fundamental theorem
that makes MCMC work: if we design a chain whose stationary distribution is our
target posterior, then running the chain long enough produces samples from that
posterior. General state spaces require additional recurrence and regularity
conditions; irreducibility and aperiodicity alone are not a universal theorem.

.. admonition:: Probability Aside -- Detailed balance

   A sufficient (but not necessary) condition for :math:`\pi` to be stationary is
   **detailed balance**:

   .. math::

      \pi_i \, T_{ij} = \pi_j \, T_{ji} \quad \text{for all } i, j

   This says: the probability of being in state :math:`i` and transitioning to
   :math:`j` equals the probability of being in state :math:`j` and transitioning
   to :math:`i`. In other words, the "flow" between every pair of states is
   balanced.

   To see that detailed balance implies stationarity, sum both sides over :math:`i`:

   .. math::

      \sum_i \pi_i \, T_{ij} = \sum_i \pi_j \, T_{ji} = \pi_j \sum_i T_{ji} = \pi_j

   where the last step uses :math:`\sum_i T_{ji} = 1` (the rows of :math:`T` sum
   to 1). This gives :math:`\sum_i \pi_i T_{ij} = \pi_j`, which is exactly the
   stationarity condition. Most MCMC algorithms (including Metropolis-Hastings and
   Gibbs sampling) are designed to satisfy detailed balance.

Let us see convergence to a stationary distribution in action with a simple
finite-state Markov chain.

.. code-block:: python

   import numpy as np

   def markov_chain_convergence():
       """Demonstrate that a finite Markov chain converges to its stationary distribution.

       We define a 3-state chain, run it for many steps, and compare the
       empirical state frequencies to the theoretical stationary distribution.
       """
       # A reversible 3-state chain with known stationary distribution.
       # T[i, j] = P(X_{t+1} = j | X_t = i).
       pi = np.array([0.2, 0.3, 0.5])
       T = np.array([
           [0.50, 0.20, 0.30],
           [2/15, 17/30, 0.30],
           [0.12, 0.18, 0.70],
       ])

       # Verify the defining identities, not just the simulation output.
       assert np.allclose(T.sum(axis=1), 1.0), "Rows must sum to 1"
       assert np.allclose(pi @ T, pi), "pi must be stationary"
       flow = pi[:, None] * T
       assert np.allclose(flow, flow.T), "detailed balance must hold"
       print(f"Stationary distribution (theory): {pi}")

       # Simulate the chain for 100,000 steps starting from state 0
       n_steps = 100_000
       rng = np.random.default_rng(42)
       state = 0
       counts = np.zeros(3)

       for _ in range(n_steps):
           # np.random.choice(3, p=T[state]) draws the next state
           # according to the transition probabilities from 'state'.
           state = rng.choice(3, p=T[state])
           counts[state] += 1

       # Empirical frequencies should match the stationary distribution
       empirical = counts / n_steps
       print(f"Empirical frequencies:             {empirical}")
       print(f"Max absolute error:                {np.max(np.abs(pi - empirical)):.4f}")

   markov_chain_convergence()


The Metropolis-Hastings Algorithm
====================================

The **Metropolis-Hastings (MH)** algorithm is the workhorse of MCMC
:cite:p:`metropolis1953,hastings1970`. It constructs
an ergodic Markov chain whose stationary distribution is any target distribution
:math:`\pi(x)` that we can evaluate up to a normalizing constant.

The algorithm is remarkably simple:

1. From the current state :math:`x`, **propose** a new state :math:`x'` from a
   proposal distribution :math:`q(x' \mid x)`.
2. Compute the **acceptance ratio**:

   .. math::

      \alpha = \min\left(1, \; \frac{\pi(x') \, q(x \mid x')}{\pi(x) \, q(x' \mid x)}\right)

3. **Accept** :math:`x'` with probability :math:`\alpha`; otherwise stay at :math:`x`.

The acceptance ratio corrects for proposal asymmetry and makes :math:`\pi` invariant
when the forward and reverse proposal densities required by the ratio are defined.
Convergence from an arbitrary starting point additionally requires an ergodic chain:
the proposal must let the chain reach every relevant part of the target and must not
force a deterministic cycle. A poor proposal can therefore be formally valid yet
practically unusable.

.. admonition:: Probability Aside -- Why the MH ratio works

   We need to verify that the MH algorithm satisfies detailed balance with respect
   to :math:`\pi`. The transition probability from :math:`x` to :math:`x'` is:

   .. math::

      T(x, x') = q(x' \mid x) \cdot \min\left(1, \frac{\pi(x') q(x \mid x')}{\pi(x) q(x' \mid x)}\right)

   To check detailed balance, we need :math:`\pi(x) T(x, x') = \pi(x') T(x', x)`.

   Without loss of generality, assume :math:`\pi(x') q(x \mid x') \leq \pi(x) q(x' \mid x)`.
   Then the acceptance ratio for moving :math:`x \to x'` is
   :math:`\frac{\pi(x') q(x \mid x')}{\pi(x) q(x' \mid x)}`, and the acceptance
   ratio for the reverse move :math:`x' \to x` is 1. So:

   .. math::

      \pi(x) T(x, x') &= \pi(x) \cdot q(x' \mid x) \cdot \frac{\pi(x') q(x \mid x')}{\pi(x) q(x' \mid x)} \\
      &= \pi(x') \cdot q(x \mid x')

   .. math::

      \pi(x') T(x', x) &= \pi(x') \cdot q(x \mid x') \cdot 1 \\
      &= \pi(x') \cdot q(x \mid x')

   Both sides are equal. Detailed balance holds, so :math:`\pi` is the stationary
   distribution.

   The key insight: the normalizing constant of :math:`\pi` cancels in the ratio
   :math:`\pi(x') / \pi(x)`. This is why MCMC does not need to compute the
   evidence :math:`P(D)` -- it only needs the unnormalized posterior.

**Random walk Metropolis-Hastings.** The simplest choice of proposal is a symmetric
random walk: :math:`x' = x + \epsilon` where :math:`\epsilon \sim \mathcal{N}(0, \sigma^2)`.
Since the proposal is symmetric (:math:`q(x' \mid x) = q(x \mid x')`), the ratio
simplifies to:

.. math::

   \alpha = \min\left(1, \frac{\pi(x')}{\pi(x)}\right)

If the proposed state has higher posterior density, always accept. If lower, accept
with probability equal to the density ratio. This allows the chain to explore
regions of lower density (important for characterizing uncertainty) while spending
most of its time in high-density regions.

Let us implement reusable random-walk MH and test it on the Beta posterior derived
above. This is deliberately a problem with an analytic answer: agreement with exact
posterior moments is a stronger correctness check than a visually plausible trace.

.. code-block:: python

   import numpy as np

   def random_walk_metropolis(log_target, initial, proposal_scale,
                              n_samples, rng):
       """Sample a one-dimensional target with a symmetric Normal proposal."""
       if n_samples < 2 or proposal_scale <= 0 or not np.isfinite(initial):
           raise ValueError("invalid sampler arguments")
       current_logp = float(log_target(initial))
       if not np.isfinite(current_logp):
           raise ValueError("initial state must have finite target density")

       samples = np.empty(n_samples)
       samples[0] = initial
       accepted = 0
       for t in range(1, n_samples):
           proposal = samples[t - 1] + rng.normal(scale=proposal_scale)
           proposal_logp = float(log_target(proposal))
           log_ratio = proposal_logp - current_logp
           if np.log(rng.random()) < min(0.0, log_ratio):
               samples[t] = proposal
               current_logp = proposal_logp
               accepted += 1
           else:
               samples[t] = samples[t - 1]
       return samples, accepted / (n_samples - 1)

   def beta_log_kernel(theta, alpha, beta):
       """Unnormalized log density of Beta(alpha, beta)."""
       if not 0.0 < theta < 1.0:
           return -np.inf
       return (alpha - 1) * np.log(theta) + (beta - 1) * np.log1p(-theta)

   alpha_post, beta_post = 9, 15
   rng = np.random.default_rng(42)
   chain, acceptance = random_walk_metropolis(
       lambda x: beta_log_kernel(x, alpha_post, beta_post),
       initial=0.5, proposal_scale=0.15, n_samples=50_000, rng=rng,
   )
   posterior = chain[5_000:]

   exact_mean = alpha_post / (alpha_post + beta_post)
   exact_var = (alpha_post * beta_post) / (
       (alpha_post + beta_post)**2 * (alpha_post + beta_post + 1)
   )
   print(f"Acceptance rate: {acceptance:.3f}")
   print(f"Mean: {posterior.mean():.4f} (exact {exact_mean:.4f})")
   print(f"Variance: {posterior.var():.5f} (exact {exact_var:.5f})")

The Normal random walk proposes values outside :math:`(0,1)`, which are correctly
rejected because their target density is zero. A transformed proposal on the logit
scale can be more efficient, but then its Jacobian or the corresponding asymmetric
Hastings correction must be included.


Gibbs Sampling
================

**Gibbs sampling** is a special case of Metropolis-Hastings where the proposal is
drawn from the **full conditional distribution** -- the distribution of one variable
given all the others. The remarkable property of Gibbs sampling is that every
proposal is accepted.

To see why, suppose we are updating variable :math:`x_k` while holding all other
variables :math:`x_{-k}` fixed. The Gibbs proposal draws :math:`x_k'` from:

.. math::

   q(x_k' \mid x_k, x_{-k}) = P(x_k' \mid x_{-k})

Now compute the MH acceptance ratio. The target distribution is the joint
:math:`\pi(x_k, x_{-k})`, and the proposal is :math:`q(x_k' \mid x_k, x_{-k}) = \pi(x_k' \mid x_{-k})`. Substituting into the MH ratio:

.. math::

   \alpha &= \min\left(1, \frac{\pi(x_k', x_{-k}) \, q(x_k \mid x_k', x_{-k})}{\pi(x_k, x_{-k}) \, q(x_k' \mid x_k, x_{-k})}\right) \\
   &= \min\left(1, \frac{\pi(x_k', x_{-k}) \cdot \pi(x_k \mid x_{-k})}{\pi(x_k, x_{-k}) \cdot \pi(x_k' \mid x_{-k})}\right)

Using the identity :math:`\pi(x_k, x_{-k}) = \pi(x_k \mid x_{-k}) \cdot \pi(x_{-k})`:

.. math::

   \alpha &= \min\left(1, \frac{\pi(x_k' \mid x_{-k}) \, \pi(x_{-k}) \cdot \pi(x_k \mid x_{-k})}{\pi(x_k \mid x_{-k}) \, \pi(x_{-k}) \cdot \pi(x_k' \mid x_{-k})}\right) = \min(1, 1) = 1

Every Gibbs proposal is accepted, but acceptance is not the same as efficiency:
strongly coupled variables can still produce a highly autocorrelated Gibbs chain.
The other requirement is that we can sample from the full conditional
:math:`P(x_k \mid x_{-k})` exactly, which is not always possible.

**Connection to ARGweaver.** ARGweaver (see :ref:`argweaver_mcmc`) uses Gibbs
sampling to update its ARG. At each iteration, one haplotype's "thread" through
the ARG is removed, and a new thread is sampled from the conditional posterior
:math:`P(\text{thread}_k \mid \text{ARG}_{-k}, D)`. Because the time-discretized
HMM allows exact computation of this conditional via the forward algorithm and
stochastic traceback, the Gibbs update is exact **within ARGweaver's
time-discretized model** and the acceptance rate is 1 :cite:p:`argweaver`. This
is precisely the strategy described in the :ref:`HMM chapter <hmms>`: the forward
algorithm computes state probabilities, and stochastic traceback samples a path.

Let us implement Gibbs sampling for a bivariate Normal distribution, where the
conditional distributions are known analytically.

.. code-block:: python

   import numpy as np

   def gibbs_bivariate_normal():
       """Gibbs sampling from a bivariate Normal distribution.

       Target: (X, Y) ~ N(mu, Sigma) where
           mu = (0, 0)
           Sigma = [[1, rho], [rho, 1]]

       The conditional distributions are:
           X | Y=y ~ N(rho*y, 1 - rho^2)
           Y | X=x ~ N(rho*x, 1 - rho^2)

       These conditionals are easy to sample from, making Gibbs ideal.
       """
       rho = 0.8                # correlation coefficient
       cond_var = 1 - rho**2    # conditional variance
       cond_std = np.sqrt(cond_var)

       n_samples = 20_000
       samples = np.zeros((n_samples, 2))
       samples[0] = [0.0, 0.0]  # starting point

       for t in range(1, n_samples):
           # Update X given Y: X | Y ~ N(rho * Y, 1 - rho^2)
           y_current = samples[t - 1, 1]
           samples[t, 0] = np.random.normal(rho * y_current, cond_std)

           # Update Y given X: Y | X ~ N(rho * X, 1 - rho^2)
           x_current = samples[t, 0]  # use the NEWLY sampled X
           samples[t, 1] = np.random.normal(rho * x_current, cond_std)

       burn_in = 1000
       post_burnin = samples[burn_in:]

       # Check moments against the true distribution
       print(f"Correlation rho = {rho}")
       print(f"Mean X: {post_burnin[:, 0].mean():.4f} (expected 0.0)")
       print(f"Mean Y: {post_burnin[:, 1].mean():.4f} (expected 0.0)")
       print(f"Var X:  {post_burnin[:, 0].var():.4f} (expected 1.0)")
       print(f"Var Y:  {post_burnin[:, 1].var():.4f} (expected 1.0)")
       empirical_corr = np.corrcoef(post_burnin[:, 0], post_burnin[:, 1])[0, 1]
       print(f"Empirical correlation: {empirical_corr:.4f} (expected {rho})")

       # Gibbs always accepts, so acceptance rate is 1.0
       print(f"Acceptance rate: 1.0 (by construction)")

       return post_burnin

   np.random.seed(42)
   gibbs_samples = gibbs_bivariate_normal()


Convergence Diagnostics
=========================

Running an MCMC chain is only half the battle. How do we know the chain has
converged to the target distribution? How many samples are actually useful? These
questions are addressed by **convergence diagnostics**.

**Warmup and initial transients.** Early draws can retain substantial dependence on
the starting point. Warmup is also the usual phase for adapting proposal parameters;
for ordinary MCMC, adaptation is then frozen before collecting draws. Discarding a
fixed fraction does not make a chain converge. Run multiple chains from dispersed
initial values and extend them until diagnostics and quantities of interest are
stable; persistent disagreement calls for a better parameterization or sampler,
not merely a larger discarded fraction.

**Trace plots.** A trace plot shows the sampled values as a function of iteration
number. A poorly mixed chain can show slow drifts, long repeated stretches, or
different regions in different chains. A plausible-looking trace is useful evidence
but cannot prove convergence.

**Autocorrelation.** Consecutive MCMC samples are correlated (each sample is a small
perturbation of the previous one). The **autocorrelation function** (ACF) at lag
:math:`k` measures this:

.. math::

   \rho(k) = \frac{\text{Cov}(X_t, X_{t+k})}{\text{Var}(X_t)}

For a well-mixed chain, the ACF decays quickly to zero. For a poorly mixed chain,
it remains positive for many lags, meaning consecutive samples carry redundant
information.

**Thinning.** To reduce autocorrelation, we can keep only every :math:`m`-th sample.
Thinning can reduce storage costs, but it usually discards useful information and
does not repair poor mixing. It is generally better to retain draws and quantify
their autocorrelation through ESS.

.. admonition:: Probability Aside -- Effective sample size

   The **effective sample size** (ESS) quantifies how many *independent* samples
   your chain is equivalent to. If you have :math:`N` total samples with
   autocorrelation, the ESS is:

   .. math::

      \text{ESS} = \frac{N}{1 + 2\sum_{k=1}^{\infty} \rho(k)}

   where :math:`\rho(k)` is the autocorrelation at lag :math:`k`. The denominator
   is called the **integrated autocorrelation time** (IAT) :math:`\tau`:

   .. math::

      \tau = 1 + 2\sum_{k=1}^{\infty} \rho(k)

   and :math:`\text{ESS} = N / \tau`. If :math:`\tau = 10`, then 10,000 MCMC
   samples are worth only about 1,000 independent samples.

   High autocorrelation (large :math:`\tau`, small ESS) means the chain is exploring
   slowly -- each new sample does not move far from the previous one. This is the
   central diagnostic of MCMC efficiency: a well-tuned algorithm has small
   :math:`\tau` and large ESS.

**Rank-normalized split** :math:`\hat{R}`. Modern :math:`\hat{R}` compares split,
rank-normalized chains and is paired with bulk and tail ESS
:cite:p:`vehtari2021`. Values near 1 are necessary but not sufficient evidence of
mixing; :math:`\hat{R} > 1.01` is a useful warning threshold, not a proof-producing
pass/fail rule. Always inspect several chains and the estimands that matter.

.. code-block:: python

   import numpy as np

   def compute_acf_and_ess(chain, max_lag=200):
       """Compute the autocorrelation function and effective sample size.

       Parameters
       ----------
       chain : ndarray of shape (N,)
           MCMC samples (after burn-in).
       max_lag : int
           Maximum lag to compute ACF for.

       Returns
       -------
       acf : ndarray of shape (max_lag + 1,)
           Autocorrelation at each lag from 0 to max_lag.
       ess : float
           Estimated effective sample size.
       """
       chain = np.asarray(chain, dtype=float)
       if chain.ndim != 1 or len(chain) < 4 or not np.all(np.isfinite(chain)):
           raise ValueError("chain must be a finite one-dimensional array")
       N = len(chain)
       max_lag = min(int(max_lag), N - 1)
       if max_lag < 2:
           raise ValueError("max_lag must allow at least one lag pair")
       centered = chain - chain.mean()
       var = np.dot(centered, centered) / N
       if var == 0:
           return np.ones(max_lag + 1), 1.0

       # Compute autocorrelation at each lag
       acf = np.zeros(max_lag + 1)
       for k in range(max_lag + 1):
           # Autocorrelation at lag k:
           # rho(k) = (1/N) * sum_{t=0}^{N-k-1} (x_t - mean)(x_{t+k} - mean) / var
           if k == 0:
               acf[k] = 1.0
           else:
               # chain[:-k] is the series shifted by 0 (first N-k elements)
               # chain[k:] is the series shifted by k (last N-k elements)
               acf[k] = np.dot(centered[:-k], centered[k:]) / ((N - k) * var)

       # Geyer's paired initial-positive sequence, made non-increasing.
       # Pair rho(1)+rho(2), rho(3)+rho(4), ... to stabilize the cutoff.
       paired_sum = 0.0
       previous_pair = np.inf
       for k in range(1, max_lag, 2):
           pair = acf[k] + acf[k + 1]
           if pair <= 0:
               break
           pair = min(pair, previous_pair)
           paired_sum += pair
           previous_pair = pair

       iat = max(1.0, 1.0 + 2.0 * paired_sum)
       ess = min(float(N), N / iat)

       return acf, ess

   # Simple MH chain targeting N(0,1) with different step sizes
   def run_mh_chain(sigma, n_samples=20_000, seed=42):
       """Run MH targeting standard Normal with step size sigma."""
       rng = np.random.default_rng(seed)
       chain = np.zeros(n_samples)
       chain[0] = 5.0  # start far from the mode
       n_acc = 0
       for t in range(1, n_samples):
           proposal = chain[t-1] + rng.normal(0, sigma)
           log_alpha = -0.5 * proposal**2 + 0.5 * chain[t-1]**2
           if np.log(rng.random()) < min(0.0, log_alpha):
               chain[t] = proposal
               n_acc += 1
           else:
               chain[t] = chain[t-1]
       return chain[2000:], n_acc / (n_samples - 1)

   for sigma in [0.1, 1.0, 2.4, 10.0]:
       chain, acc_rate = run_mh_chain(sigma)
       acf, ess = compute_acf_and_ess(chain)
       print(f"sigma={sigma:5.1f}: acceptance={acc_rate:.3f}, "
             f"ESS={ess:.0f}, IAT={len(chain)/ess:.1f}, "
             f"ACF at lag 10={acf[10]:.3f}")


Practical Considerations
==========================

The theoretical foundations of MCMC are elegant, but making MCMC work well in
practice requires careful attention to several practical issues. These issues are
not mere technicalities -- they determine whether your MCMC run produces useful
results in hours or useless results in weeks.

Proposal Tuning
-----------------

The most critical practical decision in random walk MH is the **proposal step
size** :math:`\sigma`. This is the standard deviation of the Normal perturbation
:math:`\epsilon \sim \mathcal{N}(0, \sigma^2)`.

- **Too small** (:math:`\sigma \ll 1`): Almost every proposal is accepted (the new
  state is barely different from the old one), but the chain explores very slowly.
  It takes many steps to traverse the parameter space. The autocorrelation is high
  and the ESS is low.

- **Too large** (:math:`\sigma \gg 1`): Most proposals land in regions of very low
  posterior density and are rejected. The chain gets stuck at one location for many
  steps before a proposal is finally accepted. Again, autocorrelation is high and
  ESS is low.

- **Just right**: There is a sweet spot where the chain makes reasonably large moves
  that are accepted often enough to explore efficiently. The famous **23.4%** limit
  applies asymptotically to a particular high-dimensional random-walk scaling regime
  for product-like targets :cite:p:`roberts1997`; the corresponding one-dimensional
  limit is about 44%. These are tuning clues, not universal targets. Efficiency is
  better assessed using ESS per unit of computation.

.. code-block:: python

   import numpy as np

   def proposal_tuning_demo():
       """Demonstrate the effect of proposal step size on MCMC efficiency.

       We run MH targeting a 5-dimensional standard Normal with different
       step sizes and compare acceptance rates and ESS.
       """
       d = 5          # dimensionality
       n_samples = 20_000
       rng = np.random.default_rng(42)

       def log_target(x):
           """Log density of a d-dimensional standard Normal."""
           return -0.5 * np.sum(x**2)

       results = []
       for sigma in [0.05, 0.2, 0.5, 1.0, 2.4 / np.sqrt(d), 3.0, 10.0]:
           chain = np.zeros((n_samples, d))
           chain[0] = np.zeros(d)
           n_accepted = 0

           for t in range(1, n_samples):
               proposal = chain[t-1] + rng.normal(0, sigma, size=d)
               log_alpha = log_target(proposal) - log_target(chain[t-1])
               if np.log(rng.random()) < min(0.0, log_alpha):
                   chain[t] = proposal
                   n_accepted += 1
               else:
                   chain[t] = chain[t-1]

           acc_rate = n_accepted / (n_samples - 1)

           # Compute ESS for the first component
           burn_in = 2_000
           first_coord = chain[burn_in:, 0]
           _, ess = compute_acf_and_ess(first_coord, max_lag=400)
           results.append((sigma, acc_rate, ess))
           print(f"sigma={sigma:.3f}: acceptance={acc_rate:.3f}, ESS={ess:.0f}")

       best = max(results, key=lambda r: r[2])
       print(f"\nLargest estimated ESS: sigma={best[0]:.3f}, "
             f"acceptance={best[1]:.3f}, ESS={best[2]:.0f}")

   proposal_tuning_demo()

Data-Informed Proposals
-------------------------

Random walk proposals are simple but uninformed -- they do not use the data to
guide the exploration. This is like a blind watchmaker making random adjustments
without listening to whether the mechanism sounds better or worse.

**SINGER's innovation** (see :ref:`sgpr`) is to replace the random walk with a
**data-informed proposal**. In SINGER's SGPR (Sub-Graph Pruning and Re-grafting)
move, a piece of the ARG is removed and then re-threaded using the forward
algorithm and stochastic traceback from the :ref:`HMM chapter <hmms>`. This
proposal incorporates the observed sequence data directly: the HMM "listens" to
the data and proposes a new thread that is already likely under the posterior.

The proposal is designed to achieve high acceptance and efficient movement in the
settings evaluated by SINGER :cite:p:`singer`. The realized acceptance rate is
model- and data-dependent; neither an HMM-informed proposal nor a rate near one by
itself guarantees good global mixing.

Parallel Tempering
--------------------

Multimodal posteriors (distributions with multiple well-separated peaks) are
notoriously difficult for standard MCMC. The chain can get trapped in one mode
for a very long time before finding enough momentum to jump to another.

**Parallel tempering** (also called replica exchange) addresses this by running
multiple chains at different "temperatures." A chain at temperature :math:`T`
targets the tempered distribution :math:`\pi(x)^{1/T}`. At :math:`T = 1`, this is
the original posterior. At :math:`T > 1`, the distribution is flattened -- the
valleys between modes are shallower, making it easier to cross between them.

Periodically, adjacent-temperature chains propose to swap their states. The swap
acceptance probability ensures that the :math:`T = 1` chain still targets the correct
posterior. Hot chains explore broadly and pass their discoveries to colder chains.

When MCMC Is Not Enough
--------------------------

Sometimes the parameter space is so large, or the posterior is so complex, that
even well-tuned MCMC cannot converge in a reasonable time. This is the situation
faced by **PHLASH** (see :ref:`phlash_svgd`), which needs to infer a
high-dimensional population size history.

Instead of MCMC, PHLASH uses **Stein Variational Gradient Descent (SVGD)** -- a
deterministic optimization method that maintains a collection of "particles" and
moves them to approximate the posterior. In PHLASH's reported experiments, this
gradient-based particle approximation gives favorable speed and accuracy for
population-size inference :cite:p:`phlash`. That empirical result should not be
generalized into a theorem that SVGD is always faster than MCMC in high dimensions.

The trade-off: SVGD is an approximation (it may not converge to the exact
posterior), while an invariant, ergodic, correctly implemented MCMC chain is
asymptotically exact for its specified target. For the
specific problem PHLASH solves -- inferring piecewise-constant population size
histories -- the speed advantage of SVGD outweighs the loss in exactness.


MCMC in Population Genetics: Three Applications
==================================================

The Timepieces in this book use three different strategies for exploring posterior
distributions. Each one can be understood as a variation on the MCMC theme
developed in this chapter.

ARGweaver: Gibbs Sampling over ARGs
--------------------------------------

ARGweaver (see :ref:`argweaver_mcmc`) uses **Gibbs sampling** to explore the space
of ancestral recombination graphs. At each iteration, one haplotype's thread is
removed from the ARG, and a new thread is sampled from the exact conditional
posterior :math:`P(\text{thread}_k \mid \text{ARG}_{-k}, D)`.

This is possible because ARGweaver discretizes time (see
:ref:`argweaver_time_discretization`), reducing the continuous coalescent to a
finite-state HMM. The forward algorithm (from :ref:`hmms`) computes the conditional
posterior exactly within that discretized model, and stochastic traceback draws a
sample. Because the draw comes from the full conditional, its Gibbs acceptance rate
is 1; successive ARG states can nevertheless remain strongly correlated
:cite:p:`argweaver`.

The cost of this elegance is the time discretization itself: it introduces an
approximation, and the number of HMM states grows with the number of time points
and samples. An acceptance rate of 100% does not remove the need to assess mixing.

SINGER: MH with Data-Informed Proposals
------------------------------------------

SINGER (see :ref:`sgpr`) also updates the ARG one thread at a time, but it works in
**continuous time** -- no discretization is needed. The SGPR (Sub-Graph Pruning and
Re-grafting) move removes a sub-graph from the ARG and re-threads it using the
branch sampling and time sampling HMMs.

Because the proposal distribution is constructed from the data (via the HMM forward
algorithm), it closely approximates the posterior. The Metropolis-Hastings acceptance
ratio can therefore be high, though it is not identically 1 as in Gibbs sampling. The key
formula (derived in :ref:`sgpr`) compares the probability of the old thread under
the new proposal to the probability of the new thread under the old proposal,
combined with the prior ratio.

SINGER's innovation is that the data-informed proposal makes MCMC practical for
continuous-time ARG inference -- something that would be hopelessly inefficient
with random walk proposals.

PHLASH: Beyond MCMC
----------------------

PHLASH (see :ref:`phlash_svgd`) takes a different path entirely. Instead of
sampling from the posterior via MCMC, it uses **Stein Variational Gradient Descent
(SVGD)** to directly approximate the posterior with a set of particles.

Why abandon MCMC? PHLASH infers a high-dimensional population size history
:math:`\eta(t)`, parameterized as a piecewise-constant function with many epochs.
The parameter space is large enough that MCMC chains converge very slowly, and the
composite likelihood used by PHLASH (which approximates the full likelihood for
computational efficiency) makes exact Gibbs updates unavailable.

SVGD maintains a collection of particles and iteratively moves them to minimize a
divergence from the posterior. Each particle update uses the gradient of the
log-posterior (the "score function," see :ref:`phlash_score_function`), making the
exploration more directed than random walk MCMC. The method produced fast inference
in the experiments reported for PHLASH, while targeting an approximation rather
than supplying MCMC's asymptotic invariance guarantee :cite:p:`phlash`.


Summary
=========

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Concept
     - Key Idea
   * - Bayesian posterior
     - :math:`P(G \mid D) \propto P(D \mid G) \, P(G)`;
       the normalizing constant :math:`P(D)` is intractable
   * - Markov chain
     - A sequence where the next state depends only on the current state;
       characterized by a transition kernel and stationary distribution
   * - Detailed balance
     - :math:`\pi(x) T(x, y) = \pi(y) T(y, x)`;
       sufficient condition for :math:`\pi` to be stationary
   * - Metropolis-Hastings
     - Propose, then accept/reject with ratio
       :math:`\min(1, \frac{\pi(x')q(x|x')}{\pi(x)q(x'|x)})`;
       requires valid forward/reverse support and an ergodic chain
   * - Gibbs sampling
     - Propose from the full conditional; always accepted;
       requires tractable conditionals
   * - Effective sample size
     - ESS = :math:`N / (1 + 2\sum_k \rho(k))`;
       measures how many independent samples the chain yields
   * - Proposal tuning
     - The :math:`23.4\%` limit is specific to high-dimensional product-target
       asymptotics; tune using ESS per unit of computation
   * - Data-informed proposals
     - SINGER's SGPR uses HMM-based proposals that incorporate the data,
       often yielding high acceptance in reported applications
   * - When MCMC fails
     - High-dimensional or complex posteriors may require alternatives
       like SVGD (used by PHLASH)

These tools form the **winding mechanism** of several Timepieces. ARGweaver
(see :ref:`argweaver_timepiece`) uses Gibbs sampling to cycle through haplotype
threads, producing exact conditional updates. SINGER (see :ref:`singer_timepiece`)
uses Metropolis-Hastings with data-informed SGPR proposals to explore the space of
continuous-time ARGs. And PHLASH (see :ref:`phlash_timepiece`) demonstrates when the
MCMC paradigm itself must be transcended, replacing random sampling with
gradient-guided particle optimization. Understanding the strengths and limitations of
MCMC -- when it shines and when it is not enough -- is essential for understanding
why each Timepiece is built the way it is.
