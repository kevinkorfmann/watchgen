.. _mcmc:

===========================
Markov Chain Monte Carlo
===========================

   *"When you can't find the answer, let randomness find it for you -- one carefully guided step at a time."*

A master watchmaker, confronted with a broken mechanism of unknown design, cannot
simply enumerate every possible gear arrangement to find the one that matches the
symptoms. The space of possibilities is too vast. Instead, the watchmaker makes an
educated guess, tries a small modification, checks whether the watch runs better, and
repeats. Over time, the modifications converge on the true mechanism -- not by
exhaustive search, but by guided exploration.

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
distribution. After enough steps, the states visited by the chain are (approximately)
independent draws from :math:`P(G \mid D)`. We can then use these samples to estimate
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
   from scipy import stats

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

       # Verify: the posterior should integrate to 1
       from scipy.special import beta as beta_fn
       integral = beta_fn(alpha_post, beta_post)
       print(f"Beta function B({alpha_post},{beta_post}) = {integral:.6f} (normalization constant)")

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

**Ergodicity** is the property that guarantees the chain converges to :math:`\pi`
regardless of the starting state. A chain is ergodic if it is:

- **Irreducible**: every state can be reached from every other state (eventually).
- **Aperiodic**: the chain does not get trapped in deterministic cycles.

For an ergodic chain, no matter where we start, the distribution of :math:`X_t`
converges to :math:`\pi` as :math:`t \to \infty`. This is the fundamental theorem
that makes MCMC work: if we design a chain whose stationary distribution is our
target posterior, then running the chain long enough produces samples from that
posterior.

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
       # Transition matrix for a 3-state chain
       # T[i, j] = P(X_{t+1} = j | X_t = i)
       T = np.array([
           [0.7, 0.2, 0.1],   # from state 0
           [0.1, 0.6, 0.3],   # from state 1
           [0.3, 0.3, 0.4],   # from state 2
       ])

       # Verify rows sum to 1
       assert np.allclose(T.sum(axis=1), 1.0), "Rows must sum to 1"

       # Find the stationary distribution by solving pi * T = pi
       # This is equivalent to finding the left eigenvector with eigenvalue 1.
       eigenvalues, eigenvectors = np.linalg.eig(T.T)
       # Find the eigenvector corresponding to eigenvalue 1
       idx = np.argmin(np.abs(eigenvalues - 1.0))
       pi = np.real(eigenvectors[:, idx])
       pi = pi / pi.sum()   # normalize to a probability distribution
       print(f"Stationary distribution (theory): {pi}")

       # Simulate the chain for 100,000 steps starting from state 0
       n_steps = 100_000
       state = 0
       counts = np.zeros(3)

       for _ in range(n_steps):
           # np.random.choice(3, p=T[state]) draws the next state
           # according to the transition probabilities from 'state'.
           state = np.random.choice(3, p=T[state])
           counts[state] += 1

       # Empirical frequencies should match the stationary distribution
       empirical = counts / n_steps
       print(f"Empirical frequencies:             {empirical}")
       print(f"Max absolute error:                {np.max(np.abs(pi - empirical)):.4f}")

       # Verify detailed balance: pi[i]*T[i,j] should equal pi[j]*T[j,i]
       for i in range(3):
           for j in range(i+1, 3):
               lhs = pi[i] * T[i, j]
               rhs = pi[j] * T[j, i]
               print(f"  pi[{i}]*T[{i},{j}] = {lhs:.4f}, pi[{j}]*T[{j},{i}] = {rhs:.4f}, "
                     f"ratio = {lhs/rhs:.4f}")

   np.random.seed(42)
   markov_chain_convergence()


The Metropolis-Hastings Algorithm
====================================

The **Metropolis-Hastings (MH)** algorithm is the workhorse of MCMC. It constructs
an ergodic Markov chain whose stationary distribution is any target distribution
:math:`\pi(x)` that we can evaluate up to a normalizing constant.

The algorithm is remarkably simple:

1. From the current state :math:`x`, **propose** a new state :math:`x'` from a
   proposal distribution :math:`q(x' \mid x)`.
2. Compute the **acceptance ratio**:

   .. math::

      \alpha = \min\left(1, \; \frac{\pi(x') \, q(x \mid x')}{\pi(x) \, q(x' \mid x)}\right)

3. **Accept** :math:`x'` with probability :math:`\alpha`; otherwise stay at :math:`x`.

That is the entire algorithm. The magic is in the acceptance ratio: it automatically
adjusts for the proposal distribution, ensuring that the chain's stationary
distribution is :math:`\pi(x)` regardless of the choice of :math:`q`.

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

Let us implement MH and use it to sample from a mixture of Gaussians -- a
distribution with multiple modes that tests whether the chain can explore the full
landscape.

.. code-block:: python

   import numpy as np

   def metropolis_hastings_mixture():
       """MH sampling from a mixture of two Gaussians.

       Target: 0.3 * N(-2, 0.5^2) + 0.7 * N(3, 1^2)
       This tests whether the chain can jump between modes.
       """
       def log_target(x):
           """Log of the (unnormalized) target density.

           A mixture of two Gaussians. We work in log space to avoid
           underflow when the density is very small.
           """
           # scipy.special.logsumexp would be more numerically stable,
           # but for clarity we use the direct computation.
           comp1 = 0.3 * np.exp(-0.5 * ((x + 2) / 0.5)**2) / (0.5 * np.sqrt(2 * np.pi))
           comp2 = 0.7 * np.exp(-0.5 * ((x - 3) / 1.0)**2) / (1.0 * np.sqrt(2 * np.pi))
           return np.log(comp1 + comp2 + 1e-300)  # small constant prevents log(0)

       # MH parameters
       n_samples = 50_000
       sigma = 1.5          # proposal standard deviation (step size)
       samples = np.zeros(n_samples)
       samples[0] = 0.0     # starting point
       n_accepted = 0

       for t in range(1, n_samples):
           # Propose: symmetric random walk
           x_current = samples[t - 1]
           x_proposed = x_current + np.random.normal(0, sigma)

           # Acceptance ratio (in log space to avoid overflow/underflow)
           log_alpha = log_target(x_proposed) - log_target(x_current)

           # Accept or reject
           # np.log(np.random.uniform()) gives a uniform draw in log space
           if np.log(np.random.uniform()) < log_alpha:
               samples[t] = x_proposed
               n_accepted += 1
           else:
               samples[t] = x_current

       acceptance_rate = n_accepted / (n_samples - 1)

       # Discard burn-in (first 5000 samples)
       burn_in = 5000
       post_burnin = samples[burn_in:]

       print(f"Acceptance rate: {acceptance_rate:.3f}")
       print(f"Sample mean: {post_burnin.mean():.3f}")
       print(f"Sample std:  {post_burnin.std():.3f}")

       # Check: samples near each mode
       near_mode1 = np.sum(post_burnin < 0) / len(post_burnin)
       near_mode2 = np.sum(post_burnin >= 0) / len(post_burnin)
       print(f"Fraction near mode 1 (x<0): {near_mode1:.3f} (expected ~0.3)")
       print(f"Fraction near mode 2 (x>=0): {near_mode2:.3f} (expected ~0.7)")

       return samples

   np.random.seed(42)
   samples_mixture = metropolis_hastings_mixture()

Now let us apply MH to a problem closer to population genetics: inferring the
mutation rate parameter :math:`\theta` from an observed site frequency spectrum
(SFS).

.. code-block:: python

   import numpy as np

   def mh_sfs_inference():
       """MH for inferring theta from an observed site frequency spectrum.

       Under the standard neutral coalescent with n samples, the expected
       number of SNPs with i derived alleles (out of n) is:
           E[SFS_i] = theta / i   for i = 1, ..., n-1

       We observe an SFS and infer theta using MCMC.
       """
       # Simulated "observed" SFS for n=10 samples with true theta=5
       n = 10
       theta_true = 5.0
       # Expected SFS: theta/i for i = 1, ..., n-1
       expected_sfs = theta_true / np.arange(1, n)
       # Observed SFS: Poisson draws around the expected values
       np.random.seed(123)
       observed_sfs = np.random.poisson(expected_sfs)
       print(f"Observed SFS: {observed_sfs}")

       def log_likelihood(theta, sfs):
           """Poisson log-likelihood for the SFS given theta.

           Each SFS entry sfs[i] is Poisson with mean theta/(i+1).
           The log-likelihood is sum of Poisson log-PMFs.
           """
           if theta <= 0:
               return -np.inf
           ll = 0.0
           for i in range(len(sfs)):
               lam = theta / (i + 1)        # expected count for frequency class i+1
               # Poisson log-PMF: k*log(lam) - lam - log(k!)
               ll += sfs[i] * np.log(lam) - lam
           return ll

       def log_prior(theta):
           """Log of an exponential prior with mean 10."""
           if theta <= 0:
               return -np.inf
           return -theta / 10.0  # Exp(rate=0.1), ignoring the constant

       # Run MH
       n_samples = 30_000
       sigma = 0.5
       chain = np.zeros(n_samples)
       chain[0] = 1.0   # start far from the true value
       n_accepted = 0

       for t in range(1, n_samples):
           theta_current = chain[t - 1]
           theta_proposed = theta_current + np.random.normal(0, sigma)

           log_alpha = (log_likelihood(theta_proposed, observed_sfs) + log_prior(theta_proposed)
                        - log_likelihood(theta_current, observed_sfs) - log_prior(theta_current))

           if np.log(np.random.uniform()) < log_alpha:
               chain[t] = theta_proposed
               n_accepted += 1
           else:
               chain[t] = theta_current

       burn_in = 5000
       post_burnin = chain[burn_in:]

       print(f"True theta: {theta_true}")
       print(f"Posterior mean: {post_burnin.mean():.3f}")
       print(f"Posterior std:  {post_burnin.std():.3f}")
       print(f"95% CI: ({np.percentile(post_burnin, 2.5):.3f}, "
             f"{np.percentile(post_burnin, 97.5):.3f})")
       print(f"Acceptance rate: {n_accepted / (n_samples - 1):.3f}")

   np.random.seed(42)
   mh_sfs_inference()


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

Every Gibbs proposal is accepted. This is enormously efficient -- no samples are
wasted on rejected proposals. The price is that we must be able to sample from the
full conditional :math:`P(x_k \mid x_{-k})` exactly, which is not always possible.

**Connection to ARGweaver.** ARGweaver (see :ref:`argweaver_mcmc`) uses Gibbs
sampling to update its ARG. At each iteration, one haplotype's "thread" through
the ARG is removed, and a new thread is sampled from the conditional posterior
:math:`P(\text{thread}_k \mid \text{ARG}_{-k}, D)`. Because the time-discretized
HMM allows exact computation of this conditional via the forward algorithm and
stochastic traceback, the Gibbs update is exact and the acceptance rate is 1. This
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

**Burn-in.** The initial samples from an MCMC chain are influenced by the starting
point, not by the target distribution. The period before the chain has "forgotten"
its starting point is called the **burn-in**. These samples should be discarded.
There is no universal formula for how long the burn-in should be -- it depends on
the problem, the proposal, and the starting point. A common heuristic is to discard
the first 10--50% of the chain.

**Trace plots.** A trace plot shows the sampled values as a function of iteration
number. A well-mixed chain looks like a "hairy caterpillar" -- rapidly fluctuating
around a stable mean, with no long-term trends or flat regions. A poorly mixed chain
shows slow drifts, long periods stuck at one value, or clear trends.

**Autocorrelation.** Consecutive MCMC samples are correlated (each sample is a small
perturbation of the previous one). The **autocorrelation function** (ACF) at lag
:math:`k` measures this:

.. math::

   \rho(k) = \frac{\text{Cov}(X_t, X_{t+k})}{\text{Var}(X_t)}

For a well-mixed chain, the ACF decays quickly to zero. For a poorly mixed chain,
it remains positive for many lags, meaning consecutive samples carry redundant
information.

**Thinning.** To reduce autocorrelation, we can keep only every :math:`m`-th sample.
If the ACF drops to near zero at lag :math:`m`, thinning by :math:`m` produces
approximately independent samples. However, thinning discards information and is
often unnecessary -- it is usually better to keep all samples and account for
autocorrelation through ESS.

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

**Gelman-Rubin** :math:`\hat{R}`. When running multiple chains from different
starting points, the :math:`\hat{R}` statistic compares the variance within each
chain to the variance between chains. If :math:`\hat{R} \approx 1`, the chains have
converged to the same distribution. Values significantly above 1 (say,
:math:`\hat{R} > 1.1`) indicate that the chains have not yet converged.

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
       N = len(chain)
       mean = chain.mean()
       var = chain.var()

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
               acf[k] = np.mean((chain[:-k] - mean) * (chain[k:] - mean)) / var

       # Compute ESS using the initial monotone sequence estimator:
       # sum autocorrelations until they become negative (conservative cutoff)
       iat = 1.0  # integrated autocorrelation time, starts at 1 (lag 0)
       for k in range(1, max_lag + 1):
           if acf[k] < 0:
               break
           iat += 2 * acf[k]

       ess = N / iat

       return acf, ess

   # Demonstrate with the MH chain from the SFS inference example
   # First, regenerate the chain
   np.random.seed(42)

   # Simple MH chain targeting N(0,1) with different step sizes
   def run_mh_chain(sigma, n_samples=20000):
       """Run MH targeting standard Normal with step size sigma."""
       chain = np.zeros(n_samples)
       chain[0] = 5.0  # start far from the mode
       n_acc = 0
       for t in range(1, n_samples):
           proposal = chain[t-1] + np.random.normal(0, sigma)
           log_alpha = -0.5 * proposal**2 + 0.5 * chain[t-1]**2
           if np.log(np.random.uniform()) < log_alpha:
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
  that are accepted often enough to explore efficiently. For random walk MH on a
  :math:`d`-dimensional target, the optimal acceptance rate is approximately
  **23.4%** (Roberts et al., 1997). This theoretical result, while derived for
  specific conditions, provides a useful practical guideline.

.. code-block:: python

   import numpy as np

   def proposal_tuning_demo():
       """Demonstrate the effect of proposal step size on MCMC efficiency.

       We run MH targeting a 5-dimensional standard Normal with different
       step sizes and compare acceptance rates and ESS.
       """
       d = 5          # dimensionality
       n_samples = 50_000

       def log_target(x):
           """Log density of a d-dimensional standard Normal."""
           return -0.5 * np.sum(x**2)

       results = []
       for sigma in [0.05, 0.2, 0.5, 1.0, 2.4 / np.sqrt(d), 3.0, 10.0]:
           chain = np.zeros((n_samples, d))
           chain[0] = np.zeros(d)
           n_accepted = 0

           for t in range(1, n_samples):
               proposal = chain[t-1] + np.random.normal(0, sigma, size=d)
               log_alpha = log_target(proposal) - log_target(chain[t-1])
               if np.log(np.random.uniform()) < log_alpha:
                   chain[t] = proposal
                   n_accepted += 1
               else:
                   chain[t] = chain[t-1]

           acc_rate = n_accepted / (n_samples - 1)

           # Compute ESS for the first component
           burn_in = 5000
           first_coord = chain[burn_in:, 0]
           N = len(first_coord)
           mean = first_coord.mean()
           var = first_coord.var()

           iat = 1.0
           for k in range(1, 500):
               if k >= N:
                   break
               rho_k = np.mean((first_coord[:-k] - mean) * (first_coord[k:] - mean)) / var
               if rho_k < 0:
                   break
               iat += 2 * rho_k

           ess = N / iat
           results.append((sigma, acc_rate, ess))
           print(f"sigma={sigma:.3f}: acceptance={acc_rate:.3f}, ESS={ess:.0f}")

       # Find the step size closest to 23.4% acceptance
       best = min(results, key=lambda r: abs(r[1] - 0.234))
       print(f"\nClosest to optimal (23.4%): sigma={best[0]:.3f} "
             f"with acceptance={best[1]:.3f} and ESS={best[2]:.0f}")

   np.random.seed(42)
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

The result is dramatically higher acceptance rates -- approaching 1.0 for large
sample sizes -- compared to the random walk proposals that earlier ARG sampling
methods used. This is the difference between a blind watchmaker making random
adjustments and one who carefully examines the mechanism before each modification.

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
moves them to approximate the posterior. SVGD is faster than MCMC for
high-dimensional problems because it uses gradient information (the slope of the
log-posterior) to guide all particles simultaneously, rather than making random
proposals one at a time.

The trade-off: SVGD is an approximation (it may not converge to the exact
posterior), while MCMC is exact in the limit of infinite samples. For the
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
posterior exactly, and stochastic traceback draws a sample. Because the sample comes
from the exact conditional, the Gibbs acceptance rate is 1 -- no proposals are
wasted.

The cost of this elegance is the time discretization itself: it introduces an
approximation, and the number of HMM states grows with the number of time points
and samples. But the guarantee of 100% acceptance makes the approach efficient
despite these limitations.

SINGER: MH with Data-Informed Proposals
------------------------------------------

SINGER (see :ref:`sgpr`) also updates the ARG one thread at a time, but it works in
**continuous time** -- no discretization is needed. The SGPR (Sub-Graph Pruning and
Re-grafting) move removes a sub-graph from the ARG and re-threads it using the
branch sampling and time sampling HMMs.

Because the proposal distribution is constructed from the data (via the HMM forward
algorithm), it closely approximates the posterior. The Metropolis-Hastings acceptance
ratio is therefore close to 1, though not exactly 1 as in Gibbs sampling. The key
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
exploration much more directed than random walk MCMC. The result is fast convergence
to an approximate posterior, at the cost of giving up the exactness guarantees of
MCMC.


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
       works for any proposal :math:`q`
   * - Gibbs sampling
     - Propose from the full conditional; always accepted;
       requires tractable conditionals
   * - Effective sample size
     - ESS = :math:`N / (1 + 2\sum_k \rho(k))`;
       measures how many independent samples the chain yields
   * - Proposal tuning
     - Optimal acceptance rate :math:`\approx 23\%` for random walk MH;
       too narrow or too wide proposals reduce ESS
   * - Data-informed proposals
     - SINGER's SGPR uses HMM-based proposals that incorporate the data,
       achieving near-perfect acceptance rates
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
