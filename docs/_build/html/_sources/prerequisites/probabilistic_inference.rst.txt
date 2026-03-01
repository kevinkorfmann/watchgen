.. _probabilistic_inference:

================================================
Likelihood-Based Probabilistic Inference
================================================

   *"The data are fixed; the model is the variable. The likelihood surface tells you which models are consistent with reality."*

A watchmaker presented with a broken watch doesn't guess randomly. She examines
the symptoms -- the watch runs 5 minutes fast per day, the second hand stutters
at 12 o'clock -- and asks: *which internal configurations would produce these
specific symptoms?* The configurations that best explain the symptoms are the
most likely diagnoses.

This is the logic of likelihood-based inference, and it is the inferential
backbone of every Timepiece in this book. Given observed genetic data (the
symptoms), we want to find the demographic history, genealogy, or set of
parameters (the internal configuration) that best explains what we see. This
chapter introduces the framework that makes this possible: the likelihood
function, maximum likelihood estimation, and Bayesian inference. It also
situates these approaches within the broader landscape of modern inference,
including the neural-network-driven methods that have emerged as an
alternative paradigm.

Unlike the other prerequisites in this section, this chapter is *not* about a
single mathematical topic. It is about the inferential logic that connects all
the other topics together. The exponential distribution, the Poisson process,
the HMM forward algorithm, the diffusion equation -- each of these is a tool
on the workbench. Likelihood-based inference is the *method* by which these
tools are put to work: the discipline of asking, for each tool, "what does the
data tell us about the parameters?"


Why Likelihood?
================

Every method in this book follows the same inferential logic:

1. **Propose a model** with parameters :math:`\theta` (a demographic history,
   a genealogy, a set of population sizes and split times).

2. **Derive the probability** of observing the data :math:`D` under this model:
   :math:`P(D \mid \theta)`. This is the **likelihood function**.

3. **Find the parameters** that make the data most probable (maximum
   likelihood), or characterize the full range of plausible parameters
   (Bayesian inference).

The likelihood function is the bridge between models and data. It transforms
an abstract mathematical model -- "the ancestral population had size 10,000
and split 2,000 generations ago" -- into a concrete, quantitative prediction
that can be compared to the genome sequences in your FASTA file.

.. admonition:: Biology Aside -- Why population genetics is well-suited to likelihood

   Population genetics models are **generative**: they describe the
   mechanistic process that produces genetic data. The coalescent tells us
   how lineages merge backward in time. The mutation model tells us how
   differences accumulate on branches. The recombination model tells us how
   the genealogy changes along the genome. Because these models are fully
   specified probability models, we can compute the probability of *any*
   observed dataset under *any* set of parameters. This is what makes
   likelihood-based inference possible -- and powerful. The alternative would
   be purely descriptive statistics (like :math:`F_{ST}` or Tajima's D),
   which summarize the data but cannot easily distinguish between different
   mechanistic explanations.


The Likelihood Function
========================

For a parameter vector :math:`\theta` (population sizes, divergence times,
migration rates, mutation rates, etc.), the **likelihood function** is:

.. math::

   \mathcal{L}(\theta) = P(D \mid \theta)

This reads: "the probability of observing data :math:`D` if the true
parameters were :math:`\theta`."

Crucially, the likelihood is a function of :math:`\theta`, not of :math:`D`.
The data are fixed (you've already sequenced the genomes); the parameters are
what you're trying to learn. Different parameter values give different
likelihoods, and the likelihood surface -- the landscape of
:math:`\mathcal{L}(\theta)` over all possible :math:`\theta` -- tells you
which parameters are consistent with your data and which are not.

In practice, we work with the **log-likelihood**:

.. math::

   \ell(\theta) = \log P(D \mid \theta)

because probabilities are often tiny numbers (the probability of a specific
genome sequence is astronomically small), and products of probabilities
become sums of log-probabilities, which are numerically stable.

When the data consist of :math:`n` independent observations
:math:`D = (d_1, d_2, \ldots, d_n)`, the likelihood factorizes:

.. math::

   \mathcal{L}(\theta) = \prod_{i=1}^{n} P(d_i \mid \theta)
   \qquad \Longrightarrow \qquad
   \ell(\theta) = \sum_{i=1}^{n} \log P(d_i \mid \theta)

This factorization -- turning products into sums via logarithms -- is the
reason log-likelihoods are so convenient. We will use it repeatedly throughout
this chapter and the rest of the book.

.. admonition:: Plain-language summary -- What the likelihood surface looks like

   Imagine a mountain landscape where the height at each point represents
   how well a particular demographic history explains your data. The peak of
   the tallest mountain is the maximum likelihood estimate -- the single
   best-fitting history. A broad, flat-topped mountain means many similar
   histories fit equally well (the data are not very informative about that
   parameter). A sharp peak means the data strongly constrain the parameter.
   A ridge connecting two peaks means two different models are hard to
   distinguish. The entire shape of this landscape -- not just its peak --
   contains information about what the data can and cannot tell you.

Each Timepiece computes the likelihood differently, but the logic is always
the same:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Timepiece
     - Data :math:`D`
     - Likelihood :math:`P(D \mid \theta)`
   * - **PSMC**
     - Het/hom sequence along genome
     - HMM forward algorithm (coalescent HMM)
   * - **dadi / moments / momi2**
     - Site frequency spectrum (SFS)
     - Poisson or multinomial over SFS entries
   * - **ARGweaver / SINGER**
     - Sequence alignment
     - Product over sites given a proposed ARG
   * - **tsdate**
     - Mutations on tree sequence edges
     - Poisson mutation likelihood x coalescent prior
   * - **PHLASH**
     - SFS + het/hom sequences
     - Composite: SFS likelihood + coalescent HMM

The rest of this chapter builds your fluency with the distributions that
appear in this table. By the end, you will have derived, coded, and
visualized likelihoods for the exponential, Poisson, and gamma distributions
-- the three workhorses of population genetics inference -- and you will have
seen how Bayesian inference, conjugate priors, and composite likelihoods
extend the framework.


The Toolkit: Key Distributions
================================

Before tackling any Timepiece, you need hands-on experience with the
probability distributions that generate genetic data. This section works
through each one from first principles, derives the likelihood and its MLE,
and implements everything in Python. The examples are drawn directly from
population genetics so that every formula you encounter here will reappear,
in recognizable form, in the Timepiece chapters.


The Exponential Distribution: Coalescence Waiting Times
---------------------------------------------------------

The most fundamental random variable in population genetics is the **waiting
time to coalescence** -- how many generations back in time until two lineages
find a common ancestor. Under the standard coalescent model (see
:ref:`coalescent_theory`), this waiting time is exponentially distributed.

The exponential distribution with rate parameter :math:`\lambda > 0` has
probability density:

.. math::

   f(t \mid \lambda) = \lambda \, e^{-\lambda t}, \qquad t \geq 0

Where does this formula come from? The exponential arises whenever we have a
process where "something happens at a constant rate per unit time." The
:math:`e^{-\lambda t}` is the probability of *surviving* (not coalescing) for
time :math:`t` -- it decays exponentially because at every instant there is a
constant hazard rate :math:`\lambda` of the event occurring. The leading
:math:`\lambda` converts this survival function into a density: it is the
probability of the event happening in the infinitesimal interval
:math:`[t, t + dt)`.

Its mean and variance can be computed by integration:

.. math::

   \mathbb{E}[t] = \int_0^\infty t \cdot \lambda e^{-\lambda t}\, dt
   = \frac{1}{\lambda}, \qquad
   \text{Var}(t) = \frac{1}{\lambda^2}

The mean :math:`1/\lambda` is intuitive: a higher rate means shorter waiting
times. The variance equals the mean squared, which means the standard
deviation equals the mean -- exponential waiting times are inherently noisy.

.. code-block:: python

   import numpy as np

   # Visualize the exponential density for different rates
   def exponential_pdf(t, lam):
       """Probability density of Exponential(lambda) at time t."""
       return lam * np.exp(-lam * t)

   t_grid = np.linspace(0, 5, 200)

   # Rate = 1 (coalescent units, N_e = reference size)
   density_1 = exponential_pdf(t_grid, lam=1.0)
   print(f"Rate=1.0: mean={1/1.0:.1f}, density at t=0: {density_1[0]:.2f}")

   # Rate = 0.5 (larger population, slower coalescence)
   density_05 = exponential_pdf(t_grid, lam=0.5)
   print(f"Rate=0.5: mean={1/0.5:.1f}, density at t=0: {density_05[0]:.2f}")

   # Rate = 2.0 (smaller population, faster coalescence)
   density_2 = exponential_pdf(t_grid, lam=2.0)
   print(f"Rate=2.0: mean={1/2.0:.1f}, density at t=0: {density_2[0]:.2f}")

   # Verify the mean by simulation
   np.random.seed(42)
   samples = np.random.exponential(scale=1.0/1.0, size=10000)
   print(f"\nSimulated mean (rate=1): {np.mean(samples):.3f} (expected: 1.0)")
   print(f"Simulated std:          {np.std(samples):.3f} (expected: 1.0)")

.. admonition:: Biology Aside -- Rate and population size

   In the coalescent, the rate of coalescence for a pair of lineages in a
   population of effective size :math:`N_e` is :math:`\lambda = 1/(2N_e)` per
   generation (in a diploid model). When we measure time in units of
   :math:`2N_e` generations (coalescent units), the rate becomes
   :math:`\lambda = 1`. Larger populations have smaller coalescence rates --
   it takes longer, on average, for two lineages to find a common ancestor
   in a large population than in a small one.

**The likelihood.** Suppose we observe :math:`n` independent coalescence
times :math:`t_1, t_2, \ldots, t_n`, each drawn from an exponential
distribution with unknown rate :math:`\lambda`. Because the observations are
independent, the joint probability (the likelihood) is the product of the
individual densities:

.. math::

   \mathcal{L}(\lambda) = \prod_{i=1}^{n} f(t_i \mid \lambda)
   = \prod_{i=1}^{n} \lambda \, e^{-\lambda t_i}

Taking the logarithm converts the product into a sum -- each factor
:math:`\lambda e^{-\lambda t_i}` contributes
:math:`\log\lambda + (-\lambda t_i)` to the log-likelihood:

.. math::

   \ell(\lambda) = \sum_{i=1}^{n} \log\!\big(\lambda \, e^{-\lambda t_i}\big)
   = \sum_{i=1}^{n} \big[\log \lambda - \lambda t_i\big]
   = n \log \lambda - \lambda \sum_{i=1}^{n} t_i

This is a function of the single unknown :math:`\lambda`, with the data
:math:`t_1, \ldots, t_n` baked in as constants. The two terms pull in
opposite directions: :math:`n\log\lambda` increases with :math:`\lambda`
(favoring a high rate), while :math:`-\lambda \sum t_i` decreases with
:math:`\lambda` (penalizing a rate that predicts shorter times than
observed). The MLE is where these two forces balance.

**Finding the MLE.** To find the maximum, we take the derivative with respect
to :math:`\lambda`, set it to zero, and solve. The derivative of
:math:`n\log\lambda` is :math:`n/\lambda` (using the rule
:math:`d/dx[\log x] = 1/x`), and the derivative of
:math:`-\lambda\sum t_i` is :math:`-\sum t_i`:

.. math::

   \frac{d\ell}{d\lambda} = \frac{n}{\lambda} - \sum_{i=1}^{n} t_i = 0
   \qquad \Longrightarrow \qquad
   \hat{\lambda}_{\text{MLE}} = \frac{n}{\sum_{i=1}^{n} t_i} = \frac{1}{\bar{t}}

To confirm this is a maximum (not a minimum), check the second derivative:
:math:`d^2\ell/d\lambda^2 = -n/\lambda^2 < 0`. The negative sign confirms
the log-likelihood is concave, so the critical point is indeed the peak.

The MLE of the rate is the reciprocal of the sample mean. This makes
intuitive sense: if lineages coalesce quickly (small :math:`\bar{t}`), the
rate must be high (small population); if they coalesce slowly (large
:math:`\bar{t}`), the rate must be low (large population).

.. admonition:: Biology Aside -- Estimating population size from coalescence times

   Since :math:`\lambda = 1/(2N_e)`, the MLE of the effective population
   size is :math:`\hat{N}_e = \bar{t}/2` (when time is in generations) or
   simply :math:`\hat{N}_e = \bar{t}` in coalescent units. This is the
   simplest possible demographic estimator: observe how long lineages take
   to coalesce, and infer the population size from the average. Every
   Timepiece in this book is, at its core, a more sophisticated version of
   this idea.

.. code-block:: python

   import numpy as np

   def exponential_log_likelihood(lam, times):
       """Log-likelihood of exponential rate parameter lambda.

       Parameters
       ----------
       lam : float
           Rate parameter (must be positive).
       times : array-like
           Observed waiting times.

       Returns
       -------
       ll : float
           Log-likelihood.
       """
       times = np.asarray(times)
       n = len(times)
       return n * np.log(lam) - lam * np.sum(times)

   # Simulate coalescence times for a population with N_e = 10,000
   # In coalescent units (2*N_e generations), rate = 1
   np.random.seed(42)
   true_rate = 1.0  # coalescent units
   n_pairs = 50     # observe 50 independent pairs
   coal_times = np.random.exponential(scale=1.0/true_rate, size=n_pairs)

   # MLE
   rate_mle = 1.0 / np.mean(coal_times)
   print(f"True rate:     {true_rate:.4f}")
   print(f"MLE rate:      {rate_mle:.4f}")
   print(f"Mean coal time: {np.mean(coal_times):.4f} (expected: 1.0)")

   # Scan the likelihood surface
   rates = np.linspace(0.3, 2.5, 200)
   ll_values = [exponential_log_likelihood(r, coal_times) for r in rates]
   best_idx = np.argmax(ll_values)
   print(f"Grid-search MLE: {rates[best_idx]:.4f}")

.. admonition:: Plain-language summary -- Reading a likelihood curve

   The code above computes the log-likelihood at 200 candidate rate values.
   If you plot ``ll_values`` against ``rates``, you get a curve with a single
   peak at the MLE. The width of the peak tells you how precisely the data
   constrain the rate: a narrow peak (lots of data) means the rate is
   well-determined; a broad peak (little data) means many rates are roughly
   equally plausible. Try changing ``n_pairs`` from 50 to 5 and watch the
   peak broaden.


The Poisson Distribution: Mutations and the SFS
--------------------------------------------------

Mutations accumulate on the branches of a genealogy as a **Poisson process**.
If a branch has length :math:`\ell` (in coalescent time units) and the
mutation rate is :math:`\mu` per site per generation, the expected number of
mutations on that branch is :math:`\mu \ell`. For a genomic region of
:math:`L` sites, the expected number is :math:`\mu \ell L`.

The Poisson distribution with mean :math:`\mu` has probability mass function:

.. math::

   P(k \mid \mu) = \frac{\mu^k \, e^{-\mu}}{k!}, \qquad k = 0, 1, 2, \ldots

This formula has three pieces: :math:`\mu^k` counts how likely it is to get
exactly :math:`k` events (each event contributes one factor of :math:`\mu`);
:math:`e^{-\mu}` is the probability of *no* events occurring (the "silence"
between mutations); and :math:`k!` corrects for the fact that the :math:`k`
events are interchangeable (order doesn't matter).

.. code-block:: python

   import numpy as np
   from scipy.special import factorial

   # Compute Poisson probabilities by hand
   def poisson_pmf(k, mu):
       """P(k | mu) = mu^k * exp(-mu) / k!"""
       return mu**k * np.exp(-mu) / factorial(k, exact=True)

   # Example: expected 3 mutations on a branch
   mu = 3.0
   for k in range(8):
       p = poisson_pmf(k, mu)
       print(f"  P(k={k} | mu={mu}) = {p:.4f}")

   # The most probable outcome is k=2 or k=3 (near the mean)
   # But k=0 (no mutations) still has probability ~5%

.. admonition:: Biology Aside -- Why Poisson?

   The Poisson distribution describes "the number of events in a fixed
   interval when events occur independently at a constant average rate."
   Mutations fit this description precisely: they occur independently at each
   nucleotide position, at a roughly constant rate per generation. The
   infinite-sites model (standard in population genetics) assumes that each
   mutation hits a new site, so the number of segregating sites in a genomic
   region is Poisson-distributed with a mean that depends on the total branch
   length of the genealogy.

**From one branch to the whole SFS.** The Poisson model for a single branch
extends naturally to the entire site frequency spectrum. The SFS counts, for
each frequency :math:`k` from 1 to :math:`n-1`, how many polymorphic sites
have exactly :math:`k` copies of the derived allele in a sample of :math:`n`
chromosomes.

Under the Poisson random field model, mutations land on branches of the
genealogy as independent Poisson events. A mutation on a branch that subtends
:math:`k` leaves out of :math:`n` total produces a site at frequency
:math:`k/n`. Because mutations are independent across sites, the total number
of sites at each frequency is itself Poisson-distributed:

.. math::

   D_k \sim \text{Poisson}(\xi_k(\theta))

where :math:`\xi_k(\theta)` is the expected count -- the total expected
branch length subtending exactly :math:`k` leaves, multiplied by the mutation
rate. The parameter :math:`\theta` encodes the demographic history that
determines these branch lengths.

**Deriving the SFS log-likelihood.** Because the :math:`n-1` SFS entries are
independent Poisson random variables, the joint probability of the entire
observed SFS :math:`\mathbf{D} = (D_1, \ldots, D_{n-1})` is the product of
:math:`n-1` Poisson probabilities:

.. math::

   P(\mathbf{D} \mid \theta)
   = \prod_{k=1}^{n-1} \frac{\xi_k(\theta)^{D_k} \, e^{-\xi_k(\theta)}}{D_k!}

Taking the log and expanding:

.. math::

   \ell_{\text{SFS}}(\theta)
   = \sum_{k=1}^{n-1} \Big[
     \underbrace{D_k \log \xi_k(\theta)}_{\text{data match}}
     - \underbrace{\xi_k(\theta)}_{\text{expected count}}
     - \underbrace{\log(D_k!)}_{\text{constant}}
   \Big]

The first term rewards models that predict high expected counts
:math:`\xi_k` where we observe many sites :math:`D_k`. The second term
penalizes models that predict too many total sites (overprediction). The
third term is a constant that depends only on the data and can be dropped
during optimization:

.. math::

   \ell_{\text{SFS}}(\theta) = \sum_{k=1}^{n-1}
   \big[ D_k \log \xi_k(\theta) - \xi_k(\theta) \big] + \text{const}

This is exactly the likelihood that ``dadi``, ``moments``, and ``momi2``
maximize. Here it is in Python, so you can see the formula in action:

.. code-block:: python

   import numpy as np
   from scipy.special import gammaln

   def sfs_log_likelihood(D_obs, xi_expected):
       """Poisson log-likelihood of observed SFS given expected SFS.

       Computes: sum_k [D_k * log(xi_k) - xi_k - log(D_k!)]
       """
       xi = np.maximum(xi_expected, 1e-300)  # avoid log(0)
       return np.sum(D_obs * np.log(xi) - xi - gammaln(D_obs + 1))

   # Quick example: neutral SFS with theta=100, n=10
   k = np.arange(1, 10)             # frequency classes 1..9
   xi = 100.0 / k                   # expected: xi_k = theta/k
   D = np.array([105, 48, 30, 25, 21, 17, 12, 14, 10])  # "observed"

   ll = sfs_log_likelihood(D, xi)
   print(f"SFS log-likelihood: {ll:.2f}")

Now let's build up the mathematical machinery behind this function,
starting from the simplest case.

**Single Poisson MLE.** Before tackling the full SFS, let's derive the MLE
for the simplest case: :math:`n` independent observations
:math:`k_1, \ldots, k_n`, each drawn from a Poisson with the same unknown
mean :math:`\mu`. The log-likelihood is:

.. math::

   \ell(\mu) = \sum_{i=1}^{n} \big[ k_i \log \mu - \mu - \log(k_i!) \big]

Grouping terms (and dropping the constant :math:`\sum \log(k_i!)`):

.. math::

   \ell(\mu) = \left(\sum_i k_i\right) \log \mu - n\mu + \text{const}

This has the same structure as the exponential log-likelihood: one term
that grows with :math:`\mu` (the :math:`\log\mu` term) and one that shrinks
(the :math:`-n\mu` penalty). Taking the derivative and setting to zero:

.. math::

   \frac{d\ell}{d\mu} = \frac{\sum_i k_i}{\mu} - n = 0
   \qquad \Longrightarrow \qquad
   \hat{\mu}_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^{n} k_i = \bar{k}

The MLE of the Poisson mean is the sample mean -- again, an intuitive result.
The second derivative :math:`d^2\ell/d\mu^2 = -\sum k_i / \mu^2 < 0`
confirms this is a maximum.

.. code-block:: python

   # Verify the Poisson MLE
   np.random.seed(42)
   mu_true = 7.0
   observations = np.random.poisson(mu_true, size=50)
   mu_mle = np.mean(observations)
   print(f"True mu:  {mu_true}")
   print(f"MLE mu:   {mu_mle:.2f} (= sample mean of {len(observations)} obs)")

**Connecting the Poisson MLE to the SFS.** For the SFS, the situation is
slightly different: each entry :math:`D_k` is Poisson with its *own* mean
:math:`\xi_k(\theta)`, and these means are linked through the demographic
model. We can't just take the sample mean -- instead, we search over
:math:`\theta` to find the demographic history that makes all the
:math:`\xi_k(\theta)` values simultaneously consistent with the observed
:math:`D_k`.

.. code-block:: python

   import numpy as np
   from scipy.special import gammaln  # log(k!) = gammaln(k+1)

   def poisson_log_likelihood(mu, counts):
       """Poisson log-likelihood for a vector of observed counts.

       Parameters
       ----------
       mu : float or ndarray
           Expected count(s). If scalar, same mean for all observations.
           If array, must match shape of counts.
       counts : ndarray
           Observed counts.

       Returns
       -------
       ll : float
           Log-likelihood.
       """
       counts = np.asarray(counts, dtype=float)
       mu = np.asarray(mu, dtype=float)
       # Full Poisson log-likelihood: k*log(mu) - mu - log(k!)
       return np.sum(counts * np.log(np.maximum(mu, 1e-300)) - mu
                     - gammaln(counts + 1))

   # ---------------------------------------------------------------
   # Example: SFS under the standard neutral model
   # Under constant population size, the expected SFS is xi_k = theta/k
   # (Watterson's result). Here theta = 4*N_e*mu*L.
   # ---------------------------------------------------------------
   n_samples = 20       # number of haploid chromosomes
   theta_true = 200.0   # true population-scaled mutation rate

   # Expected SFS: xi_k = theta / k for k = 1, ..., n-1
   k_values = np.arange(1, n_samples)
   xi_expected = theta_true / k_values

   # Simulate observed SFS by Poisson sampling
   np.random.seed(42)
   D_observed = np.random.poisson(xi_expected)

   print("Frequency k:  ", k_values[:8])
   print("Expected xi:  ", np.round(xi_expected[:8], 1))
   print("Observed D:   ", D_observed[:8])

   # Compute SFS log-likelihood at the true theta
   ll_true = poisson_log_likelihood(xi_expected, D_observed)
   print(f"\nLog-likelihood at true theta={theta_true}: {ll_true:.2f}")

   # Scan over candidate theta values
   thetas = np.linspace(100, 350, 300)
   ll_scan = []
   for th in thetas:
       xi_candidate = th / k_values
       ll_scan.append(poisson_log_likelihood(xi_candidate, D_observed))
   ll_scan = np.array(ll_scan)

   theta_mle = thetas[np.argmax(ll_scan)]
   print(f"MLE theta (grid search): {theta_mle:.1f}")

   # Analytical MLE: total observed mutations / sum(1/k)
   harmonic = np.sum(1.0 / k_values)
   S = np.sum(D_observed)  # total segregating sites
   theta_mle_exact = S / harmonic
   print(f"MLE theta (analytical):  {theta_mle_exact:.1f}")
   print(f"This is Watterson's estimator!")

.. admonition:: Biology Aside -- Watterson's estimator

   The analytical MLE of :math:`\theta` from the SFS under the neutral model
   is :math:`\hat{\theta}_W = S / \sum_{k=1}^{n-1} 1/k`, where :math:`S`
   is the total number of segregating sites. This is **Watterson's
   estimator** -- one of the most famous results in population genetics,
   derived in 1975. It is the starting point for neutrality tests like
   Tajima's D, which compares Watterson's estimator (based on the total
   number of segregating sites) with a frequency-weighted estimator (based on
   the average pairwise differences). When the two disagree, it signals a
   departure from the standard neutral model -- for example, a recent
   population expansion or a selective sweep.


The Gamma Distribution: Ages and Rates
-----------------------------------------

The **gamma distribution** appears throughout population genetics as a
flexible model for positive continuous quantities -- most prominently as the
prior distribution on node ages in ``tsdate``.

A gamma random variable with shape :math:`\alpha > 0` and rate
:math:`\beta > 0` has density:

.. math::

   f(t \mid \alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)}
   \, t^{\alpha - 1} \, e^{-\beta t}, \qquad t > 0

where :math:`\Gamma(\alpha) = \int_0^\infty x^{\alpha-1} e^{-x} dx` is the
gamma function. The mean is :math:`\alpha/\beta` and the variance is
:math:`\alpha/\beta^2`.

.. admonition:: Plain-language summary -- What shape and rate control

   The **shape** :math:`\alpha` controls the form of the distribution:

   - :math:`\alpha = 1`: the gamma reduces to an exponential (our coalescence
     time distribution).
   - :math:`\alpha < 1`: the density spikes near zero and has a heavy tail --
     useful for modeling quantities that are often very small but occasionally
     very large.
   - :math:`\alpha > 1`: the density has a peak away from zero, forming a
     bell-shaped curve skewed to the right.

   The **rate** :math:`\beta` controls the scale: larger :math:`\beta`
   compresses the distribution toward zero (shorter times), smaller
   :math:`\beta` stretches it out (longer times).

   In ``tsdate``, the gamma distribution is used to approximate the posterior
   distribution of each node's age. The shape and rate are updated iteratively
   by Expectation Propagation as the algorithm absorbs information from
   mutations and the coalescent prior.

The exponential distribution is a special case: :math:`\text{Gamma}(1, \lambda)
= \text{Exponential}(\lambda)`. This means the exponential likelihood we
derived above is a special case of the gamma likelihood with known
:math:`\alpha = 1`.

**Deriving the gamma log-likelihood.** For :math:`n` independent observations
:math:`t_1, \ldots, t_n` from a :math:`\text{Gamma}(\alpha, \beta)`, the
joint density is:

.. math::

   \mathcal{L}(\alpha, \beta) = \prod_{i=1}^{n}
   \frac{\beta^\alpha}{\Gamma(\alpha)} \, t_i^{\alpha-1} \, e^{-\beta t_i}

Taking the logarithm, each factor contributes
:math:`\alpha\log\beta - \log\Gamma(\alpha) + (\alpha-1)\log t_i - \beta t_i`:

.. math::

   \ell(\alpha, \beta) = n \alpha \log \beta - n \log \Gamma(\alpha)
   + (\alpha - 1) \sum_{i=1}^{n} \log t_i - \beta \sum_{i=1}^{n} t_i

The four terms have clear roles: :math:`n\alpha\log\beta` and
:math:`-n\log\Gamma(\alpha)` are normalization terms that depend on the
parameters but not on individual data points; :math:`(\alpha-1)\sum\log t_i`
measures how the data agree with the shape parameter (larger :math:`\alpha`
favors larger values of :math:`t`); and :math:`-\beta\sum t_i` penalizes
large rate parameters when the data contain large values.

**Partial MLEs.** To find the MLE of :math:`\beta` for a given
:math:`\alpha`, take the partial derivative with respect to :math:`\beta`:

.. math::

   \frac{\partial \ell}{\partial \beta} = \frac{n\alpha}{\beta} - \sum t_i = 0
   \qquad \Longrightarrow \qquad
   \hat{\beta}(\alpha) = \frac{n\alpha}{\sum t_i}

This is analogous to the exponential MLE (which is the special case
:math:`\alpha = 1`). However, there is no closed-form MLE for :math:`\alpha`
itself -- the derivative :math:`\partial\ell/\partial\alpha` involves the
**digamma function** :math:`\psi(\alpha) = d\log\Gamma(\alpha)/d\alpha`,
giving the equation :math:`\log\hat{\beta} - \psi(\alpha) =
\frac{1}{n}\sum\log t_i` which must be solved numerically. The code below
uses profile likelihood: for each candidate :math:`\alpha`, plug in the
optimal :math:`\hat{\beta}(\alpha)` and maximize over :math:`\alpha` alone.

.. code-block:: python

   import numpy as np
   from scipy.special import gammaln, digamma
   from scipy.optimize import minimize_scalar

   def gamma_log_likelihood(alpha, beta, data):
       """Log-likelihood for Gamma(alpha, beta) observations.

       Parameters
       ----------
       alpha : float
           Shape parameter.
       beta : float
           Rate parameter.
       data : ndarray
           Observed positive values.

       Returns
       -------
       ll : float
           Log-likelihood.
       """
       data = np.asarray(data)
       n = len(data)
       return (n * alpha * np.log(beta)
               - n * gammaln(alpha)
               + (alpha - 1) * np.sum(np.log(data))
               - beta * np.sum(data))

   def gamma_mle(data):
       """Compute the MLE of Gamma(alpha, beta) by profiling.

       For fixed alpha, the MLE of beta is n*alpha / sum(data).
       We optimize over alpha using this profile likelihood.
       """
       data = np.asarray(data)
       n = len(data)
       sum_data = np.sum(data)
       sum_log_data = np.sum(np.log(data))

       def neg_profile_ll(alpha):
           beta_hat = n * alpha / sum_data
           return -(n * alpha * np.log(beta_hat)
                    - n * gammaln(alpha)
                    + (alpha - 1) * sum_log_data
                    - beta_hat * sum_data)

       result = minimize_scalar(neg_profile_ll, bounds=(0.01, 100),
                                method='bounded')
       alpha_hat = result.x
       beta_hat = n * alpha_hat / sum_data
       return alpha_hat, beta_hat

   # Simulate node ages from a gamma prior (as in tsdate)
   np.random.seed(42)
   true_alpha, true_beta = 3.0, 0.5  # mean = 6.0, variance = 12.0
   node_ages = np.random.gamma(shape=true_alpha, scale=1.0/true_beta, size=100)

   alpha_hat, beta_hat = gamma_mle(node_ages)
   print(f"True:  alpha={true_alpha}, beta={true_beta}, mean={true_alpha/true_beta}")
   print(f"MLE:   alpha={alpha_hat:.3f}, beta={beta_hat:.3f}, "
         f"mean={alpha_hat/beta_hat:.3f}")
   print(f"Sample mean: {np.mean(node_ages):.3f}")

.. admonition:: Biology Aside -- Why gamma for node ages?

   The age of an internal node in a genealogy is the sum of exponential
   waiting times (one for each coalescent interval the node passes through).
   A sum of exponential random variables follows a gamma distribution. Even
   when the population size varies over time (making the rates differ), the
   gamma remains a good approximation because it is flexible enough to
   capture a wide range of shapes. This is why ``tsdate`` uses the gamma
   family as its variational approximation: it is mathematically convenient
   (as we'll see when we discuss conjugate priors below) and biologically
   motivated.


The Gaussian Distribution: Smoothness Priors
-----------------------------------------------

The **Gaussian** (normal) distribution appears in population genetics
inference not as a data model, but as a **prior** -- specifically, a
smoothness prior on demographic histories.

The Gaussian density for a scalar :math:`x` with mean :math:`\mu` and
variance :math:`\sigma^2` is:

.. math::

   f(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}
   \exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)

For a vector :math:`\mathbf{x}` of dimension :math:`d`, the multivariate
Gaussian with mean :math:`\boldsymbol{\mu}` and covariance matrix
:math:`\Sigma` has density:

.. math::

   f(\mathbf{x}) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}}
   \exp\!\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top
   \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)

.. admonition:: Biology Aside -- Smoothness in demographic history

   PHLASH represents the demographic history as a vector of log-population
   sizes :math:`\mathbf{h} = (\log N_1, \log N_2, \ldots, \log N_M)` at
   :math:`M` time points. It places a multivariate Gaussian prior on
   :math:`\mathbf{h}` with a covariance matrix that encodes smoothness:
   adjacent time points are correlated, so the population size cannot jump
   wildly from one epoch to the next without incurring a penalty.

   Concretely, the prior penalizes large *differences* between adjacent
   entries:

   .. math::

      \log p(\mathbf{h}) \propto -\frac{1}{2\sigma^2}
      \sum_{k=1}^{M-1} (h_{k+1} - h_k)^2

   This is a random-walk prior: it says "I expect the demographic history to
   change gradually." The hyperparameter :math:`\sigma^2` controls how much
   change is allowed per time step -- smaller :math:`\sigma^2` enforces
   smoother histories, larger :math:`\sigma^2` allows more variation.

**Gaussian MLE.** For :math:`n` observations :math:`x_1, \ldots, x_n` from
:math:`N(\mu, \sigma^2)`, the log-likelihood is:

.. math::

   \ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi\sigma^2)
   - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2

The first term is a normalization penalty that grows with :math:`\sigma^2`
(wider distributions spread probability thinner). The second term measures
how well :math:`\mu` fits the data -- it is minimized when :math:`\mu` is at
the center of the data.

Taking the derivative with respect to :math:`\mu` (with :math:`\sigma^2`
held fixed):

.. math::

   \frac{\partial \ell}{\partial \mu} =
   \frac{1}{\sigma^2}\sum_{i=1}^{n}(x_i - \mu) = 0
   \qquad \Longrightarrow \qquad
   \hat{\mu} = \frac{1}{n}\sum_{i=1}^{n} x_i = \bar{x}

And with respect to :math:`\sigma^2` (with :math:`\mu = \hat{\mu}`):

.. math::

   \frac{\partial \ell}{\partial \sigma^2} =
   -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_{i=1}^{n}(x_i - \bar{x})^2 = 0
   \qquad \Longrightarrow \qquad
   \hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2

These are the sample mean and (biased) sample variance -- the same
"differentiate, set to zero, solve" recipe we've used throughout.

.. code-block:: python

   import numpy as np

   def gaussian_log_likelihood(mu, sigma2, data):
       """Log-likelihood for N(mu, sigma^2) observations."""
       data = np.asarray(data)
       n = len(data)
       return (-0.5 * n * np.log(2 * np.pi * sigma2)
               - 0.5 * np.sum((data - mu)**2) / sigma2)

   def smoothness_prior_log_density(h, sigma=1.0):
       """Log-density of the Gaussian random-walk smoothness prior.

       Penalizes large jumps between adjacent log-population sizes.

       Parameters
       ----------
       h : ndarray, shape (M,)
           Log-population sizes at M time points.
       sigma : float
           Smoothness scale (standard deviation of allowed changes).

       Returns
       -------
       lp : float
           Log prior density (up to a constant).
       """
       diffs = np.diff(h)
       return -0.5 * np.sum(diffs**2) / sigma**2

   # Demonstrate: smooth vs. rough demographic histories
   M = 30  # time points
   h_smooth = np.zeros(M)
   h_smooth[10:20] = -0.5  # gentle bottleneck

   h_rough = np.zeros(M)
   h_rough[::2] = 1.0  # wild oscillations
   h_rough[1::2] = -1.0

   sigma = 0.5
   lp_smooth = smoothness_prior_log_density(h_smooth, sigma)
   lp_rough = smoothness_prior_log_density(h_rough, sigma)
   print(f"Smooth history prior: {lp_smooth:.2f}")
   print(f"Rough history prior:  {lp_rough:.2f}")
   print(f"The smooth history is exp({lp_smooth - lp_rough:.0f}) times "
         f"more probable under the prior")


Maximum Likelihood Estimation (MLE)
=====================================

The previous section showed the MLE recipe for individual distributions. Here
we formalize it and show how it works for a more realistic problem.

The **maximum likelihood estimate** (MLE) is the parameter value that
maximizes the likelihood:

.. math::

   \hat{\theta}_{\text{MLE}} = \arg\max_\theta \; P(D \mid \theta)

Equivalently, it maximizes the log-likelihood (since :math:`\log` is
monotonically increasing). The recipe is:

1. Write down the log-likelihood :math:`\ell(\theta)`.
2. Take the derivative :math:`d\ell/d\theta` (or gradient, for vector
   :math:`\theta`).
3. Set the derivative to zero and solve for :math:`\theta`.
4. Verify that the solution is a maximum (the second derivative is negative).

When step 3 has a closed-form solution (as for the exponential, Poisson, and
Gaussian), the MLE is called **analytically tractable**. When it doesn't (as
for the gamma shape parameter, or for complex demographic models), we use
numerical optimization -- gradient ascent, Newton's method, or specialized
algorithms like ``scipy.optimize.minimize``.

The MLE has attractive statistical properties:

- **Consistency**: as the amount of data grows, the MLE converges to the true
  parameter values.
- **Efficiency**: among all consistent estimators, the MLE achieves the
  smallest possible variance (asymptotically).
- **Invariance**: if :math:`\hat{\theta}` is the MLE of :math:`\theta`, then
  :math:`f(\hat{\theta})` is the MLE of :math:`f(\theta)` for any function
  :math:`f`.

In practice, ``dadi``, ``moments``, and ``momi2`` all use MLE as their
primary inference strategy: they search the space of demographic parameters
for the values that maximize the Poisson log-likelihood of the observed SFS.

.. admonition:: Practical caveat -- Local optima

   The likelihood surface for complex demographic models is rarely convex.
   It can have multiple peaks (local optima), ridges, and flat regions. A
   gradient-based optimizer starting from a single initial point may find a
   local optimum rather than the global one. This is why ``momi2`` and
   ``dadi`` recommend running the optimizer from multiple random starting
   points and keeping the best result. It is also why simpler models (fewer
   parameters) are often preferred: they have smoother likelihood surfaces
   with fewer local optima.


Worked Example: Inferring Population Size from the SFS
--------------------------------------------------------

Here is a complete MLE example that mirrors what ``dadi`` and ``moments`` do
internally: given an observed SFS, find the population size ratio :math:`\nu`
that maximizes the Poisson log-likelihood.

.. code-block:: python

   import numpy as np
   from scipy.optimize import minimize_scalar
   from scipy.special import gammaln

   def expected_sfs_two_epoch(n, theta, nu, T):
       """Expected SFS for a two-epoch model (approximate).

       Population was size N_ref in the past, then changed to nu*N_ref
       at time T (in coalescent units) before present.

       This uses the neutral expectation xi_k = theta/k as a baseline,
       then scales by a correction factor that depends on nu and T.
       (A full calculation would integrate the coalescent with variable
       population size; here we use the simple approximation for
       pedagogical clarity.)

       Parameters
       ----------
       n : int
           Sample size (number of haploid chromosomes).
       theta : float
           Baseline population-scaled mutation rate (4*N_ref*mu*L).
       nu : float
           Ratio of current to ancestral population size.
       T : float
           Time of size change in coalescent units.

       Returns
       -------
       xi : ndarray, shape (n-1,)
           Expected SFS entries for k = 1, ..., n-1.
       """
       k = np.arange(1, n)
       # Under constant size: xi_k = theta / k
       # Size change modifies the total branch length.
       # For a simple two-epoch model, the expected total coalescence
       # time is approximately T + nu*(total_time - T) when T is small
       # relative to total coalescent time.
       # Here we use the exact 1-population result for the k-th entry:
       # larger nu -> more singletons (population expansion signature)
       # smaller nu -> fewer singletons (bottleneck signature)
       base = theta / k
       # Correction: expansion (nu > 1) inflates low-frequency classes
       correction = 1.0 + (nu - 1.0) * (1.0 - np.exp(-T * k / nu))
       return base * np.maximum(correction, 0.01)

   def sfs_poisson_loglik(D_obs, xi_expected):
       """Poisson log-likelihood of observed SFS given expected."""
       xi = np.maximum(xi_expected, 1e-300)
       return np.sum(D_obs * np.log(xi) - xi - gammaln(D_obs + 1))

   # Generate data: true model has nu=3.0 (threefold expansion), T=0.2
   np.random.seed(42)
   n = 30
   theta = 500.0
   nu_true = 3.0
   T_true = 0.2

   xi_true = expected_sfs_two_epoch(n, theta, nu_true, T_true)
   D_obs = np.random.poisson(xi_true)

   # Scan nu values and compute log-likelihood for each
   nus = np.linspace(0.5, 8.0, 200)
   lls = [sfs_poisson_loglik(D_obs, expected_sfs_two_epoch(n, theta, nu, T_true))
          for nu in nus]
   lls = np.array(lls)

   nu_mle = nus[np.argmax(lls)]
   print(f"True nu:  {nu_true:.2f}")
   print(f"MLE nu:   {nu_mle:.2f}")
   print(f"Log-likelihood at MLE:  {np.max(lls):.2f}")
   print(f"Log-likelihood at true: {sfs_poisson_loglik(D_obs, xi_true):.2f}")


Fisher Information and Confidence Intervals
---------------------------------------------

The MLE tells you the *best* parameter value, but not how confident you
should be in it. The **Fisher information** quantifies the precision of
the MLE.

For a single parameter :math:`\theta`, the Fisher information is:

.. math::

   I(\theta) = -\mathbb{E}\!\left[\frac{d^2 \ell}{d\theta^2}\right]

What does this mean? The second derivative :math:`d^2\ell/d\theta^2` measures
the **curvature** of the log-likelihood curve. At the MLE (the peak), the
curvature is always negative (the peak bends downward). The more negative it
is, the sharper the peak, and the more precisely the data pin down
:math:`\theta`. The expectation :math:`\mathbb{E}[\cdot]` averages this
curvature over all possible datasets -- it measures the curvature we'd
*expect* to see. The negative sign flips this to a positive number, giving
us the **Fisher information**: a measure of how much information the data
carry about :math:`\theta`.

For :math:`n` independent observations, the Fisher information scales
linearly: :math:`I_n(\theta) = n \cdot I_1(\theta)`, where :math:`I_1` is
the information from a single observation. More data means more information.

The key result (proven in statistical theory): for large samples, the MLE is
approximately normally distributed with variance :math:`1/I(\theta)`:

.. math::

   \hat{\theta}_{\text{MLE}} \;\dot{\sim}\; N\!\left(\theta, \;
   \frac{1}{I(\theta)}\right)

This connects the curvature of the likelihood to the uncertainty in the
estimate: sharp peak (large :math:`I`) means small variance (precise MLE);
flat peak (small :math:`I`) means large variance (imprecise MLE).

This gives an approximate **95% confidence interval**:

.. math::

   \hat{\theta} \pm 1.96 \cdot \frac{1}{\sqrt{I(\hat{\theta})}}

where 1.96 is the :math:`z`-value for 95% coverage under the normal
distribution, and :math:`1/\sqrt{I(\hat{\theta})}` is the **standard error**
of the MLE.

.. admonition:: Plain-language summary -- Fisher information as a ruler

   The Fisher information measures how "informative" the data are about the
   parameter. If you double the number of observations, you roughly double
   the Fisher information and halve the variance of the MLE -- meaning the
   confidence interval shrinks by a factor of :math:`\sqrt{2}`. This is why
   more data gives more precise estimates. It's also why some parameters are
   harder to estimate than others: if the likelihood surface is flat in some
   direction (low Fisher information), even a large dataset won't pin down
   that parameter precisely.

**Example: Fisher information for the exponential.** Let's work through the
full calculation. We showed that the log-likelihood is
:math:`\ell(\lambda) = n\log\lambda - \lambda\sum t_i`. We already computed
the first derivative: :math:`d\ell/d\lambda = n/\lambda - \sum t_i`.
Differentiating again:

.. math::

   \frac{d^2 \ell}{d\lambda^2} = -\frac{n}{\lambda^2}

This is already non-random (it doesn't depend on the data :math:`t_i`), so
the expectation is just itself: :math:`I(\lambda) = n/\lambda^2`. The
variance of the MLE is :math:`1/I(\lambda) = \lambda^2/n`, and the standard
error is :math:`\lambda/\sqrt{n}`.

The 95% CI is :math:`\hat{\lambda} \pm 1.96 \cdot \hat{\lambda}/\sqrt{n}`.
Notice that the CI width is proportional to :math:`1/\sqrt{n}` -- quadrupling
the data cuts the CI width in half.

.. code-block:: python

   import numpy as np

   # Confidence interval for the exponential rate parameter
   np.random.seed(42)
   true_rate = 1.0
   n_obs = 50
   times = np.random.exponential(scale=1.0/true_rate, size=n_obs)

   rate_mle = 1.0 / np.mean(times)
   fisher_info = n_obs / rate_mle**2
   se = 1.0 / np.sqrt(fisher_info)  # standard error
   ci_low = rate_mle - 1.96 * se
   ci_high = rate_mle + 1.96 * se

   print(f"True rate:       {true_rate:.4f}")
   print(f"MLE rate:        {rate_mle:.4f}")
   print(f"Fisher info:     {fisher_info:.2f}")
   print(f"Standard error:  {se:.4f}")
   print(f"95% CI:          [{ci_low:.4f}, {ci_high:.4f}]")
   print(f"True rate in CI: {ci_low <= true_rate <= ci_high}")

   # Demonstrate: more data -> tighter CI
   for n in [10, 50, 200, 1000]:
       t = np.random.exponential(scale=1.0/true_rate, size=n)
       lam_hat = 1.0 / np.mean(t)
       se_n = lam_hat / np.sqrt(n)
       print(f"  n={n:4d}:  MLE={lam_hat:.3f}, "
             f"CI width={2*1.96*se_n:.3f}")


Bayesian Inference
===================

Maximum likelihood gives a single best answer. **Bayesian inference** goes
further: it characterizes the entire range of plausible parameter values by
computing the **posterior distribution**:

.. math::

   P(\theta \mid D) = \frac{P(D \mid \theta) \, P(\theta)}{P(D)}

This equation (Bayes' theorem) has three components:

- **Likelihood** :math:`P(D \mid \theta)`: how well the parameters explain
  the data (same as above).
- **Prior** :math:`P(\theta)`: what we believed about the parameters *before*
  seeing the data. This encodes biological knowledge or constraints -- for
  example, population sizes must be positive, or the demographic history
  should be smooth.
- **Posterior** :math:`P(\theta \mid D)`: what we believe about the parameters
  *after* seeing the data. This is the target of inference.

The denominator :math:`P(D) = \int P(D \mid \theta) P(\theta) d\theta` (the
**marginal likelihood** or **evidence**) is a normalizing constant that ensures
the posterior integrates to 1. It is typically intractable to compute directly,
which is why methods like MCMC (see :ref:`mcmc`) and SVGD
(see :ref:`phlash_svgd`) are needed.

.. admonition:: Plain-language summary -- Prior, likelihood, posterior

   Think of it as a conversation between two sources of information:

   - The **prior** is your background knowledge: "population sizes are
     typically between 1,000 and 1,000,000" or "the demographic history
     should be smooth, not wildly oscillating."
   - The **likelihood** is what the data say: "these particular allele
     frequencies are 100 times more probable under a bottleneck model than
     under constant size."
   - The **posterior** is the synthesis: "given both my background knowledge
     and the data, the population most likely experienced a bottleneck of
     ~5,000 individuals, with a 95% credible interval of 2,000--15,000."

   The posterior gives you not just a point estimate but a full picture of
   uncertainty. This is why Bayesian methods (ARGweaver, SINGER, PHLASH) can
   provide credible intervals and uncertainty bands, while pure MLE methods
   (dadi, moments) require additional steps like bootstrapping to quantify
   uncertainty.


Conjugate Priors: When Bayesian Inference Has Closed-Form Solutions
---------------------------------------------------------------------

In general, computing the posterior requires numerical methods (MCMC, EP,
SVGD). But for certain combinations of likelihood and prior, the posterior
has the **same functional form** as the prior. These are called **conjugate
priors**, and they give exact, closed-form Bayesian updates.

Conjugate priors are not just a mathematical curiosity -- they are the engine
behind ``tsdate``'s Expectation Propagation algorithm, and they appear in
simplified form in several other Timepieces.

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Likelihood
     - Conjugate prior
     - Posterior
     - Used in
   * - Poisson(:math:`\lambda`)
     - Gamma(:math:`\alpha, \beta`)
     - Gamma(:math:`\alpha', \beta'`)
     - **tsdate**
   * - Binomial(:math:`n, p`)
     - Beta(:math:`a, b`)
     - Beta(:math:`a', b'`)
     - *(general)*
   * - Normal(:math:`\mu, \sigma^2`)
     - Normal(:math:`\mu_0, \sigma_0^2`)
     - Normal(:math:`\mu', \sigma'^2`)
     - **PHLASH** (approx.)

The Gamma-Poisson Conjugacy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is the most important conjugacy for this book, because it is the
foundation of ``tsdate``'s variational gamma algorithm.

**Setup.** Suppose the number of mutations :math:`k` on a branch of length
:math:`t` follows a Poisson distribution with rate :math:`\mu t`:

.. math::

   P(k \mid t) = \frac{(\mu t)^k \, e^{-\mu t}}{k!}

and we place a gamma prior on :math:`t`:

.. math::

   p(t) = \frac{\beta^\alpha}{\Gamma(\alpha)} \, t^{\alpha - 1} \, e^{-\beta t}

**Deriving the posterior.** By Bayes' theorem:

.. math::

   p(t \mid k) &\propto P(k \mid t) \, p(t) \\
   &\propto (\mu t)^k e^{-\mu t} \cdot t^{\alpha-1} e^{-\beta t} \\
   &= \mu^k \cdot t^{k + \alpha - 1} \cdot e^{-(\mu + \beta) t}

Dropping constants that don't depend on :math:`t`, we recognize this as
the kernel of a gamma distribution with updated parameters:

.. math::

   t \mid k \;\sim\; \text{Gamma}(\alpha + k, \;\beta + \mu)

This is the Bayesian update rule:

- **Shape update**: :math:`\alpha' = \alpha + k` (each observed mutation
  increments the shape by 1).
- **Rate update**: :math:`\beta' = \beta + \mu` (the mutation rate adds to
  the rate parameter).

.. admonition:: Biology Aside -- What the Bayesian update means for node dating

   Consider a branch in a genealogy with 3 observed mutations and a
   per-branch mutation rate of :math:`\mu = 0.5` per coalescent unit. If our
   prior on the branch length is :math:`\text{Gamma}(2, 1)` (mean 2, moderate
   uncertainty), then after observing the 3 mutations, the posterior is
   :math:`\text{Gamma}(2 + 3, 1 + 0.5) = \text{Gamma}(5, 1.5)`.

   The posterior mean shifts from :math:`2/1 = 2.0` to :math:`5/1.5 = 3.33`
   -- the data (3 mutations suggesting a longer branch) have pulled the
   estimate upward from the prior. If we had observed 0 mutations instead,
   the posterior would be :math:`\text{Gamma}(2, 1.5)` with mean
   :math:`2/1.5 = 1.33` -- the data (no mutations suggesting a shorter
   branch) have pulled the estimate downward.

   This is exactly what ``tsdate`` does at every edge in the tree sequence:
   it starts with a coalescent prior, observes the mutation data, and
   updates to a gamma posterior. The Expectation Propagation algorithm
   (see :ref:`tsdate_variational_gamma`) iterates these updates across the
   tree until the node age estimates are globally consistent.

.. code-block:: python

   import numpy as np
   from scipy.stats import gamma as gamma_dist

   def gamma_poisson_update(alpha_prior, beta_prior, k_observed, mu):
       """Bayesian update: Gamma prior + Poisson likelihood -> Gamma posterior.

       Parameters
       ----------
       alpha_prior : float
           Prior shape parameter.
       beta_prior : float
           Prior rate parameter.
       k_observed : int
           Number of observed mutations.
       mu : float
           Per-branch mutation rate.

       Returns
       -------
       alpha_post : float
           Posterior shape.
       beta_post : float
           Posterior rate.
       """
       return alpha_prior + k_observed, beta_prior + mu

   # Demonstrate the Bayesian update for branch length estimation
   alpha_prior, beta_prior = 2.0, 1.0
   mu = 0.5

   print("Prior: Gamma({}, {})  ->  mean={:.2f}, var={:.2f}".format(
       alpha_prior, beta_prior,
       alpha_prior/beta_prior, alpha_prior/beta_prior**2))

   for k in [0, 1, 3, 10]:
       a_post, b_post = gamma_poisson_update(alpha_prior, beta_prior, k, mu)
       mean_post = a_post / b_post
       var_post = a_post / b_post**2
       print(f"  k={k:2d} mutations -> Gamma({a_post:.1f}, {b_post:.1f}), "
             f"mean={mean_post:.2f}, var={var_post:.2f}")

   # Show how the posterior concentrates with more data
   print("\nMultiple branches, each with mu=0.5:")
   alpha, beta = 2.0, 1.0  # start from prior
   for i, k in enumerate([2, 1, 3, 0, 4, 2, 1, 3, 2, 2]):
       alpha, beta = gamma_poisson_update(alpha, beta, k, mu)
       print(f"  After branch {i+1} (k={k}): Gamma({alpha:.1f}, {beta:.1f}), "
             f"mean={alpha/beta:.3f}, CI=["
             f"{gamma_dist.ppf(0.025, alpha, scale=1/beta):.3f}, "
             f"{gamma_dist.ppf(0.975, alpha, scale=1/beta):.3f}]")


The Beta-Binomial Conjugacy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This conjugacy is less central to this book but appears frequently in
population genetics and is instructive.

**Setup.** Observe :math:`k` successes in :math:`n` trials (e.g., :math:`k`
copies of the derived allele in a sample of :math:`n` chromosomes):

.. math::

   P(k \mid p) = \binom{n}{k} p^k (1-p)^{n-k}

with a Beta prior on the success probability :math:`p`:

.. math::

   p(p) = \frac{p^{a-1}(1-p)^{b-1}}{B(a,b)}

where :math:`B(a,b)` is the Beta function (a normalizing constant).

**Deriving the posterior.** Applying Bayes' theorem, the posterior is
proportional to the product of likelihood and prior:

.. math::

   p(p \mid k) &\propto P(k \mid p) \cdot p(p) \\
   &\propto p^k (1-p)^{n-k} \cdot p^{a-1}(1-p)^{b-1} \\
   &= p^{(a+k)-1} \cdot (1-p)^{(b+n-k)-1}

We collected the powers of :math:`p` and :math:`(1-p)` by adding exponents.
The result is the kernel of a Beta distribution:

.. math::

   p \mid k \;\sim\; \text{Beta}(a + k, \; b + n - k)

The update rule is: add the observed successes to :math:`a`, and add the
observed failures to :math:`b`. The prior "pseudo-counts" :math:`a` and
:math:`b` act as if we had already seen :math:`a-1` successes and
:math:`b-1` failures before collecting any data.

.. code-block:: python

   import numpy as np
   from scipy.stats import beta as beta_dist

   def beta_binomial_update(a_prior, b_prior, k, n):
       """Bayesian update: Beta prior + Binomial likelihood -> Beta posterior."""
       return a_prior + k, b_prior + n - k

   # Example: estimate allele frequency from a sample
   # Prior: Beta(1, 1) = Uniform(0, 1) -- no prior information
   a_prior, b_prior = 1.0, 1.0

   # Observe 7 derived alleles out of 20 chromosomes
   k_obs, n_obs = 7, 20
   a_post, b_post = beta_binomial_update(a_prior, b_prior, k_obs, n_obs)

   mean_post = a_post / (a_post + b_post)
   ci = beta_dist.ppf([0.025, 0.975], a_post, b_post)

   print(f"Observed: {k_obs}/{n_obs} derived alleles")
   print(f"MLE frequency: {k_obs/n_obs:.3f}")
   print(f"Posterior mean: {mean_post:.3f}")
   print(f"95% credible interval: [{ci[0]:.3f}, {ci[1]:.3f}]")

   # With a stronger prior: "I expect low-frequency variants"
   # Beta(1, 10) has mean 0.09
   a_prior2, b_prior2 = 1.0, 10.0
   a_post2, b_post2 = beta_binomial_update(a_prior2, b_prior2, k_obs, n_obs)
   mean_post2 = a_post2 / (a_post2 + b_post2)
   ci2 = beta_dist.ppf([0.025, 0.975], a_post2, b_post2)
   print(f"\nWith informative prior Beta(1,10):")
   print(f"Posterior mean: {mean_post2:.3f}")
   print(f"95% credible interval: [{ci2[0]:.3f}, {ci2[1]:.3f}]")
   print("(The informative prior pulls the estimate toward lower frequencies)")

.. admonition:: Plain-language summary -- Why conjugate priors matter

   Conjugate priors are special because they keep the math tractable. When
   the prior and posterior belong to the same family, the Bayesian update
   reduces to updating two numbers (the parameters of the distribution)
   rather than computing an integral over all possible parameter values.
   This is computationally free -- no MCMC needed, no approximation
   required.

   In ``tsdate``, the gamma-Poisson conjugacy is what makes Expectation
   Propagation feasible: at each step, the algorithm absorbs the information
   from one mutation observation by simply updating :math:`\alpha` and
   :math:`\beta`. Thousands of these updates can be performed in
   milliseconds. Without conjugacy, each update would require an expensive
   numerical integration.


Bayesian inference in this book appears in:

- **ARGweaver**: MCMC sampling over ARGs, with a coalescent prior on the
  genealogy
- **SINGER**: MCMC sampling with data-informed proposals (SGPR)
- **tsdate**: Expectation Propagation with a coalescent prior on node ages
  (gamma-Poisson conjugacy)
- **PHLASH**: SVGD with a Gaussian smoothness prior on the demographic history


Composite and Approximate Likelihoods
=======================================

In many settings, computing the exact likelihood is too expensive. Several
Timepieces use **approximations** that retain the essential structure while
being computationally feasible.

- **Composite likelihood**: treats different parts of the data as independent
  when they are not. For example, ``momi2`` treats each SNP as independent
  (ignoring linkage), and PHLASH combines an SFS likelihood with a pairwise
  HMM likelihood. Composite likelihoods give consistent parameter estimates
  but require corrections (Godambe information matrix, block bootstrap) for
  accurate uncertainty quantification.

- **Pairwise likelihood**: instead of modeling all :math:`n` samples jointly,
  PSMC and SMC++ model the coalescence of a single pair of haplotypes at a
  time. This reduces an :math:`n`-sample problem to a 2-sample problem at the
  cost of ignoring higher-order genealogical correlations.

- **Pseudo-likelihood**: tsinfer uses a heuristic likelihood based on the
  Li-Stephens copying model, which approximates the full genealogical
  likelihood with a computationally tractable HMM.

These approximations are not defects -- they are engineering choices that make
inference possible at genomic scale. Understanding *which* approximation each
tool makes, and what it costs in terms of statistical efficiency, is essential
for interpreting results correctly.


Worked Example: Composite Likelihood from Two Data Sources
------------------------------------------------------------

PHLASH combines an SFS likelihood with a pairwise coalescent HMM likelihood.
Here we build a simplified version to show how composite likelihoods work.

Suppose we want to estimate the effective population size :math:`N_e` (or
equivalently, the coalescence rate :math:`\lambda = 1/(2N_e)`) using two
independent data sources:

1. **The SFS**: from a sample of :math:`n` chromosomes, we observe counts of
   segregating sites at each frequency.
2. **Pairwise heterozygosity**: for a single diploid, we observe the fraction
   of sites that are heterozygous, which depends on the expected pairwise
   coalescence time.

Each source provides its own likelihood. The composite likelihood simply
multiplies them (adds the log-likelihoods):

.. math::

   \ell_{\text{comp}}(\lambda) = \ell_{\text{SFS}}(\lambda)
   + \ell_{\text{het}}(\lambda)

.. code-block:: python

   import numpy as np
   from scipy.special import gammaln

   def sfs_log_likelihood(theta, D_obs, n):
       """Poisson SFS log-likelihood under constant population size.

       Under the neutral coalescent, xi_k = theta / k.
       """
       k = np.arange(1, n)
       xi = theta / k
       xi = np.maximum(xi, 1e-300)
       return np.sum(D_obs * np.log(xi) - xi - gammaln(D_obs + 1))

   def het_log_likelihood(theta, n_het, n_sites):
       """Binomial log-likelihood for heterozygosity.

       The expected heterozygosity per site is approximately
       theta / (theta + n_sites) for small theta, or more precisely,
       the probability of a het site is 1 - exp(-theta / n_sites)
       under the Poisson model.
       """
       # Expected fraction of heterozygous sites
       p_het = 1.0 - np.exp(-theta / n_sites)
       p_het = np.clip(p_het, 1e-300, 1 - 1e-300)
       n_hom = n_sites - n_het
       return n_het * np.log(p_het) + n_hom * np.log(1.0 - p_het)

   # Simulate data from a population with theta_true = 200
   np.random.seed(42)
   theta_true = 200.0
   n_samples = 20
   n_sites = 100_000

   # Source 1: SFS
   k = np.arange(1, n_samples)
   xi_true = theta_true / k
   D_obs = np.random.poisson(xi_true)

   # Source 2: heterozygosity
   p_het_true = 1.0 - np.exp(-theta_true / n_sites)
   n_het = np.random.binomial(n_sites, p_het_true)

   # Compute individual and composite log-likelihoods
   thetas = np.linspace(50, 400, 300)
   ll_sfs = [sfs_log_likelihood(th, D_obs, n_samples) for th in thetas]
   ll_het = [het_log_likelihood(th, n_het, n_sites) for th in thetas]
   ll_comp = [s + h for s, h in zip(ll_sfs, ll_het)]

   ll_sfs = np.array(ll_sfs)
   ll_het = np.array(ll_het)
   ll_comp = np.array(ll_comp)

   theta_mle_sfs = thetas[np.argmax(ll_sfs)]
   theta_mle_het = thetas[np.argmax(ll_het)]
   theta_mle_comp = thetas[np.argmax(ll_comp)]

   print(f"True theta:           {theta_true:.1f}")
   print(f"MLE from SFS only:    {theta_mle_sfs:.1f}")
   print(f"MLE from het only:    {theta_mle_het:.1f}")
   print(f"MLE from composite:   {theta_mle_comp:.1f}")
   print(f"\nThe composite MLE balances the two data sources.")
   print(f"If they agree, the composite is sharper (more precise).")
   print(f"If they disagree, the composite finds a compromise.")

.. admonition:: Plain-language summary -- Why composite likelihoods work

   A composite likelihood combines multiple incomplete views of the same
   underlying process. Each view provides partial information: the SFS
   captures the frequency distribution but ignores linkage, while the
   pairwise HMM captures linkage but only for one pair. By adding their
   log-likelihoods, we get an estimate that benefits from both sources.

   The catch is that the two sources are not truly independent (they arise
   from the same genealogy), so the composite likelihood overestimates its
   own precision. The composite MLE is still *consistent* -- it converges to
   the true value with enough data -- but the standard errors from the
   composite likelihood are too small. This is why PHLASH uses the posterior
   (via SVGD with a prior) rather than the curvature of the composite
   likelihood for uncertainty quantification.


.. _amortized_inference:

The Other Paradigm: Neural Networks and Amortized Inference
=============================================================

Every Timepiece in this book follows the classical paradigm: **define a
generative model, derive (or approximate) the likelihood, optimize or sample**.
In recent years, a fundamentally different approach has emerged from machine
learning: **amortized inference** via neural networks.

The key idea
-------------

Instead of computing the likelihood for each new dataset, amortized inference
trains a neural network to *learn the mapping* from data to parameters
directly:

.. code-block:: text

   Classical (this book):    Data  ->  Likelihood function  ->  Optimizer/MCMC  ->  Parameters
   Amortized:                Data  ->  Trained neural network  ->  Parameters (instantly)

The training process is:

1. **Simulate** thousands (or millions) of datasets from the generative model
   under randomly sampled parameters.
2. **Train** a neural network to predict the parameters (or the full
   posterior) from each simulated dataset.
3. **Apply** the trained network to the real data to get parameter estimates
   in milliseconds.

This approach goes by many names: **simulation-based inference (SBI)**,
**neural posterior estimation (NPE)**, **likelihood-free inference**, or
**approximate Bayesian computation with neural density estimators (ABC-NDE)**.
Tools like ``dadi-ml``, ``pg-gan``, ``dinf``, ``popai``, and ``sbi``
implement variants of this idea for population genetics.

What amortized inference does well
------------------------------------

- **Speed at inference time.** Once trained, a neural network produces
  estimates in milliseconds -- compared to hours or days for MCMC. This is
  transformative for large-scale studies with thousands of populations.
- **Flexibility with summary statistics.** Neural networks can learn which
  features of the data are informative, without requiring the user to derive
  a likelihood function. This is powerful for complex models where the
  likelihood is intractable.
- **Scalability to complex models.** Models with many populations, continuous
  migration, and selection can be handled by simulating from them, even when
  no analytical likelihood exists.

What likelihood-based inference does well
-------------------------------------------

- **Transparency.** Every step of the inference is mathematically specified.
  You can inspect the likelihood surface, diagnose convergence, understand
  why a particular estimate was returned, and know exactly which assumptions
  are made. Neural networks are harder to interrogate -- their internal
  representations are opaque.
- **Statistical guarantees.** MLEs have well-understood asymptotic properties
  (consistency, efficiency, normality). Bayesian posteriors from MCMC
  converge to the true posterior under mild conditions. Neural amortized
  estimators provide no such guarantees in general -- their quality depends
  on the training data, the network architecture, and the summary statistics.
- **Extrapolation.** Likelihood-based methods work for *any* parameter
  values within the model, including regions far from the training
  distribution. Neural networks can fail silently when the real data fall
  outside the distribution of their training simulations -- a dangerous
  property when studying unusual populations or novel demographic scenarios.
- **No simulation budget.** Likelihood-based methods evaluate the model
  analytically or via efficient numerical algorithms. Amortized methods
  require simulating enough training data to cover the parameter space, which
  can be prohibitively expensive for high-dimensional models.
- **Interpretability of uncertainty.** Bayesian posteriors from MCMC or SVGD
  have clear probabilistic interpretations. Uncertainty estimates from neural
  networks (e.g., from neural posterior estimation) are only as reliable as
  the network's calibration, which is difficult to verify.

.. admonition:: Biology Aside -- When to use which paradigm

   **Use likelihood-based methods** (the focus of this book) when:

   - You need trustworthy uncertainty quantification (e.g., for
     publication-quality confidence intervals on divergence times)
   - The model is well-specified and the likelihood can be computed or
     well-approximated
   - Transparency and reproducibility are paramount
   - You are studying a small number of populations with a well-defined model

   **Consider amortized/neural methods** when:

   - The model is too complex for any analytical likelihood (e.g., many
     populations with continuous migration and selection)
   - You need to screen many populations quickly and will follow up
     interesting cases with more rigorous methods
   - You want to explore which summary statistics are informative
   - Speed at inference time is critical (e.g., real-time analysis pipelines)

   In practice, the two paradigms are **complementary**, not competing.
   Amortized methods can provide fast initial estimates that serve as starting
   points for likelihood-based refinement. Likelihood-based methods can
   validate the accuracy of neural estimators on specific datasets. The most
   rigorous analyses may use both.

Why this book focuses on the likelihood approach
--------------------------------------------------

This book is about understanding mechanisms -- opening the watch and seeing
every gear. Likelihood-based inference is inherently mechanistic: the
likelihood function *is* the model, expressed as a probability. Every
equation you derive, every algorithm you implement, directly encodes the
biological process that generated the data. When you build a PSMC from
scratch, you understand *exactly* how population size history maps to the
probability of observing a heterozygous site at a given genomic position.
That understanding doesn't come from training a neural network -- it comes
from building the gear train yourself.

Moreover, every amortized method ultimately relies on a generative model to
produce training data. Understanding how that model works -- how coalescence,
mutation, and recombination generate genetic variation -- is essential even if
you ultimately use a neural network for inference. The foundations in this book
are prerequisites for *both* paradigms.


Summary
========

This chapter has equipped you with the inferential toolkit that every
Timepiece depends on. Here is what you should take away:

**Four distributions** that generate genetic data:

.. list-table::
   :header-rows: 1
   :widths: 20 30 20 30

   * - Distribution
     - Role in population genetics
     - MLE
     - Where it appears
   * - Exponential(:math:`\lambda`)
     - Coalescence waiting times
     - :math:`\hat{\lambda} = 1/\bar{t}`
     - PSMC, ARGweaver, SINGER
   * - Poisson(:math:`\mu`)
     - Mutation counts, SFS entries
     - :math:`\hat{\mu} = \bar{k}`
     - dadi, moments, momi2, tsdate
   * - Gamma(:math:`\alpha, \beta`)
     - Node ages, branch lengths
     - Numerical
     - tsdate (prior + posterior)
   * - Gaussian(:math:`\mu, \sigma^2`)
     - Smoothness priors
     - :math:`\hat{\mu} = \bar{x}`
     - PHLASH (prior on log-sizes)

**Three inference strategies**:

1. **MLE** -- find the peak of the likelihood surface. Used by dadi, moments,
   momi2. Fast but gives only a point estimate.
2. **Bayesian inference** -- compute the full posterior distribution. Used by
   ARGweaver, SINGER, tsdate, PHLASH. More informative but computationally
   harder.
3. **Composite likelihood** -- combine multiple approximate likelihoods. Used
   by PHLASH, momi2. A practical compromise when the exact likelihood is
   intractable.

**One key conjugacy**: Gamma prior + Poisson likelihood = Gamma posterior.
This powers ``tsdate``'s Expectation Propagation and makes fast, exact
Bayesian updates possible.

The inferential logic, in four lines:

1. **Model**: Define a mechanistic model of how genetic data arise (coalescent
   + mutation + recombination + demography).
2. **Likelihood**: Compute (or approximate) :math:`P(\text{data} \mid \theta)`
   -- how probable the observed data are under each candidate parameter set.
3. **Optimize or sample**: Find the best parameters (MLE) or characterize
   the full posterior (Bayesian inference via MCMC, EP, or SVGD).
4. **Interpret**: Translate the statistical results back into biological
   conclusions about population sizes, divergence times, migration rates, and
   selection pressures.

Every Timepiece in this book implements steps 1--3 in a different way, tailored
to different data types and biological questions. But the underlying logic is
always the same: *the data are fixed, the model is the variable, and the
likelihood tells you which models are consistent with reality.*

Next: :ref:`coalescent_theory` -- the biological foundation that every
Timepiece depends on.
