.. _phlash_random_discretization:

================================
Random Time Discretization
================================

   *The tourbillon: a mechanism that cancels positional errors by rotating through them.*

In a mechanical watch, gravity pulls on the balance wheel and introduces a
systematic error that depends on the watch's orientation. A **tourbillon** is
a cage that slowly rotates the entire escapement through all orientations, so
that the errors from different positions cancel over time. It does not
eliminate the error at any single moment -- it ensures that the *average* error
is zero.

PHLASH's random time discretization works on exactly the same principle. Any
fixed time grid introduces discretization bias -- a systematic error that
depends on where the breakpoints fall. By **randomly sampling** the grid for
each gradient computation and averaging, the biases from different grids
cancel. No single evaluation is unbiased, but the average over many
evaluations is.

This chapter explains why fixed discretization creates bias, how random
discretization cancels it, and how to implement the randomized scheme.

.. admonition:: Biology Aside -- Why discretization matters for demographic inference

   Population size history is a continuous function of time -- :math:`N(t)`
   could change smoothly or have sharp transitions like bottlenecks. To
   compute the likelihood on a computer, we must approximate this continuous
   function on a discrete grid of time points. Where we place those grid
   points affects our answer: a grid that is too coarse in the period of a
   bottleneck will smooth it out; a grid whose breakpoints happen to align
   with a size change will capture it precisely. These are systematic errors
   -- they bias the inferred demographic history in a particular direction.
   For biologists, this means that the inferred history depends not just on
   the data but on an arbitrary methodological choice. Random discretization
   eliminates this dependence by averaging over many different grid
   placements.


The Discretization Problem
===========================

Recall from :ref:`psmc_discretization` that PSMC divides the time axis into
:math:`M` intervals :math:`[t_0, t_1), [t_1, t_2), \ldots, [t_{M-1}, t_M)`.
Within each interval, the coalescence rate is treated as constant. The
transition matrix, emission probabilities, and forward-backward algorithm
all operate on this discrete grid.

The discretization introduces two types of error:

1. **Approximation error in the transition matrix.** The continuous-time
   transition density :math:`q(t|s)` (derived in :ref:`psmc_continuous`)
   must be integrated over the intervals to produce the discrete transition
   probabilities :math:`p_{kl}`. The accuracy of this integration depends on
   the interval widths and the behavior of :math:`q(t|s)` within each interval.
   Wide intervals lose resolution; narrow intervals in some regions waste
   computational resources.

2. **Approximation error in the emission probabilities.** Within interval
   :math:`k`, the emission probability is computed at a representative time
   (e.g., the midpoint or the expected coalescence time conditional on
   falling in the interval). This point approximation introduces error
   whenever the emission probability varies appreciably across the interval.

For a fixed grid, these errors are **systematic**: they always go in the same
direction for a given demographic history and a given grid. If the grid is too
coarse in a region where :math:`N(t)` changes rapidly, the estimate will
consistently smooth over the change. If the grid boundaries happen to align
poorly with a bottleneck, the estimate will consistently misplace it.


Why Randomization Cancels Bias
================================

Let :math:`\hat{\ell}_G(\eta)` denote the discretized log-likelihood computed
on a particular grid :math:`G = \{t_0, t_1, \ldots, t_M\}`. The
discretization error is:

.. math::

   \hat{\ell}_G(\eta) = \ell(\eta) + b_G(\eta)

where :math:`\ell(\eta)` is the (unattainable) continuous-time log-likelihood
and :math:`b_G(\eta)` is the bias introduced by grid :math:`G`.

For a fixed grid, :math:`b_G(\eta)` is a fixed function of :math:`\eta` -- it
tilts the likelihood surface in a particular direction, biasing inference
toward demographic histories that happen to look good on that grid.

Now suppose we draw the grid randomly from some distribution :math:`\pi` over
grids. The key property is:

.. math::

   \mathbb{E}_{G \sim \pi}\left[b_G(\eta)\right] = 0

That is, the expected bias over random grids is zero. Different grids have
different biases, but these biases point in different directions and cancel
when averaged. The gradient of the expected log-likelihood is therefore
unbiased:

.. math::

   \mathbb{E}_{G \sim \pi}\left[\nabla_\eta \hat{\ell}_G(\eta)\right]
   = \nabla_\eta \ell(\eta)

This is exactly the property that SVGD needs: unbiased gradient estimates.
The variance of the gradient is increased by the randomization (each
individual gradient estimate is noisier), but SVGD is a stochastic algorithm
that naturally handles noisy gradients -- just like stochastic gradient
descent tolerates noisy mini-batch gradients in neural network training.


How the Grid Is Sampled
========================

PHLASH samples the time breakpoints as follows:

1. **Fix the endpoints.** The first breakpoint :math:`t_0 = 0` and the last
   breakpoint :math:`t_M = t_{\max}` are always the same. The maximum time
   :math:`t_{\max}` is chosen large enough that coalescence beyond it is
   negligible.

2. **Sample interior breakpoints.** The interior breakpoints
   :math:`t_1, \ldots, t_{M-1}` are drawn from a distribution that places
   them roughly log-spaced (denser near the present, sparser in the deep
   past), but with random perturbations. One natural choice is to sample
   :math:`\log t_k` uniformly in overlapping intervals that tile
   :math:`[\log t_0, \log t_M]`.

3. **Sort.** The sampled breakpoints are sorted to form a valid partition
   of :math:`[0, t_{\max}]`.

.. code-block:: python

   import numpy as np

   def sample_random_grid(M, t_max=10.0, t_min=1e-4, rng=None):
       """Sample a random time discretization grid.

       Interior breakpoints are log-uniformly spaced with random jitter,
       ensuring approximately uniform density per unit log-time.

       Parameters
       ----------
       M : int
           Number of time intervals.
       t_max : float
           Maximum time (coalescent units).
       t_min : float
           Minimum positive breakpoint.
       rng : numpy.random.Generator or None
           Random number generator.

       Returns
       -------
       grid : ndarray, shape (M+1,)
           Sorted breakpoints [0, t_1, ..., t_{M-1}, t_max].
       """
       if rng is None:
           rng = np.random.default_rng()

       # Sample M-1 interior breakpoints in log-space with jitter
       log_min, log_max = np.log(t_min), np.log(t_max)
       # Evenly spaced anchors in log-space
       anchors = np.linspace(log_min, log_max, M - 1)
       # Add uniform jitter of half the spacing
       spacing = (log_max - log_min) / (M - 1)
       jitter = rng.uniform(-0.5 * spacing, 0.5 * spacing, size=M - 1)
       log_breakpoints = anchors + jitter

       # Convert back, sort, and add endpoints
       interior = np.sort(np.exp(log_breakpoints))
       grid = np.concatenate([[0.0], interior, [t_max]])
       return grid

   # Demonstrate: sample three different grids
   rng = np.random.default_rng(42)
   for i in range(3):
       grid = sample_random_grid(M=32, t_max=10.0, rng=rng)
       print(f"Grid {i}: {len(grid)} breakpoints, "
             f"t_1={grid[1]:.5f}, t_mid={grid[16]:.4f}, "
             f"t_max={grid[-1]:.1f}")

   # Show that the grids differ (tourbillon rotates through configurations)
   g1 = sample_random_grid(32, rng=np.random.default_rng(0))
   g2 = sample_random_grid(32, rng=np.random.default_rng(1))
   max_diff = np.max(np.abs(g1 - g2))
   print(f"\nMax difference between two grids: {max_diff:.4f}")
   print("(Different grids = different biases = cancellation when averaged)")

The distribution over grids is chosen so that the expected density of
breakpoints per unit log-time is approximately uniform. This ensures that all
timescales receive adequate resolution in expectation, even though any single
draw may have gaps.

.. admonition:: Plain-language summary -- Log-spacing and human timescales

   Demographic events span a huge range of timescales: a recent population
   expansion might have occurred 500 generations ago (~12,500 years), while
   the ancestral human-Neanderthal divergence is ~20,000 generations ago
   (~500,000 years), and deep coalescent events can reach millions of years.
   A log-spaced grid naturally allocates more resolution to recent events
   (which affect more of the genome and are easier to detect) and less to
   ancient events. The random jitter on top of this log-spacing ensures that
   no specific time point is systematically favored or disfavored.

.. admonition:: The tourbillon analogy, precisely

   In a tourbillon, the escapement rotates through all angular positions
   over one revolution, and the average rate error over the revolution is
   zero. In PHLASH, the discretization grid "rotates" through all possible
   placements of the breakpoints, and the average discretization bias over
   many draws is zero. The period of rotation is one SVGD iteration (each
   iteration uses a fresh grid). Just as a tourbillon must complete many
   rotations for the cancellation to be effective, PHLASH must average over
   many random grids for the bias to vanish.


Practical Considerations
=========================

**How many breakpoints?** The number of intervals :math:`M` is a
hyperparameter. More intervals give finer resolution but increase the cost
of the forward algorithm (which scales as :math:`O(LM^2)` per pair). In
practice, :math:`M = 32` to :math:`M = 64` intervals are typical, matching
the range used by PSMC.

**How many random draws per iteration?** In the simplest implementation,
each SVGD step uses a single random grid, and the averaging happens
implicitly across SVGD iterations. More sophisticated implementations can
average the gradient over several grids within a single step to reduce
variance, at the cost of proportionally more computation.

**Interaction with the score function algorithm.** The score function
algorithm (described in the :ref:`next chapter <phlash_score_function>`)
computes the gradient of the log-likelihood for a given grid. Because the
algorithm's complexity is :math:`O(LM^2)`, the cost of each random grid
evaluation is the same as a fixed-grid evaluation. The randomization adds
no computational overhead per evaluation -- it only adds variance, which
decreases as :math:`1/\sqrt{K}` with the number of averaged grids :math:`K`.


Connection to Debiased Estimation
===================================

Random discretization belongs to a family of techniques in computational
statistics known as **debiased estimation** or **randomized numerical
integration**. The idea appears in many contexts:

- **Stochastic mini-batching** in SGD: each mini-batch gives a noisy but
  unbiased gradient estimate; the noise decreases with more data points
  in the batch.

- **Russian roulette estimators** for infinite series: randomly truncate
  a series expansion and reweight, producing an unbiased estimate of the
  infinite sum.

- **Randomized quadrature**: randomly shift a quadrature grid to debiase
  the numerical integral, then average across shifts.

PHLASH's random discretization is closest to randomized quadrature: the
time integral in the transition density is approximated on a shifted grid,
and the shift is random. The key insight is that unbiased gradients are
sufficient for convergent optimization (or, in PHLASH's case, convergent
posterior sampling via SVGD).

.. code-block:: python

   def debiased_gradient_estimate(eta, observed_sfs, n_grids=10, M=32,
                                   rng=None):
       """Estimate the likelihood gradient by averaging over random grids.

       This demonstrates the debiasing principle: each individual gradient
       is biased, but their average converges to the true gradient.

       Parameters
       ----------
       eta : ndarray, shape (M,)
           Population sizes at each time interval.
       observed_sfs : ndarray
           Observed SFS.
       n_grids : int
           Number of random grids to average over.
       M : int
           Number of time intervals per grid.
       rng : numpy.random.Generator or None
           Random number generator.

       Returns
       -------
       mean_gradient : ndarray
           Averaged gradient estimate.
       std_gradient : ndarray
           Standard deviation across grid evaluations.
       """
       if rng is None:
           rng = np.random.default_rng()

       gradients = []
       for _ in range(n_grids):
           grid = sample_random_grid(M, rng=rng)
           # In a real implementation, this would compute the HMM gradient
           # on this grid. Here we simulate a noisy gradient.
           true_gradient = -0.1 * np.log(eta)  # placeholder: pulls toward 1
           noise = rng.normal(0, 0.05, size=len(eta))
           gradients.append(true_gradient + noise)

       gradients = np.array(gradients)
       return gradients.mean(axis=0), gradients.std(axis=0)

   # Demonstrate variance reduction via averaging
   eta = np.exp(np.zeros(32))  # constant population size
   rng = np.random.default_rng(42)
   for K in [1, 5, 20]:
       mean_grad, std_grad = debiased_gradient_estimate(
           eta, D_observed, n_grids=K, rng=rng
       )
       print(f"K={K:2d} grids: mean |grad| = {np.mean(np.abs(mean_grad)):.4f}, "
             f"std = {np.mean(std_grad):.4f}")
   print("(Variance decreases as 1/sqrt(K) -- more grids, less noise)")


What Comes Next
================

We now know how to compute the composite likelihood on a random grid. But
computing the likelihood is not enough -- we need its **gradient** with
respect to the demographic parameters, because SVGD requires gradients. The
:ref:`next chapter <phlash_score_function>` derives the score function
algorithm that computes this gradient in :math:`O(LM^2)` time, 30--90x
faster than reverse-mode automatic differentiation. This is the gear train --
the mechanism that transmits the mainspring's energy efficiently to the hands.
