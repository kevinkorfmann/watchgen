.. _clues_wright_fisher_hmm:

===========================================
The Wright-Fisher HMM
===========================================

   *The escapement: how selection biases the random walk of allele frequencies.*

This chapter builds the Hidden Markov Model at the heart of CLUES. The hidden
states are discretized allele frequencies, and the transitions encode one-generation
steps of the Wright-Fisher diffusion with selection. By the end, you will have
implemented the frequency bin construction and the transition matrix -- the
mathematical backbone of the entire inference machine.


Step 1: The Wright-Fisher Diffusion With Selection
====================================================

Recall from the :ref:`coalescent theory prerequisite <coalescent_theory>` that the
Wright-Fisher model describes a population of :math:`2N` gene copies (haploid size)
at a biallelic locus. Each generation, the next generation's alleles are drawn by
sampling with replacement from the current generation, weighted by fitness.

Let :math:`x_t` be the derived allele frequency at generation :math:`t`. Under
the diploid fitness model with genotype fitnesses :math:`w_{AA} = 1`,
:math:`w_{AD} = 1 + hs`, :math:`w_{DD} = 1 + s`:

The **expected frequency** of the derived allele in the next generation is:

.. math::

   \mathbb{E}[x_{t+1} \mid x_t] = \frac{x_t^2 (1 + s) + x_t(1 - x_t)(1 + hs)}{
   x_t^2 (1 + s) + 2 x_t(1 - x_t)(1 + hs) + (1 - x_t)^2}

This is obtained by computing the mean fitness-weighted frequency. The numerator
counts the expected contribution of derived alleles: homozygous :math:`DD` (frequency
:math:`x_t^2`, each contributing 2 derived copies with fitness :math:`1+s`) plus
heterozygous :math:`AD` (frequency :math:`2x_t(1-x_t)`, each contributing 1 derived
copy with fitness :math:`1+hs`). After simplification and dividing by the mean
fitness :math:`\bar{w}`:

.. math::

   \mathbb{E}[x_{t+1}] = \frac{x_t(1 + s \cdot x_t + hs(1 - x_t))}{1 + s \cdot x_t(x_t + 2h(1 - x_t))}

For **additive selection** (:math:`h = 0.5`), this simplifies to:

.. math::

   \mathbb{E}[x_{t+1}] = \frac{x_t(1 + s x_t)}{1 + s x_t} + \frac{s x_t(1 - x_t)}{2(1 + s x_t)}

The **variance** of the frequency change per generation, due to genetic drift, is:

.. math::

   \text{Var}[x_{t+1} \mid x_t] = \frac{x_t(1 - x_t)}{2N}

This is the binomial sampling variance: drawing :math:`2N` copies with probability
:math:`x_t` gives a variance of :math:`x_t(1-x_t)/(2N)`.

.. admonition:: Why does selection affect the mean but not the variance?

   The mean frequency change has two components: drift (zero mean) and selection
   (directional bias). The variance is dominated by the sampling noise of
   reproduction, which depends on :math:`x_t` and :math:`N` but not on :math:`s`
   (to first order). Selection shifts the *center* of the distribution of next-
   generation frequencies; drift determines its *width*. For typical values of
   :math:`s` (less than 0.1), the correction to the variance is negligible.


Going Backward in Time
-----------------------

CLUES models the allele frequency trajectory **backward in time** (from present to
past), because the backward direction is natural for the coalescent. Going backward,
the expected frequency at time :math:`t+1` (one generation further into the past)
given frequency :math:`x_t` at time :math:`t` is the *inverse* of the forward
mapping.

If the forward mapping is :math:`x_{t+1} = g(x_t)`, then the backward mapping is
:math:`x_t = g^{-1}(x_{t+1})`. For the general dominance model, the backward
mean is:

.. math::

   \mu(x) = x + \frac{s(-1 + x) \cdot x \cdot (-x + h(-1 + 2x))}{-1 + s(2h(-1 + x) - x) \cdot x}

This is the formula used in the CLUES2 source code. For additive selection
(:math:`h = 0.5`), it simplifies to:

.. math::

   \mu(x) = x - \frac{s \cdot x(1-x)}{2(1 + sx)}

**Intuition:** Going backward in time, a positively selected allele (:math:`s > 0`)
was at a *lower* frequency in the past. So the backward mean is shifted *downward*
relative to the current frequency -- the :math:`s`-dependent term is negative when
:math:`x > 0` and :math:`s > 0`. (The formula above uses the convention that
:math:`\mu(x) < x` when :math:`s > 0`, consistent with looking backward.)

.. code-block:: python

   import numpy as np

   def backward_mean(x, s, h=0.5):
       """Expected allele frequency one generation further into the past.

       Given current derived allele frequency x, selection coefficient s,
       and dominance coefficient h, compute the mean of the backward
       Wright-Fisher transition.

       For additive selection (h=0.5), this simplifies to:
           mu = x + s * x * (1-x) / (2 * (1 + s*x))

       Parameters
       ----------
       x : float
           Current derived allele frequency (0 < x < 1).
       s : float
           Selection coefficient. Positive = derived allele favored.
       h : float
           Dominance coefficient. Default 0.5 (additive).

       Returns
       -------
       mu : float
           Expected frequency one generation into the past.
       """
       numerator = s * (-1 + x) * x * (-x + h * (-1 + 2 * x))
       denominator = -1 + s * (2 * h * (-1 + x) - x) * x
       return x + numerator / denominator

   def backward_std(x, N):
       """Standard deviation of allele frequency change per generation.

       Parameters
       ----------
       x : float
           Current derived allele frequency.
       N : float
           Haploid effective population size (= 2 * diploid N_e).

       Returns
       -------
       sigma : float
           Standard deviation of the frequency change.
       """
       return np.sqrt(x * (1.0 - x) / N)

   # Verify: under neutrality (s=0), backward mean equals current frequency
   for x in [0.1, 0.3, 0.5, 0.7, 0.9]:
       mu = backward_mean(x, s=0.0)
       print(f"x = {x:.1f}, s = 0:   mu = {mu:.6f} (expected: {x:.6f})")

   # Under positive selection, backward mean < current frequency
   # (the allele was rarer in the past)
   print()
   for x in [0.1, 0.3, 0.5, 0.7, 0.9]:
       mu = backward_mean(x, s=0.05)
       print(f"x = {x:.1f}, s = 0.05: mu = {mu:.6f} (shift: {mu - x:+.6f})")


Step 2: Frequency Bin Construction
====================================

The continuous allele frequency :math:`x \in [0, 1]` must be discretized into
:math:`K` bins for the HMM. The choice of bin placement matters: we need more
resolution near the boundaries (0 and 1), where the dynamics are slowest and the
allele may linger for many generations before fixing or being lost.

CLUES uses the **quantile function of a Beta(1/2, 1/2) distribution** to place the
frequency bins. The Beta(1/2, 1/2) distribution (also known as the arcsine
distribution) has most of its mass near 0 and 1, so its quantiles are densely
packed near the boundaries and sparser in the middle -- exactly the spacing we want.

.. admonition:: Probability Aside: the Beta distribution and its quantile function

   The **Beta distribution** :math:`\text{Beta}(\alpha, \beta)` is a family of
   distributions on :math:`[0, 1]`. Its density is:

   .. math::

      f(x; \alpha, \beta) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}

   The special case :math:`\text{Beta}(1/2, 1/2)` has density
   :math:`f(x) = 1/(\pi \sqrt{x(1-x)})`, which is U-shaped: it concentrates
   mass near 0 and 1. This is related to the stationary distribution of the
   Wright-Fisher diffusion under neutrality, making it a natural choice for
   spacing frequency bins.

   The **quantile function** (inverse CDF) :math:`Q(p)` maps a uniform
   probability :math:`p \in [0,1]` to the value :math:`x` such that
   :math:`P(X \leq x) = p`. Evaluating :math:`Q` at equally-spaced points in
   :math:`[0, 1]` produces unevenly-spaced :math:`x` values -- more
   concentrated where the CDF is flat (near 0 and 1), which is exactly where
   the diffusion dynamics are slowest.

**The construction.** Given :math:`K` bins (default :math:`K = 450`):

1. Create :math:`K` equally-spaced points :math:`u_1, u_2, \ldots, u_K` in
   :math:`[0, 1]`.

2. Map each through the Beta(1/2, 1/2) quantile function:
   :math:`x_k = Q_{\text{Beta}(1/2, 1/2)}(u_k)`.

3. Set the boundary values: :math:`x_1 = 0` (loss), :math:`x_K = 1` (fixation).
   These are **absorbing states** -- once the allele is lost or fixed, it stays
   there.

4. For log-probability computations, use :math:`\log(x_1) = \log(\epsilon)` and
   :math:`\log(1 - x_K) = \log(\epsilon)` with :math:`\epsilon = 10^{-12}` to
   avoid :math:`-\infty`.

.. code-block:: python

   from scipy.stats import beta as beta_dist

   def build_frequency_bins(K=450):
       """Construct the K allele frequency bins using Beta(1/2, 1/2) quantiles.

       The bins are denser near 0 and 1, where the Wright-Fisher dynamics
       are slowest (frequency changes are small when the allele is very
       rare or very common).

       Parameters
       ----------
       K : int
           Number of frequency bins. Default 450 (as in CLUES2).

       Returns
       -------
       freqs : ndarray of shape (K,)
           Frequency bin centers. freqs[0] = 0 (loss), freqs[-1] = 1 (fixation).
       logfreqs : ndarray of shape (K,)
           log(freqs), with a small epsilon to avoid -inf at boundaries.
       log1minusfreqs : ndarray of shape (K,)
           log(1 - freqs), with a small epsilon at boundaries.
       """
       # Step 1: equally-spaced points in [0, 1]
       u = np.linspace(0.0, 1.0, K)

       # Step 2: map through Beta(1/2, 1/2) quantile function
       freqs = beta_dist.ppf(u, 0.5, 0.5)

       # Step 3: set boundary values for log computations
       eps = 1e-12
       freqs[0] = eps       # temporarily, for log computation
       freqs[-1] = 1 - eps  # temporarily, for log computation

       logfreqs = np.log(freqs)
       log1minusfreqs = np.log(1.0 - freqs)

       # Now set the actual boundary values
       freqs[0] = 0.0
       freqs[-1] = 1.0

       return freqs, logfreqs, log1minusfreqs

   # Build the bins and examine the spacing
   freqs, logfreqs, log1minusfreqs = build_frequency_bins(K=450)
   print(f"Number of bins: {len(freqs)}")
   print(f"First 10 bins: {np.round(freqs[:10], 6)}")
   print(f"Last 10 bins:  {np.round(freqs[-10:], 6)}")
   print(f"Bin spacing near 0: {np.diff(freqs[:5])}")
   print(f"Bin spacing near 0.5: {np.diff(freqs[220:226])}")
   print(f"Bin spacing near 1: {np.diff(freqs[-5:])}")

The output shows that the bins are very finely spaced near 0 and 1 (spacing
:math:`\sim 10^{-5}`) and coarser near 0.5 (spacing :math:`\sim 0.005`). This is
exactly what we need: the Wright-Fisher diffusion spends a lot of time near the
boundaries before absorption, and we need fine resolution there to track the
trajectory accurately.


Step 3: The Transition Matrix
==============================

The transition matrix :math:`\mathbf{P}` has entries :math:`P_{ij}` giving the
probability of moving from frequency bin :math:`i` to frequency bin :math:`j` in
one generation (backward in time). We approximate each row using a **normal
distribution** centered at the backward mean :math:`\mu_i` with standard deviation
:math:`\sigma_i`.

For each starting frequency :math:`x_i` (with :math:`0 < x_i < 1`):

1. Compute the backward mean :math:`\mu_i = \mu(x_i)` and standard deviation
   :math:`\sigma_i = \sqrt{x_i(1 - x_i)/(2N)}`.

2. For the boundary bin :math:`j = 0` (frequency 0, allele loss):

   .. math::

      P_{i,0} = \Phi\left(\frac{(x_0 + x_1)/2 - \mu_i}{\sigma_i}\right)

   All probability mass below the midpoint between the first two bins is placed
   in the loss bin.

3. For the boundary bin :math:`j = K-1` (frequency 1, fixation):

   .. math::

      P_{i,K-1} = 1 - \Phi\left(\frac{(x_{K-2} + x_{K-1})/2 - \mu_i}{\sigma_i}\right)

4. For interior bins :math:`0 < j < K-1`:

   .. math::

      P_{i,j} = \Phi\left(\frac{(x_j + x_{j+1})/2 - \mu_i}{\sigma_i}\right)
              - \Phi\left(\frac{(x_{j-1} + x_j)/2 - \mu_i}{\sigma_i}\right)

5. **Absorbing states:** :math:`P_{0,0} = 1` and :math:`P_{K-1,K-1} = 1`.
   Once the allele is lost or fixed, it stays there forever.

6. **Renormalize** each row to sum to 1, to correct for numerical truncation.

.. admonition:: Why the normal approximation?

   The Wright-Fisher model produces a binomial distribution of next-generation
   frequencies, but for large :math:`N`, the binomial is well approximated by a
   normal distribution with the same mean and variance. This is the **diffusion
   approximation** -- the same idea that underpins the Wright-Fisher diffusion
   equation. For population sizes in the thousands, the approximation is
   excellent. CLUES exploits this to compute transition probabilities using the
   fast, numerically stable normal CDF rather than expensive binomial sums.

**Sparsity optimization.** Most of the probability mass in row :math:`i` is
concentrated within :math:`\pm 3.3\sigma_i` of the mean :math:`\mu_i`. Beyond
this range, the probability is less than 0.1%. CLUES only computes transition
probabilities within this range, producing a **sparse, banded transition matrix**.
This is Approximation A1 from the CLUES2 paper and provides the first major speedup.

.. code-block:: python

   from scipy.stats import norm

   def build_transition_matrix(freqs, N, s, h=0.5):
       """Build the K x K log-transition matrix for the Wright-Fisher HMM.

       Each entry P[i, j] is the log-probability of transitioning from
       frequency bin i to frequency bin j in one generation (backward in time).

       Parameters
       ----------
       freqs : ndarray of shape (K,)
           Frequency bin centers (from build_frequency_bins).
       N : float
           Haploid effective population size.
       s : float
           Selection coefficient.
       h : float
           Dominance coefficient (default 0.5 = additive).

       Returns
       -------
       logP : ndarray of shape (K, K)
           Log-transition matrix.
       """
       K = len(freqs)
       logP = np.full((K, K), -np.inf)

       # Midpoints between consecutive bins (used for CDF integration)
       midpoints = (freqs[1:] + freqs[:-1]) / 2.0

       # Absorbing states: loss and fixation
       logP[0, 0] = 0.0       # log(1) = 0
       logP[K - 1, K - 1] = 0.0

       for i in range(1, K - 1):
           x = freqs[i]
           mu = backward_mean(x, s, h)
           sigma = backward_std(x, N)

           if sigma < 1e-15:
               # Degenerate case: no drift, all mass at mu
               closest = np.argmin(np.abs(freqs - mu))
               logP[i, closest] = 0.0
               continue

           # Only compute within +/- 3.3 sigma (99.9% of probability mass)
           lower_freq = mu - 3.3 * sigma
           upper_freq = mu + 3.3 * sigma
           j_lower = max(0, np.searchsorted(freqs, lower_freq) - 1)
           j_upper = min(K, np.searchsorted(freqs, upper_freq) + 1)

           # Compute probability mass in each bin using normal CDF
           row = np.zeros(K)
           for j in range(j_lower, j_upper):
               if j == 0:
                   # All mass below midpoint[0] goes to bin 0 (loss)
                   row[j] = norm.cdf(midpoints[0], loc=mu, scale=sigma)
               elif j == K - 1:
                   # All mass above midpoint[-1] goes to bin K-1 (fixation)
                   row[j] = 1.0 - norm.cdf(midpoints[-1], loc=mu, scale=sigma)
               else:
                   # Mass between midpoints[j-1] and midpoints[j]
                   row[j] = (norm.cdf(midpoints[j], loc=mu, scale=sigma)
                             - norm.cdf(midpoints[j - 1], loc=mu, scale=sigma))

           # Renormalize (corrects for truncation beyond 3.3 sigma)
           row_sum = row.sum()
           if row_sum > 0:
               row /= row_sum
           else:
               row[np.argmin(np.abs(freqs - mu))] = 1.0

           # Convert to log probabilities
           logP[i, :] = np.where(row > 0, np.log(row), -np.inf)

       return logP

   # Build and verify the transition matrix
   freqs, logfreqs, log1minusfreqs = build_frequency_bins(K=50)
   N_haploid = 20000.0  # haploid size (= 2 * diploid N_e of 10,000)

   # Neutral transition matrix
   logP_neutral = build_transition_matrix(freqs, N_haploid, s=0.0)
   P_neutral = np.exp(logP_neutral)
   print("Row sums (should be ~1.0):", np.round(P_neutral.sum(axis=1)[:5], 6))

   # Selection transition matrix (s = 0.05)
   logP_selected = build_transition_matrix(freqs, N_haploid, s=0.05)
   P_selected = np.exp(logP_selected)
   print("Row sums (should be ~1.0):", np.round(P_selected.sum(axis=1)[:5], 6))


Visualizing the Effect of Selection on Transitions
----------------------------------------------------

Let's compare the transition probabilities from a frequency of :math:`x = 0.3`
under neutrality vs. positive selection:

.. code-block:: python

   # Find the bin closest to x = 0.3
   target_freq = 0.3
   i = np.argmin(np.abs(freqs - target_freq))
   print(f"Starting frequency bin: x[{i}] = {freqs[i]:.4f}")

   # Extract transition probabilities from that bin
   trans_neutral = np.exp(logP_neutral[i, :])
   trans_selected = np.exp(logP_selected[i, :])

   # Show the shift in the distribution
   mean_neutral = np.sum(freqs * trans_neutral)
   mean_selected = np.sum(freqs * trans_selected)
   print(f"Mean next frequency (neutral):  {mean_neutral:.6f}")
   print(f"Mean next frequency (s=0.05):   {mean_selected:.6f}")
   print(f"Shift due to selection:         {mean_selected - mean_neutral:+.6f}")
   print(f"(Going backward, s>0 shifts the allele to lower past frequency)")

Under positive selection, the backward transition shifts the distribution *downward*
-- the allele was at a lower frequency in the past, consistent with it having
been pushed upward by selection.


Step 4: The Dominance Extension
================================

The general dominance model allows the fitness of the heterozygote :math:`AD` to be
anywhere between :math:`AA` and :math:`DD`. The dominance coefficient :math:`h`
controls this:

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - :math:`h`
     - Name
     - Heterozygote fitness
   * - 0
     - Recessive
     - :math:`w_{AD} = 1` (same as :math:`AA`)
   * - 0.5
     - Additive
     - :math:`w_{AD} = 1 + s/2` (halfway between)
   * - 1
     - Dominant
     - :math:`w_{AD} = 1 + s` (same as :math:`DD`)

The backward mean for general :math:`h` is the full formula we implemented in
``backward_mean``. The key difference from additive selection:

- **Recessive** (:math:`h = 0`): selection is weak when the allele is rare
  (most copies are in heterozygotes, which have no fitness advantage). The allele
  frequency trajectory shows a slow initial rise followed by rapid fixation.

- **Dominant** (:math:`h = 1`): selection is strong even when the allele is rare
  (heterozygotes already have the full advantage). The trajectory shows a rapid
  initial rise followed by a slow approach to fixation.

.. code-block:: python

   # Compare backward means under different dominance models
   x_vals = np.linspace(0.01, 0.99, 50)
   s = 0.05

   for h_val, label in [(0.0, "Recessive (h=0)"),
                          (0.5, "Additive (h=0.5)"),
                          (1.0, "Dominant (h=1)")]:
       shifts = [backward_mean(x, s, h=h_val) - x for x in x_vals]
       # The shift should be negative (allele was rarer in the past)
       max_shift_freq = x_vals[np.argmin(shifts)]
       print(f"{label}: max backward shift at x = {max_shift_freq:.2f}, "
             f"shift = {min(shifts):.6f}")

   # For recessive: max shift is at high x (selection acts mainly on homozygotes)
   # For dominant: max shift is at low x (selection acts on heterozygotes)
   # For additive: max shift is at x ~ 0.5


Step 5: Precomputing the Standard Normal CDF
==============================================

CLUES2 avoids calling ``scipy.stats.norm.cdf`` millions of times by precomputing a
lookup table of the standard normal CDF. The transition matrix computation for each
row involves evaluating the normal CDF at several points, and doing this :math:`K`
times per generation can be expensive. The lookup table uses linear interpolation
for speed.

.. code-block:: python

   def build_normal_cdf_lookup(n_points=2000):
       """Precompute a lookup table for the standard normal CDF.

       The standard normal CDF Phi(z) is evaluated on a grid of z-values
       and stored for fast interpolation. Any normal CDF can be computed
       from the standard normal by the transformation:
           Phi((x - mu) / sigma) = P(N(mu, sigma^2) <= x)

       Parameters
       ----------
       n_points : int
           Number of grid points for the lookup table.

       Returns
       -------
       z_bins : ndarray
           Grid of z-values (from ~-37 to ~37 for n_points=2000).
       z_cdf : ndarray
           Standard normal CDF values at each z-bin.
       """
       # Create points in (0, 1), avoiding exactly 0 and 1
       u = np.linspace(0.0, 1.0, n_points)
       u[0] = 1e-10
       u[-1] = 1 - 1e-10

       # Map through the inverse normal CDF (quantile function)
       z_bins = norm.ppf(u)
       z_cdf = norm.cdf(z_bins)  # = u (by construction), but useful for interpolation

       return z_bins, z_cdf

   def fast_normal_cdf(x, mu, sigma, z_bins, z_cdf):
       """Evaluate the normal CDF at x using the precomputed lookup table.

       Transforms (x - mu) / sigma to a standard normal z-score,
       then interpolates from the lookup table.

       Parameters
       ----------
       x : float
           Point at which to evaluate the CDF.
       mu : float
           Mean of the normal distribution.
       sigma : float
           Standard deviation.
       z_bins : ndarray
           Precomputed z-values.
       z_cdf : ndarray
           Precomputed CDF values.

       Returns
       -------
       cdf_value : float
           Phi((x - mu) / sigma).
       """
       z = (x - mu) / sigma
       return np.interp(z, z_bins, z_cdf)

   # Verify the lookup table
   z_bins, z_cdf = build_normal_cdf_lookup()
   test_points = [-2.0, -1.0, 0.0, 1.0, 2.0]
   print("Verification of fast normal CDF lookup:")
   for z in test_points:
       exact = norm.cdf(z)
       approx = fast_normal_cdf(z, 0.0, 1.0, z_bins, z_cdf)
       print(f"  z = {z:+.1f}: exact = {exact:.8f}, approx = {approx:.8f}, "
             f"error = {abs(exact - approx):.2e}")


Step 6: Putting It Together -- Building the Complete Transition Matrix
======================================================================

Here is the optimized version that uses the sparse computation and precomputed
lookup table, closely following the CLUES2 implementation:

.. code-block:: python

   def build_transition_matrix_fast(freqs, N, s, z_bins, z_cdf, h=0.5):
       """Build the log-transition matrix using sparse computation.

       This is the optimized version that only computes entries within
       3.3 sigma of the mean, matching the CLUES2 implementation.

       Parameters
       ----------
       freqs : ndarray of shape (K,)
           Frequency bins.
       N : float
           Haploid effective population size.
       s : float
           Selection coefficient.
       z_bins : ndarray
           Precomputed standard normal z-values.
       z_cdf : ndarray
           Precomputed standard normal CDF values.
       h : float
           Dominance coefficient.

       Returns
       -------
       logP : ndarray of shape (K, K)
           Log-transition matrix.
       lower_indices : ndarray of shape (K,)
           First nonzero column index for each row.
       upper_indices : ndarray of shape (K,)
           Last nonzero column index (+1) for each row.
       """
       K = len(freqs)
       logP = np.full((K, K), -np.inf)
       lower_indices = np.zeros(K, dtype=int)
       upper_indices = np.zeros(K, dtype=int)

       # Midpoints between bins
       midpoints = (freqs[1:] + freqs[:-1]) / 2.0

       # Absorbing states
       logP[0, 0] = 0.0
       logP[K - 1, K - 1] = 0.0
       lower_indices[0] = 0
       upper_indices[0] = 1
       lower_indices[K - 1] = K - 1
       upper_indices[K - 1] = K

       for i in range(1, K - 1):
           x = freqs[i]
           mu = backward_mean(x, s, h)
           sigma = backward_std(x, N)

           if sigma < 1e-15:
               closest = np.argmin(np.abs(freqs - mu))
               logP[i, closest] = 0.0
               lower_indices[i] = closest
               upper_indices[i] = closest + 1
               continue

           # Bounds for sparse computation (3.3 sigma captures 99.9%)
           lower_freq = mu - 3.3 * sigma
           upper_freq = mu + 3.3 * sigma
           j_lo = max(0, np.searchsorted(freqs, lower_freq) - 1)
           j_hi = min(K, np.searchsorted(freqs, upper_freq) + 1)

           # Compute row probabilities
           row = np.zeros(K)
           for j in range(j_lo, j_hi):
               if j == 0:
                   row[j] = fast_normal_cdf(midpoints[0], mu, sigma,
                                             z_bins, z_cdf)
               elif j == K - 1:
                   row[j] = 1.0 - fast_normal_cdf(midpoints[-1], mu, sigma,
                                                    z_bins, z_cdf)
               else:
                   row[j] = (fast_normal_cdf(midpoints[j], mu, sigma,
                                              z_bins, z_cdf)
                             - fast_normal_cdf(midpoints[j - 1], mu, sigma,
                                                z_bins, z_cdf))

           # Renormalize and convert to log
           row_sum = row.sum()
           if row_sum > 0:
               row /= row_sum
           logP[i, :] = np.where(row > 0, np.log(row), -np.inf)

           # Record nonzero range for fast summation (Approximation A2)
           nonzero = np.where(row > 0)[0]
           if len(nonzero) > 0:
               lower_indices[i] = nonzero[0]
               upper_indices[i] = nonzero[-1] + 1
           else:
               lower_indices[i] = 0
               upper_indices[i] = K

       return logP, lower_indices, upper_indices

   # Build and verify
   freqs, logfreqs, log1minusfreqs = build_frequency_bins(K=100)
   z_bins, z_cdf = build_normal_cdf_lookup()
   N_haploid = 20000.0

   logP, lo, hi = build_transition_matrix_fast(
       freqs, N_haploid, s=0.02, z_bins=z_bins, z_cdf=z_cdf)

   # Check sparsity: how many nonzero entries per row?
   nnz_per_row = [hi[i] - lo[i] for i in range(len(freqs))]
   print(f"Average nonzero entries per row: {np.mean(nnz_per_row):.1f}")
   print(f"Max nonzero entries: {max(nnz_per_row)}")
   print(f"Matrix size: {len(freqs)}x{len(freqs)} = {len(freqs)**2}")
   print(f"Sparsity: {100 * (1 - np.mean(nnz_per_row)/len(freqs)):.1f}% zeros")


The Log-Sum-Exp Trick
-----------------------

Throughout CLUES, all probabilities are stored in **log space** to avoid numerical
underflow. The key operation is the **log-sum-exp**:

.. math::

   \log\left(\sum_j e^{a_j}\right) = a_{\max} + \log\left(\sum_j e^{a_j - a_{\max}}\right)

where :math:`a_{\max} = \max_j a_j`. This prevents overflow or underflow when
the :math:`a_j` values span a huge range.

.. code-block:: python

   def logsumexp(a):
       """Compute log(sum(exp(a))) in a numerically stable way.

       This is the fundamental building block for all HMM computations
       in log space. The trick: subtract the maximum before exponentiating,
       then add it back after taking the log.

       Parameters
       ----------
       a : ndarray
           Array of log-probabilities.

       Returns
       -------
       result : float
           log(sum(exp(a))).
       """
       a_max = np.max(a)
       if a_max == -np.inf:
           return -np.inf
       return a_max + np.log(np.sum(np.exp(a - a_max)))

   # Verify: log(exp(-1000) + exp(-1001)) should be about -999.69
   result = logsumexp(np.array([-1000.0, -1001.0]))
   print(f"logsumexp([-1000, -1001]) = {result:.4f}")
   print(f"Expected: {-1000 + np.log(1 + np.exp(-1)):.4f}")


Exercises
=========

.. admonition:: Exercise 1: Frequency bins and the boundary effect

   Build frequency bins with :math:`K = 50, 100, 200, 450` and compare:

   (a) The spacing between the first 5 bins (near :math:`x = 0`).
   (b) The spacing at :math:`x \approx 0.5`.
   (c) Build the transition matrix for each :math:`K` with :math:`N = 10000`
       (haploid) and :math:`s = 0.02`. What fraction of the matrix is nonzero?
       How does the sparsity change with :math:`K`?

.. admonition:: Exercise 2: Selection shifts the transition distribution

   For :math:`N = 20000` (haploid), :math:`K = 100`, starting frequency
   :math:`x = 0.5`:

   (a) Plot the transition probability distribution (row :math:`i` of the
       transition matrix) for :math:`s = -0.05, 0, 0.01, 0.05`.
   (b) Compute the mean and variance of the destination frequency for each
       :math:`s`. How does the mean shift with :math:`s`? Does the variance change?
   (c) What happens when :math:`|s|` is very large (say, :math:`s = 0.5`)?
       Does the normal approximation still look reasonable?

.. admonition:: Exercise 3: Dominance changes the dynamics

   Using the same parameters as Exercise 2, compare the backward mean
   :math:`\mu(x)` for :math:`h = 0, 0.5, 1` across all frequencies
   :math:`x \in (0, 1)`.

   (a) At which frequency is the selection-driven shift largest for each :math:`h`?
   (b) Why does the recessive case (:math:`h = 0`) have its largest shift at high
       :math:`x`, while the dominant case (:math:`h = 1`) has it at low :math:`x`?
   (c) Verify that all three :math:`h` values give the same shift when
       :math:`x = 0.5`. Why is this true?


Solutions
=========

.. admonition:: Solution 1: Frequency bins and the boundary effect

   .. code-block:: python

      for K in [50, 100, 200, 450]:
          freqs, _, _ = build_frequency_bins(K)
          z_bins, z_cdf = build_normal_cdf_lookup()

          spacing_near_0 = np.diff(freqs[:5])
          idx_half = K // 2
          spacing_near_half = np.diff(freqs[idx_half-2:idx_half+3])

          print(f"\nK = {K}:")
          print(f"  First 5 bins: {np.round(freqs[:5], 8)}")
          print(f"  Spacing near 0:   {np.round(spacing_near_0, 8)}")
          print(f"  Spacing near 0.5: {np.round(spacing_near_half, 6)}")

          logP, lo, hi = build_transition_matrix_fast(
              freqs, 10000.0, s=0.02, z_bins=z_bins, z_cdf=z_cdf)
          nnz = sum(hi[i] - lo[i] for i in range(K))
          print(f"  Nonzero entries: {nnz} / {K*K} = {100*nnz/(K*K):.1f}%")

   The boundary spacing shrinks as :math:`K` increases, giving finer resolution
   near fixation/loss. The sparsity increases with :math:`K` because each row's
   nonzero band width (in bin indices) grows only as :math:`\sqrt{K}`, while the
   total row length grows linearly.

.. admonition:: Solution 2: Selection shifts the transition distribution

   .. code-block:: python

      K = 100
      freqs, _, _ = build_frequency_bins(K)
      z_bins, z_cdf = build_normal_cdf_lookup()
      N_haploid = 20000.0

      i_half = np.argmin(np.abs(freqs - 0.5))
      print(f"Starting bin: x[{i_half}] = {freqs[i_half]:.4f}")

      for s_val in [-0.05, 0.0, 0.01, 0.05]:
          logP, _, _ = build_transition_matrix_fast(
              freqs, N_haploid, s=s_val, z_bins=z_bins, z_cdf=z_cdf)
          row = np.exp(logP[i_half, :])
          mean_freq = np.sum(freqs * row)
          var_freq = np.sum((freqs - mean_freq)**2 * row)
          print(f"  s = {s_val:+.2f}: mean = {mean_freq:.6f}, "
                f"var = {var_freq:.8f}, shift = {mean_freq - freqs[i_half]:+.6f}")

   **What happens:**

   - The mean shifts proportionally to :math:`s` (positive :math:`s` shifts the
     backward mean downward; negative :math:`s` shifts it upward).
   - The variance is nearly identical for all values of :math:`s` -- it depends
     on :math:`x` and :math:`N`, not on :math:`s`.
   - For very large :math:`|s|` (like 0.5), the normal approximation breaks down
     because the mean shift per generation can be comparable to the standard
     deviation. In practice, :math:`|s| < 0.1` is the regime where CLUES works
     well.

.. admonition:: Solution 3: Dominance changes the dynamics

   .. code-block:: python

      x_vals = np.linspace(0.01, 0.99, 200)
      s = 0.05

      for h_val, label in [(0.0, "Recessive"), (0.5, "Additive"), (1.0, "Dominant")]:
          shifts = np.array([backward_mean(x, s, h=h_val) - x for x in x_vals])
          peak_x = x_vals[np.argmin(shifts)]  # most negative shift
          print(f"{label} (h={h_val}): peak shift at x = {peak_x:.2f}, "
                f"shift = {shifts.min():.6f}")

      # Check that all h values agree at x = 0.5
      x_test = 0.5
      for h_val in [0.0, 0.5, 1.0]:
          mu = backward_mean(x_test, s, h=h_val)
          print(f"h = {h_val}, x = 0.5: mu = {mu:.8f}, shift = {mu - x_test:+.8f}")

   **(a)** Recessive: peak shift near :math:`x \approx 0.8`. Dominant: peak shift
   near :math:`x \approx 0.2`. Additive: peak shift near :math:`x \approx 0.5`.

   **(b)** For the recessive case, selection only acts on :math:`DD` homozygotes
   (frequency :math:`x^2`), so the selective effect is proportional to :math:`x^2`.
   This is largest when :math:`x` is high. For the dominant case, selection acts
   on both :math:`DD` and :math:`AD` (total frequency :math:`1 - (1-x)^2`),
   which is already large when :math:`x` is small.

   **(c)** At :math:`x = 0.5`, the heterozygote frequency :math:`2x(1-x) = 0.5`
   regardless of :math:`h`. The homozygote frequencies are also :math:`0.25` each.
   At this symmetric point, the dominance coefficient doesn't matter because the
   average fitness effect over all genotypes is the same.
