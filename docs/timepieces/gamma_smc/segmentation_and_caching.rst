.. _gamma_smc_segmentation_caching:

================================
Segmentation and Caching
================================

   *The regulator: the practical engineering that keeps the mechanism running fast and stable.*

The previous chapters described the mathematical components of Gamma-SMC: the
gamma approximation (:ref:`gamma_smc_gamma_approximation`), the flow field
(:ref:`gamma_smc_flow_field`), and the forward-backward algorithm
(:ref:`gamma_smc_forward_backward`). In principle, these are sufficient to
run inference. In practice, two engineering innovations make Gamma-SMC
*ultrafast*: **segmentation with caching** (which skips over long stretches
of identical observations) and **entropy clipping** (which prevents the gamma
approximation from drifting into invalid parameter regions).

In a mechanical watch, the regulator is the mechanism that ensures the
escapement beats at a consistent rate -- not too fast, not too slow, even as
the mainspring tension changes. In Gamma-SMC, segmentation and caching
regulate the computational load (handling the many homozygous positions
efficiently), and entropy clipping regulates the approximation quality
(preventing drift that would produce nonsensical results).


Segmentation: Exploiting Sparsity
====================================

The human genome has roughly 1 heterozygous site per 1,000 base pairs. This
means that 99.9% of positions are homozygous -- long stretches of :math:`Y_i = 0`
separated by rare :math:`Y_i = 1` observations. Processing each position
individually would be wasteful: most of the work goes into applying the flow
field and emission update at positions where the observation is always the same.

.. admonition:: Biology Aside -- Why genomic data is so sparse

   Two randomly chosen human chromosomes differ at about 1 in every 1,000
   nucleotides -- a reflection of the relatively small effective population
   size of humans (~10,000-20,000) and the low per-base mutation rate
   (~1.2 × 10\ :sup:`-8` per generation). This means that for a 3-billion-base
   genome, only about 3 million positions are heterozygous in a typical
   diploid individual. The vast majority of the genome is identical between
   the two haplotypes. This extreme sparsity is a universal feature of
   within-species variation in most organisms (though the ratio varies -- it
   is ~1/500 in *Drosophila* and ~1/2,000 in some plants). Gamma-SMC turns
   this biological fact into a computational advantage: instead of processing
   3 billion positions, it processes only ~3 million segments.

Gamma-SMC exploits this sparsity by **segmenting** the genome: consecutive
positions with the same observation type (homozygous or missing) are grouped
into a single segment, which is handled by a single cached lookup.

A segment consists of:

1. A stretch of :math:`n_\text{miss}` missing positions (from the genomic mask)
2. A stretch of :math:`n_\text{hom}` homozygous positions
3. A single observation at the end (heterozygous or the last position in the
   segment)

The key insight is that the effect of :math:`n` consecutive identical
observations can be **precomputed**: if we know how :math:`(\alpha, \beta)`
changes after one step of "flow field + hom emission," then we can compute the
effect of :math:`n` such steps by repeated composition. This composition is
precomputed for each grid point and cached.

.. code-block:: python

   import numpy as np

   def segment_observations(observations):
       """Segment a sequence into (n_miss, n_hom, final_obs) tuples.

       Consecutive missing and homozygous positions are grouped together.
       Each segment ends at a heterozygous site or the end of the sequence.

       Parameters
       ----------
       observations : list of int
           Observation at each position: 1 (het), 0 (hom), -1 (missing).

       Returns
       -------
       segments : list of (int, int, int)
           Each tuple is (n_miss, n_hom, final_obs).
       """
       segments = []
       n_miss = 0
       n_hom = 0

       for y in observations:
           if y == -1:  # missing
               n_miss += 1
           elif y == 0:  # hom
               n_hom += 1
           else:  # het (y == 1): close the current segment
               segments.append((n_miss, n_hom, 1))
               n_miss = 0
               n_hom = 0

       # Final segment (may end with hom or missing)
       if n_miss > 0 or n_hom > 0:
           segments.append((n_miss, n_hom, 0))

       return segments

   # Demonstrate: typical genomic pattern (sparse hets)
   obs = [-1]*3 + [0]*50 + [1] + [0]*200 + [1] + [0]*100
   segs = segment_observations(obs)
   total_positions = sum(nm + nh + (1 if fo == 1 else 0) for nm, nh, fo in segs)
   print(f"Sequence: {len(obs)} positions -> {len(segs)} segments")
   for i, (nm, nh, fo) in enumerate(segs):
       label = "het" if fo == 1 else "hom"
       print(f"  Segment {i}: {nm} miss, {nh} hom, ends with {label}")
   print(f"Speedup: {len(obs)}/{len(segs)} = {len(obs)/len(segs):.0f}x")


Locus Skipping via Caching
=============================

The caching strategy works as follows:

**Preprocessing (per parameter set).** For each element of the flow field grid
:math:`(l_\mu, l_C)` and for each number of steps :math:`k` from 1 to a
maximum cache size :math:`K`:

1. Starting from :math:`(l_\mu, l_C)`, apply the flow field once
   (transition), then apply the homozygous emission update (add :math:`\theta`
   to :math:`\beta`). Record the result.

2. Repeat step 1 from the result, for a total of :math:`k` times. Record the
   final :math:`(l_\mu^{(k)}, l_C^{(k)})`.

3. Do the same for missing observations (apply the flow field once but
   *skip* the emission update, since missing observations carry no
   information).

This produces two sets of cached flow fields:

- :math:`\mathcal{F}_\text{hom}^{(k)}(l_\mu, l_C)`: the result of :math:`k`
  consecutive hom steps
- :math:`\mathcal{F}_\text{miss}^{(k)}(l_\mu, l_C)`: the result of :math:`k`
  consecutive missing steps

**Inference.** During the forward pass, when encountering a segment with
:math:`n_\text{miss}` missing positions and :math:`n_\text{hom}` homozygous
positions:

1. Look up :math:`\mathcal{F}_\text{miss}^{(n_\text{miss})}` at the current
   :math:`(l_\mu, l_C)` to skip all missing positions at once.

2. Look up :math:`\mathcal{F}_\text{hom}^{(n_\text{hom})}` at the resulting
   :math:`(l_\mu, l_C)` to skip all homozygous positions at once.

3. Apply the heterozygous emission update at the end of the segment.

Each segment, regardless of how many positions it spans, is handled by
**two cache lookups and one emission update** -- :math:`O(1)` per segment.

.. admonition:: Parameter dependence of caches

   Unlike the flow field :math:`\mathcal{F}` (which is parameter-independent),
   the caches depend on :math:`\theta` and :math:`\rho`:

   - :math:`\theta` enters through the emission step (hom adds :math:`\theta`
     to :math:`\beta`).
   - :math:`\rho` enters because the flow field displacement is scaled by
     :math:`\rho` (recall :math:`\alpha' = \alpha + u \cdot \rho`).

   Therefore, the caches must be **recomputed** whenever :math:`\theta` or
   :math:`\rho` changes. However, this recomputation is fast (it amounts to
   composing known flow field lookups) and is done as a preprocessing step
   before the forward pass.


Handling Long Segments
========================

When :math:`n_\text{hom}` or :math:`n_\text{miss}` exceeds the maximum cache
size :math:`K`, the segment is processed in chunks: apply the :math:`K`-step
cache, then handle the remainder recursively. For example, a stretch of 150
homozygous positions with :math:`K = 100` would be handled as one
100-step cache lookup followed by one 50-step cache lookup.


Entropy Clipping
==================

The gamma approximation at the transition step is imperfect. Over many
consecutive transition steps (long homozygous or missing stretches), small
errors can accumulate, causing the gamma parameters to drift into regions
where the approximation breaks down.

The specific failure mode is an **entropy increase**: the differential entropy
of the gamma posterior exceeds the entropy of the prior. This is unphysical
-- observing data should reduce uncertainty, not increase it.

.. admonition:: Plain-language summary -- What entropy clipping prevents

   Entropy measures how spread out (uncertain) a probability distribution is.
   The prior -- our belief before seeing any data -- has a certain amount of
   uncertainty. After observing data, we should know *more*, not less. If the
   gamma approximation accumulates small errors over thousands of homozygous
   positions, those errors can push the posterior into an impossible state
   where it is *more uncertain than the prior*. Entropy clipping catches this
   before it happens: whenever the posterior becomes too diffuse, it is
   pulled back to the edge of the valid region. The mean TMRCA estimate is
   preserved (we don't change our best guess), but the excess uncertainty
   is trimmed away. If the entropy
were allowed to exceed the prior entropy, the combination of forward and
backward passes could produce invalid gamma parameters (specifically,
:math:`b + b' - 1 < 0`, which is not a valid rate parameter).

**The entropy bound.** The differential entropy of
:math:`\text{Gamma}(\alpha, \beta)` is:

.. math::

   h(\alpha, \beta) = \alpha - \ln \beta + \ln \Gamma(\alpha) + (1 - \alpha) \psi(\alpha)

For the exponential prior :math:`\text{Gamma}(1, 1)`, this evaluates to:

.. math::

   h(1, 1) = 1

Any legitimate posterior must have :math:`h(\alpha, \beta) \leq 1`. When this
bound is violated after a transition step, Gamma-SMC **clips** the posterior
back to the boundary of the valid region.

**The clipping rule.** When entropy exceeds the threshold:

1. Fix the **mean** :math:`\mu_\Gamma = \alpha / \beta` (the best estimate of
   the TMRCA should not change).
2. **Reduce the CV** until the entropy equals the threshold.

In :math:`(l_\mu, l_C)` coordinates, this means keeping :math:`l_\mu` fixed
and reducing :math:`l_C` to the maximum value consistent with
:math:`h(l_\mu, l_C) = 1`. Since the entropy is monotonically increasing in
:math:`l_C` (for fixed :math:`l_\mu` and :math:`l_C \leq 0`), the maximum
valid :math:`l_C` can be found by bisection.

In practice, this maximum is precomputed over a grid of :math:`l_\mu` values
and interpolated during inference.

.. code-block:: python

   from scipy.special import gammaln, digamma

   def gamma_entropy(alpha, beta):
       """Differential entropy of Gamma(alpha, beta).

       Parameters
       ----------
       alpha : float
           Shape parameter (alpha >= 1).
       beta : float
           Rate parameter.

       Returns
       -------
       h : float
           Differential entropy.
       """
       return (alpha - np.log(beta) + gammaln(alpha)
               + (1 - alpha) * digamma(alpha))

   def entropy_clip(alpha, beta, h_max=1.0, tol=1e-8):
       """Clip gamma parameters so that entropy does not exceed h_max.

       Keeps the mean alpha/beta fixed and reduces the CV (increases alpha)
       until the entropy equals h_max.

       Parameters
       ----------
       alpha, beta : float
           Current gamma parameters.
       h_max : float
           Maximum allowed entropy (default 1.0, the prior entropy).
       tol : float
           Bisection tolerance.

       Returns
       -------
       alpha_clipped, beta_clipped : float
           Clipped parameters with h <= h_max.
       """
       if gamma_entropy(alpha, beta) <= h_max:
           return alpha, beta  # no clipping needed

       mean = alpha / beta  # fix the mean

       # Bisect on alpha: increasing alpha decreases entropy (for fixed mean)
       lo, hi = alpha, 1e6
       for _ in range(100):
           mid = (lo + hi) / 2
           b_mid = mid / mean
           if gamma_entropy(mid, b_mid) > h_max:
               lo = mid
           else:
               hi = mid
           if hi - lo < tol:
               break

       alpha_new = (lo + hi) / 2
       beta_new = alpha_new / mean
       return alpha_new, beta_new

   # Verify: prior Gamma(1, 1) has entropy exactly 1
   print(f"Entropy of Gamma(1, 1) = {gamma_entropy(1.0, 1.0):.6f}")
   print(f"Entropy of Gamma(5, 5) = {gamma_entropy(5.0, 5.0):.6f}")

   # Simulate a case where entropy exceeds the threshold
   # Gamma(1.01, 0.8) has entropy > 1 (beta < 1 means too diffuse)
   a, b = 1.01, 0.8
   h_before = gamma_entropy(a, b)
   a_clip, b_clip = entropy_clip(a, b)
   h_after = gamma_entropy(a_clip, b_clip)
   print(f"\nBefore clip: Gamma({a}, {b}), entropy = {h_before:.4f}")
   print(f"After clip:  Gamma({a_clip:.4f}, {b_clip:.4f}), "
         f"entropy = {h_after:.4f}")
   print(f"Mean preserved: {a/b:.4f} -> {a_clip/b_clip:.4f}")

.. admonition:: Why does entropy clipping guarantee valid combination?

   If :math:`h(\alpha, \beta) \leq 1` and :math:`\alpha \geq 1`, then
   :math:`\beta \geq 1`. This follows from the monotonicity of the entropy
   in :math:`\alpha`:

   .. math::

      \frac{\partial h}{\partial \alpha} = 1 + (1 - \alpha) \psi^{(1)}(\alpha) > 0

   where :math:`\psi^{(1)}` is the trigamma function (always positive). So
   entropy is increasing in :math:`\alpha`. Setting :math:`\alpha = 1` gives
   :math:`h = 1 - \ln \beta`, and requiring :math:`h \leq 1` gives
   :math:`\beta \geq 1`.

   Therefore, both the forward and backward distributions satisfy
   :math:`\beta \geq 1`, guaranteeing :math:`b + b' - 1 \geq 1 > 0`. The
   combined posterior :math:`\text{Gamma}(a + a' - 1, b + b' - 1)` is always
   valid.


The Complete Forward Pass Algorithm
=======================================

Putting it all together, the Gamma-SMC forward pass for a single pair of
haplotypes is:

.. code-block:: text

   Initialize: (l_mu, l_C) = (0, 0)  [prior: Gamma(1, 1)]

   For each segment (n_miss, n_hom, final_obs):
       1. Look up cached miss flow: (l_mu, l_C) <- F_miss^(n_miss)(l_mu, l_C)
       2. Clip if entropy exceeds threshold
       3. Look up cached hom flow:  (l_mu, l_C) <- F_hom^(n_hom)(l_mu, l_C)
       4. Clip if entropy exceeds threshold
       5. Apply emission update for final_obs:
          - If het: alpha += 1, beta += theta
          - If hom: beta += theta
          - If missing: no change
       6. Convert back to (l_mu, l_C)
       7. If this is an output position, record (l_mu, l_C)

At steps 1 and 3, if :math:`n_\text{miss}` or :math:`n_\text{hom}` is 0, the
step is skipped. If it exceeds the cache size :math:`K`, it is handled in
chunks.

The backward pass uses the same algorithm on the reversed sequence, with
an additional flow field step when aligning backward densities to output
positions (see :ref:`gamma_smc_forward_backward`).

.. code-block:: python

   def gamma_smc_forward_segmented(observations, theta, rho, flow_field,
                                    h_max=1.0):
       """Gamma-SMC forward pass with segmentation and entropy clipping.

       This is the full algorithm combining all four components:
       segmentation, caching (via repeated flow field application),
       flow field queries, and entropy clipping.

       Parameters
       ----------
       observations : list of int
           1 (het), 0 (hom), -1 (missing) at each position.
       theta : float
           Scaled mutation rate.
       rho : float
           Scaled recombination rate.
       flow_field : object
           Flow field with .query(l_mu, l_C) method.
       h_max : float
           Entropy threshold for clipping.

       Returns
       -------
       results : list of (float, float)
           (alpha, beta) at each het position.
       """
       segments = segment_observations(observations)
       alpha, beta = 1.0, 1.0  # prior: Gamma(1, 1)
       results = []

       for n_miss, n_hom, final_obs in segments:
           # Step 1: skip missing positions (flow field only, no emission)
           for _ in range(n_miss):
               l_mu = np.log10(alpha / beta)
               l_C = np.log10(1.0 / np.sqrt(alpha))
               dl_mu, dl_C = flow_field.query(l_mu, l_C)
               l_mu += rho * dl_mu
               l_C += rho * dl_C
               alpha = 10.0 ** (-2 * l_C)
               beta = alpha * 10.0 ** (-l_mu)

           # Step 2: entropy clip after missing block
           alpha, beta = entropy_clip(alpha, beta, h_max)

           # Step 3: skip homozygous positions (flow field + hom emission)
           for _ in range(n_hom):
               l_mu = np.log10(alpha / beta)
               l_C = np.log10(1.0 / np.sqrt(alpha))
               dl_mu, dl_C = flow_field.query(l_mu, l_C)
               l_mu += rho * dl_mu
               l_C += rho * dl_C
               alpha = 10.0 ** (-2 * l_C)
               beta = alpha * 10.0 ** (-l_mu)
               beta += theta  # hom emission

           # Step 4: entropy clip after hom block
           alpha, beta = entropy_clip(alpha, beta, h_max)

           # Step 5: emission update for the final observation
           if final_obs == 1:  # het
               alpha += 1
               beta += theta
           elif final_obs == 0:  # trailing hom (end of sequence)
               beta += theta

           results.append((alpha, beta))

       return results

   # Demonstrate on a realistic-scale pattern
   np.random.seed(42)
   n_sites = 10000
   obs = [0] * n_sites
   het_positions = sorted(np.random.choice(n_sites, size=10, replace=False))
   for pos in het_positions:
       obs[pos] = 1

   class ZeroFlow:
       def query(self, l_mu, l_C): return 0.0, 0.0

   segments = segment_observations(obs)
   results = gamma_smc_forward_segmented(obs, 0.001, 0.0004, ZeroFlow())
   print(f"Input: {n_sites} positions, {sum(obs)} hets")
   print(f"Segments: {len(segments)} (one per het + final)")
   print(f"Speedup: {n_sites}/{len(segments)} = "
         f"{n_sites/len(segments):.0f}x fewer operations")
   a_final, b_final = results[-1]
   print(f"Final posterior: Gamma({a_final:.1f}, {b_final:.4f}), "
         f"mean = {a_final/b_final:.2f}")


Performance Characteristics
==============================

The segmentation and caching strategy gives Gamma-SMC its performance
advantage:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Operation
     - Cost per segment
     - Cost per position
   * - Cache lookup (miss)
     - :math:`O(1)` (bilinear interpolation)
     - N/A (amortized over :math:`n_\text{miss}`)
   * - Cache lookup (hom)
     - :math:`O(1)` (bilinear interpolation)
     - N/A (amortized over :math:`n_\text{hom}`)
   * - Emission update (het)
     - :math:`O(1)` (parameter increment)
     - :math:`O(1)`
   * - Entropy clip
     - :math:`O(1)` (table lookup + interpolation)
     - N/A (amortized)

The total cost is proportional to the **number of segments**, not the number
of genomic positions. Since the number of segments equals the number of
heterozygous sites (plus a small number from mask boundaries), the effective
cost is :math:`O(H)` where :math:`H` is the number of heterozygous sites.
For typical human data, :math:`H \approx N / 1000`, giving a
:math:`\sim 1000\times` speedup over a naive per-position algorithm.


Summary
=========

The segmentation and caching system is the engineering that makes Gamma-SMC
practical:

- **Segmentation** groups consecutive identical observations into segments
  that can be processed as a unit.
- **Caching** precomputes the effect of multi-step transitions, reducing each
  segment to two cache lookups.
- **Entropy clipping** prevents approximation drift, guaranteeing that the
  forward-backward combination always produces a valid gamma posterior.

These three mechanisms -- the regulator of the Gamma-SMC watch -- transform
the :math:`O(N)` forward pass into an :math:`O(H)` algorithm that is
ultrafast in practice.

----

.. _gamma_smc_recap:

Recap: The Complete Gamma-SMC Timepiece
==========================================

You have now disassembled and rebuilt every gear in the Gamma-SMC mechanism
across four chapters:

- **The Gamma Approximation** (:ref:`gamma_smc_gamma_approximation`):
  Poisson-gamma conjugacy for exact emission updates, and PDE-based gamma
  projection for approximate transition updates. This was the escapement --
  the mathematical insight that makes the continuous-state approach possible.

- **The Flow Field** (:ref:`gamma_smc_flow_field`): A precomputed 2D vector
  field that encodes the transition dynamics, evaluated once and reused for
  any dataset. This was the gear train -- the precision-machined transmission.

- **The Forward-Backward CS-HMM** (:ref:`gamma_smc_forward_backward`): The
  forward algorithm sweeps left-to-right, the backward algorithm is a forward
  pass on the reversed sequence, and the combination gives
  :math:`\text{Gamma}(a + a' - 1, b + b' - 1)`. This was the mainspring --
  the engine that drives inference.

- **Segmentation and Caching** (this chapter): Locus skipping, cached
  multi-step lookups, and entropy clipping that make the algorithm ultrafast
  and numerically stable. This was the regulator -- the engineering that keeps
  the mechanism running smoothly.

Gamma-SMC demonstrates that the pairwise TMRCA inference problem -- the same
problem PSMC solves with discretized time and iterative EM -- can be solved
in a single forward-backward pass with :math:`O(1)` cost per position, no
time discretization, and no parameter iteration. The trade-off is that
Gamma-SMC assumes constant population size and provides per-site posteriors
rather than a demographic history. But as a computational mechanism, it is
remarkably elegant: two numbers :math:`(\alpha, \beta)` glide continuously
through a precomputed flow field, accumulating evidence at each position,
until the full posterior emerges at every site along the genome.

*The clock ticks continuously. And you can read the time at every position.*
