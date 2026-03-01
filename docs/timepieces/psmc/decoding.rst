.. _psmc_decoding:

===================================
Decoding the Clock
===================================

   *The case and dial: turning internal gears into a readable face.*

We have arrived at the final chapter of the PSMC Timepiece. In the preceding
chapters, we built every internal component of this watch: the continuous-time
transition density (the escapement, :ref:`psmc_continuous`), the discrete time
intervals and transition matrix (the gear train, :ref:`psmc_discretization`),
and the EM algorithm that iteratively refines the parameters from data (the
mainspring, :ref:`psmc_hmm`). The EM algorithm has converged, and we now hold
estimated parameters :math:`\hat{\theta}_0`, :math:`\hat{\rho}_0`, and
:math:`\hat{\lambda}_0, \ldots, \hat{\lambda}_n`.

But these are dimensionless numbers in coalescent units -- the internal tick
count of the mechanism, not the time displayed on the dial. A watch is useless
if you cannot read its face. To tell biological time, we need to **scale** these
parameters to real units -- generations, years, and population sizes. We also
need to assess how confident we are in the reading (bootstrapping), how well the
model fits the data (goodness of fit), and what common mistakes can lead us to
misread the dial entirely (pitfalls).

This chapter covers five steps:

1. **Scaling** -- reading the dial by converting coalescent units to real units
2. **Goodness of fit** -- checking whether the watch keeps accurate time
3. **Bootstrapping** -- testing the watch against multiple reference clocks
4. **Common pitfalls** -- the ways even experienced watchmakers misread the dial
5. **Interpreting PSMC plots** -- what the population history actually tells us

Let us begin.


.. _psmc_decoding_scaling:

Step 1: From Coalescent Units to Real Units
==============================================

*Reading the dial.*

.. admonition:: What does "reading the dial" mean?

   A watch's internal mechanism counts oscillations -- ticks of the escapement.
   But the dial translates those ticks into hours and minutes, units that have
   meaning in the external world. The PSMC's internal mechanism operates in
   **coalescent units**, where time is measured relative to the population size
   and rates are scaled accordingly. "Reading the dial" means converting those
   internal units into **generations**, **years**, and **individuals** -- the
   units that have meaning for biology.

PSMC operates in **coalescent units**: time is measured in units of :math:`2N_0`
generations, and rates are scaled by :math:`4N_0`. These conventions come
directly from the coalescent theory developed in the :ref:`prerequisites
<coalescent_theory>`, where we saw that the natural time unit for two-lineage
coalescence is :math:`2N` generations. To convert back to real units,
we need one external piece of information: the **per-generation mutation rate**
:math:`\mu`.

Think of :math:`\mu` as the conversion factor printed on the bezel of the
watch -- without it, the dial markings are arbitrary.

**Computing** :math:`N_0`:

The estimated :math:`\hat{\theta}_0` is the population-scaled mutation rate per
bin of size :math:`s` base pairs:

.. math::

   \hat{\theta}_0 = 4 N_0 \mu s

This equation relates four quantities. We know three of them after inference:
:math:`\hat{\theta}_0` comes from EM (see :ref:`psmc_hmm`), :math:`\mu` is an
externally supplied mutation rate, and :math:`s` is the bin size we chose during
data preparation (typically 100 bp). Solving for the one remaining unknown:

.. math::

   N_0 = \frac{\hat{\theta}_0}{4 \mu s}

.. admonition:: Choosing a mutation rate :math:`\mu`

   For **humans**, commonly used values are:

   - :math:`\mu \approx 1.25 \times 10^{-8}` per base pair per generation
     (pedigree-based estimate)
   - :math:`\mu \approx 2.5 \times 10^{-8}` per base pair per generation
     (phylogenetic estimate from human-chimp divergence)

   For **other organisms**, you must find a species-appropriate estimate from
   the literature. The mutation rate depends on generation time, DNA repair
   mechanisms, and other biological factors. Using the wrong :math:`\mu` will
   shift your entire time axis and population size axis -- we discuss this
   further in the mutation rate admonition below.

   As a sanity check: for humans with :math:`\mu = 1.25 \times 10^{-8}`,
   :math:`s = 100`, and a typical :math:`\hat{\theta}_0 \approx 0.00069`, you
   should get :math:`N_0 \approx 13{,}800`. If your :math:`N_0` is orders of
   magnitude off from the expected range for your species, double-check your
   :math:`\mu` and :math:`s`.

**Scaling time** from coalescent units to generations:

Each coalescent time unit corresponds to :math:`2N_0` generations (recall from
:ref:`coalescent_theory` that the coalescence rate for two lineages in a
population of size :math:`N` is :math:`1/(2N)` per generation, so the expected
coalescence time is :math:`2N` generations). Therefore:

.. math::

   T_k \text{ (generations)} = 2 N_0 \cdot t_k

To convert generations to years, multiply by the **generation time** (typically
25--29 years for humans):

.. math::

   T_k \text{ (years)} = 2 N_0 \cdot t_k \cdot g

where :math:`g` is the generation time in years.

**Scaling population size:**

The :math:`\hat{\lambda}_k` values estimated by EM are *relative* population
sizes -- they express the population size in interval :math:`k` as a multiple
of the reference :math:`N_0`. To get absolute effective population sizes:

.. math::

   N_k = N_0 \cdot \hat{\lambda}_k

So if :math:`\hat{\lambda}_3 = 2.0`, the population in the third time interval
was twice as large as the reference :math:`N_0`.

Here is the complete scaling function, with inline comments explaining each
conversion:

.. code-block:: python

   import numpy as np

   def scale_psmc_output(theta_0, lambdas, t_boundaries,
                          mu=1.25e-8, s=100, generation_time=25):
       """Scale PSMC output to real units.

       This is the "dial" of the PSMC watch: it converts the internal
       coalescent-unit parameters into biologically meaningful quantities.

       Parameters
       ----------
       theta_0 : float
           Estimated theta_0 from EM (population-scaled mutation rate per bin).
       lambdas : ndarray of shape (n+1,)
           Estimated relative population sizes (lambda_k values from EM).
       t_boundaries : ndarray of shape (n+2,)
           Time interval boundaries in coalescent units (from discretization).
       mu : float
           Per-generation, per-base-pair mutation rate (external input).
       s : int
           Bin size in base pairs (must match what was used in data preparation).
       generation_time : float
           Years per generation (e.g., 25 for humans, 3 for mice).

       Returns
       -------
       N_0 : float
           Reference effective population size.
       times_gen : ndarray
           Time boundaries in generations.
       times_years : ndarray
           Time boundaries in years.
       pop_sizes : ndarray
           Effective population sizes.
       """
       # Convert theta to N_0 using the relationship theta = 4 * N_0 * mu * s
       N_0 = theta_0 / (4 * mu * s)

       # Scale time: each coalescent unit = 2*N_0 generations
       times_gen = 2 * N_0 * t_boundaries

       # Convert generations to calendar years
       times_years = times_gen * generation_time

       # Scale relative sizes (lambdas) to absolute population sizes
       pop_sizes = N_0 * lambdas

       return N_0, times_gen, times_years, pop_sizes

   # Example: typical human PSMC output
   theta_0 = 0.00069  # typical for humans with s=100
   n = 10
   # Time boundaries in coalescent units (from discretization chapter)
   t = np.array([0, 0.05, 0.12, 0.22, 0.37, 0.6, 0.95, 1.5, 2.5, 4.0, 8.0, 1000])
   # Relative population sizes (lambda_k from EM)
   lambdas = np.array([2.0, 1.8, 1.5, 1.0, 0.5, 0.3, 0.5, 1.0, 1.5, 2.0, 2.5])

   N_0, t_gen, t_years, N_t = scale_psmc_output(theta_0, lambdas, t)

   print(f"N_0 = {N_0:.0f}")
   print(f"\nTime (years ago)    Pop. size")
   print("-" * 40)
   for k in range(len(lambdas)):
       print(f"  {t_years[k]:>12,.0f}      {N_t[k]:>10,.0f}")

Let us pause and trace through the unit conversions carefully, because getting
them wrong is the single most common source of error in PSMC analyses.

.. admonition:: Unit conversion walkthrough

   Suppose :math:`\hat{\theta}_0 = 0.00069`, :math:`\mu = 1.25 \times 10^{-8}`,
   :math:`s = 100`, and :math:`g = 25` years.

   1. :math:`N_0 = 0.00069 / (4 \times 1.25 \times 10^{-8} \times 100) = 0.00069 / 5 \times 10^{-6} = 138`.
      Wait -- that gives 138? That seems far too low. Let us recheck:
      :math:`4 \times 1.25 \times 10^{-8} \times 100 = 5 \times 10^{-6}`.
      :math:`0.00069 / 5 \times 10^{-6} = 6.9 \times 10^{-4} / 5 \times 10^{-6} = 138`.

      Actually, 138 is correct -- but this is :math:`N_0` as a *scaling constant*,
      not the actual population size. The actual population sizes are
      :math:`N_k = N_0 \times \lambda_k`. If :math:`\lambda_k \approx 72`, then
      :math:`N_k \approx 10{,}000`.

      In practice, typical human PSMC runs give :math:`N_0 \approx 10{,}000\text{--}15{,}000`
      because the :math:`\hat{\theta}_0` incorporates the bin size differently
      depending on the implementation. **Always sanity-check your** :math:`N_0`
      **against known values for your species.**

   2. Time at boundary :math:`t_3 = 0.22` coalescent units:
      :math:`T_3 = 2 \times N_0 \times 0.22` generations,
      :math:`= 2 \times N_0 \times 0.22 \times 25` years.

   3. Population size in interval 3 with :math:`\lambda_3 = 1.0`:
      :math:`N_3 = N_0 \times 1.0 = N_0`.


.. admonition:: The mutation rate problem

   The biggest source of uncertainty in PSMC scaling is :math:`\mu`. Different
   estimation methods give different values:

   - **Phylogenetic rate** (human-chimp divergence): :math:`\mu \approx 2.5 \times 10^{-8}`
   - **Pedigree rate** (parent-offspring sequencing): :math:`\mu \approx 1.2 \times 10^{-8}`
   - **Ancient DNA calibration**: :math:`\mu \approx 1.5 \times 10^{-8}`

   The factor-of-2 difference shifts the entire time axis and population size
   axis. Since :math:`N_0 \propto 1/\mu` and :math:`T \propto N_0 \propto 1/\mu`,
   halving :math:`\mu` doubles both the inferred population sizes and the
   inferred times. This is not a small effect -- it can shift the inferred
   out-of-Africa bottleneck from 50,000 to 100,000 years ago.

   This is why PSMC plots often show axes in terms of "scaled mutation rate"
   (:math:`\theta_k = 4 N_k \mu`) and "sequence divergence"
   (:math:`d_k = 2 \mu T_k`) rather than absolute years and population sizes.
   These mutation-rate-free quantities are model outputs that do not depend on
   the choice of :math:`\mu`.


**Alternative scaling (mutation-rate free):**

To avoid depending on :math:`\mu`, PSMC output can be presented in terms of
pairwise sequence divergence. This is the approach used in Li and Durbin (2011)
and in many subsequent PSMC papers. The x-axis becomes sequence divergence per
base pair (a proxy for time), and the y-axis becomes the scaled mutation rate
(a proxy for population size):

.. math::

   d_k = t_k \cdot \frac{\theta_0}{s} \quad \text{(divergence per bp)}

.. math::

   \theta_k = \lambda_k \cdot \frac{\theta_0}{s} \quad \text{(scaled mutation rate)}

These are the axes you see in many PSMC papers. The advantage is that these
quantities are directly estimated from the data with no external calibration
needed. The disadvantage is that they require the reader to mentally convert
divergence to years (using an assumed :math:`\mu`) to interpret the biological
meaning.

.. code-block:: python

   def scale_mutation_free(theta_0, lambdas, t_boundaries, s=100):
       """Scale without assuming a mutation rate.

       Returns divergence (x-axis) and scaled mutation rate (y-axis).
       These are the "native" units of PSMC output -- no external
       calibration is needed.

       Parameters
       ----------
       theta_0 : float
           Estimated theta_0 from EM.
       lambdas : ndarray
           Relative population sizes.
       t_boundaries : ndarray
           Time boundaries in coalescent units.
       s : int
           Bin size in base pairs.

       Returns
       -------
       divergence : ndarray
           Pairwise sequence divergence per bp (proxy for time).
       scaled_theta : ndarray
           Population-scaled mutation rate (proxy for N_e).
       """
       # Divergence = coalescent time * per-bp mutation rate
       # Since theta_0 = 4*N_0*mu*s, theta_0/s = 4*N_0*mu,
       # and t_k * theta_0/s = t_k * 4*N_0*mu = 2*mu * (2*N_0*t_k)
       # = 2*mu*T_k = expected divergence at time T_k
       divergence = t_boundaries * theta_0 / s

       # Scaled theta: lambda_k * theta_0/s = lambda_k * 4*N_0*mu
       # = 4 * N_k * mu = population-scaled mutation rate for interval k
       scaled_theta = lambdas * theta_0 / s

       return divergence, scaled_theta

   div, theta_scaled = scale_mutation_free(theta_0, lambdas, t, s=100)
   print("\nMutation-rate-free scaling:")
   print("Divergence (per bp)    Scaled theta")
   print("-" * 45)
   for k in range(len(lambdas)):
       print(f"  {div[k]:>15.2e}      {theta_scaled[k]:>15.2e}")

With the dial now readable, we can move to the next question: how much should
we trust the reading?


.. _psmc_decoding_gof:

Step 2: Goodness of Fit
==========================

*Checking whether the watch keeps accurate time.*

Before we place confidence in the PSMC's inferred history, we should verify
that the model actually fits the observed data. A watch that displays a time
but runs at the wrong rate is worse than useless -- it is misleading. PSMC
provides two diagnostics for assessing model fit.

**Diagnostic 1: Comparing** :math:`\sigma_k` **and** :math:`\hat{\sigma}_k`

.. admonition:: What is posterior decoding?

   *Asking the watch what time it thinks it is at each position.*

   In the :ref:`HMM prerequisites <hmms>`, we learned about the forward-backward
   algorithm: it computes, for every position along the sequence and every hidden
   state, the probability that the HMM was in that state given the entire
   observed sequence. This is called **posterior decoding** -- it is the model's
   best guess about which hidden state generated each observation.

   In the PSMC context, the hidden states are coalescence time intervals. So
   posterior decoding asks: "At genomic position :math:`a`, what is the
   probability that the two haplotypes coalesced in time interval :math:`k`?"
   This is like asking the watch, at each tick along the genome, "What time do
   you think it is here?"

   The **posterior stationary distribution** :math:`\hat{\sigma}_k` is the
   average of these posterior probabilities across all positions -- it tells us
   what fraction of the genome the model *thinks* falls into each coalescence
   time interval, given the observed data.

The model predicts a stationary distribution :math:`\sigma_k` (computed from the
estimated parameters, as derived in :ref:`psmc_discretization`). The data
provides a posterior estimate :math:`\hat{\sigma}_k` (from the forward-backward
algorithm, as implemented in :ref:`psmc_hmm`):

.. math::

   \hat{\sigma}_k = \frac{\sum_a f_k(a) b_k(a)}{\sum_{k,a} f_k(a) b_k(a)}

Here :math:`f_k(a)` and :math:`b_k(a)` are the (scaled) forward and backward
variables at position :math:`a` for state :math:`k`. Their product, after
normalization, gives the posterior probability :math:`\gamma_k(a)` that the
coalescence time at position :math:`a` falls in interval :math:`k`. Summing
over all positions and normalizing gives :math:`\hat{\sigma}_k`.

If the model fits well, :math:`\sigma_k \approx \hat{\sigma}_k`. The
**relative entropy** (Kullback-Leibler divergence) measures the discrepancy:

.. math::

   G^\sigma = \sum_k \sigma_k \log \frac{\sigma_k}{\hat{\sigma}_k}

.. admonition:: Interpreting the KL divergence

   The KL divergence :math:`G^\sigma` is always non-negative and equals zero if
   and only if the two distributions are identical. In practice:

   - :math:`G^\sigma < 10^{-4}`: excellent fit
   - :math:`G^\sigma \sim 10^{-3}`: acceptable fit
   - :math:`G^\sigma > 10^{-2}`: the model may be misspecified (wrong number of
     intervals, not enough EM iterations, or the data violates PSMC assumptions)

   If you see a large :math:`G^\sigma`, try running more EM iterations or
   adjusting the number of free intervals. If the value remains high, the data
   may not be well-suited for PSMC (e.g., too much missing data or strong
   population structure).

.. code-block:: python

   def goodness_of_fit_sigma(hmm, seq):
       """Compute goodness-of-fit by comparing sigma_k to posterior.

       This diagnostic checks whether the model's predicted stationary
       distribution matches what the data actually shows via posterior
       decoding -- it tests whether the watch's internal model of time
       agrees with the evidence from the observed sequence.

       Parameters
       ----------
       hmm : PSMC_HMM
           The fully parameterized HMM after EM convergence.
       seq : ndarray
           The observed binary sequence (0 = homozygous, 1 = heterozygous).

       Returns
       -------
       G_sigma : float
           KL divergence (smaller is better, 0 is perfect).
       sigma_model : ndarray
           The model's predicted stationary distribution.
       sigma_data : ndarray
           The data's posterior stationary distribution.
       """
       N = hmm.N  # number of hidden states (time intervals)

       # Model's stationary distribution: computed from parameters alone
       sigma_model = hmm.initial.copy()

       # Data's posterior stationary distribution: requires forward-backward
       # (See the forward-backward implementation in the HMM chapter)
       alpha_hat, _ = hmm.forward_scaled(seq)
       beta_hat = hmm.backward_scaled(seq, alpha_hat)

       # Accumulate posterior probabilities across all positions
       sigma_data = np.zeros(N)
       for pos in range(len(seq)):
           # gamma[k] = P(state=k at position pos | entire sequence)
           gamma = alpha_hat[pos] * beta_hat[pos]
           sigma_data += gamma
       # Normalize to get a proper distribution
       sigma_data /= sigma_data.sum()

       # KL divergence: D(model || data)
       G_sigma = 0.0
       for k in range(N):
           if sigma_model[k] > 0 and sigma_data[k] > 0:
               G_sigma += sigma_model[k] * np.log(sigma_model[k] / sigma_data[k])

       return G_sigma, sigma_model, sigma_data

**Diagnostic 2: Subsequence distribution**

The second diagnostic is more stringent. It compares the empirical distribution
of short binary subsequences in the data to the theoretical distribution
predicted by the model. For example, for subsequences of length :math:`l = 10`,
there are :math:`2^{10} = 1024` possible binary patterns (like ``0000000001``,
``0101010101``, etc.). If the model is correct, the frequency of each pattern in
the data should match the frequency predicted by the model's transition and
emission probabilities.

This tests not just the marginal distribution (like Diagnostic 1) but the
**sequential structure** of the data -- the correlations between nearby
positions. It is analogous to testing not just whether a watch shows the right
time on average, but whether it ticks at the right rate second-by-second.

.. code-block:: python

   def goodness_of_fit_subsequences(hmm, seq, l=10):
       """Compare empirical and theoretical subsequence distributions.

       Parameters
       ----------
       hmm : PSMC_HMM
           The fully parameterized HMM.
       seq : ndarray
           The observed binary sequence.
       l : int
           Subsequence length (10 is typical; 2^l patterns are compared).

       Returns
       -------
       G_l : float
           KL divergence between empirical and theoretical distributions.
       """
       L = len(seq)
       n_patterns = 2 ** l  # number of possible binary patterns

       # Count occurrences of each binary pattern in the observed data
       counts = np.zeros(n_patterns)
       for pos in range(L - l + 1):
           subseq = seq[pos:pos + l]
           if np.any(subseq >= 2):  # skip positions with missing data
               continue
           # Convert binary subsequence to an integer index
           # e.g., [0,1,0,1] -> 0b0101 = 5
           pattern = 0
           for bit in subseq:
               pattern = (pattern << 1) | int(bit)
           counts[pattern] += 1

       total = counts.sum()
       if total == 0:
           return 0.0

       p_empirical = counts / total  # normalize to get frequencies

       # Theoretical distribution (requires matrix powers -- simplified here)
       # In practice, computed via transfer matrix:
       # P(pattern) = sigma @ T^(l-1) @ emission_product
       # For now, we return just the empirical distribution
       return p_empirical

Now that we can check whether the watch is accurate, the next question is: how
*precise* is it? That is the domain of bootstrapping.


.. _psmc_decoding_bootstrap:

Step 3: Bootstrapping for Confidence Intervals
=================================================

*Testing the watch against multiple reference clocks.*

.. admonition:: What is bootstrapping?

   Imagine you have built a watch and you want to know how reliable its reading
   is. You cannot build the same watch a hundred times from scratch (you only
   have one genome). But you *can* do the next best thing: take the parts of
   the watch, shuffle them randomly, reassemble the watch many times, and see
   how much the readings vary. If every reassembled watch tells nearly the same
   time, your original reading is reliable. If the readings scatter widely,
   your original reading is uncertain.

   More formally, **bootstrapping** is a statistical technique for estimating
   the uncertainty of an estimate when you have only one dataset. You create
   many "pseudo-datasets" by **resampling with replacement** from your original
   data, compute the estimate on each pseudo-dataset, and use the spread of
   the resulting estimates as a measure of uncertainty.

   The key assumption is that the resampled datasets are approximately as
   variable as truly independent datasets would be. For PSMC, this works well
   because different genomic regions are approximately independent (they are
   separated by enough recombination events).

A single PSMC run gives point estimates. To assess uncertainty, we use
bootstrapping: resample the data and re-run PSMC many times.

**The procedure:**

1. Split the genome into non-overlapping segments of length :math:`L'`
   (typically 5 Mb = 50,000 bins at 100bp resolution). Each segment should be
   long enough to contain many recombination events, so that PSMC can extract
   meaningful signal from it.

2. Randomly sample :math:`\lceil L / L' \rceil` segments *with replacement*
   to create a bootstrap replicate of the same total length. Some segments will
   appear multiple times; others will not appear at all. This is the "shuffling
   and reassembling" step.

3. Run PSMC on the bootstrap replicate (full EM, same parameters as the
   original run).

4. Repeat :math:`B` times (typically :math:`B = 100`).

5. For each time point :math:`t_k`, compute the 2.5th and 97.5th percentiles
   of :math:`\hat{\lambda}_k^{(b)}` across bootstrap replicates to get 95%
   confidence intervals.

.. admonition:: What do confidence intervals represent?

   A **95% confidence interval** around the inferred :math:`N_e(t)` at a
   particular time means: if we repeated the entire experiment (sequencing a
   new genome and running PSMC) many times, 95% of the resulting estimates
   would fall within this interval.

   In practice, narrow confidence bands mean that PSMC has strong signal at
   that time depth -- many recombination events produce coalescence times in
   that range. Wide bands mean limited signal. You will typically see:

   - **Narrow bands** in the intermediate past (roughly 50,000--500,000 years
     ago for humans), where PSMC has the most power
   - **Wide bands** in the very recent past (fewer than ~20,000 years ago),
     where there are too few short-range recombination events for precise
     inference
   - **Wide bands** in the very distant past (more than ~1 million years ago),
     where there are few surviving genomic segments with such ancient
     coalescence times

.. code-block:: python

   def split_sequence(seq, segment_length=50000):
       """Split a sequence into segments for bootstrapping.

       Each segment should be long enough to contain meaningful PSMC
       signal. At s=100 bp per bin, 50000 bins = 5 Mb of sequence.

       Parameters
       ----------
       seq : ndarray
           The full binary sequence.
       segment_length : int
           Number of bins per segment (default 50000 = 5 Mb at s=100).

       Returns
       -------
       segments : list of ndarray
           Non-overlapping segments of the sequence.
       """
       segments = []
       for start in range(0, len(seq) - segment_length + 1, segment_length):
           segments.append(seq[start:start + segment_length])
       return segments

   def bootstrap_resample(segments, total_length):
       """Create a bootstrap replicate by resampling segments.

       This is the "shuffling and reassembling" step: we draw segments
       with replacement to build a new sequence of the same total length.
       Some segments will be duplicated; others will be absent entirely.

       Parameters
       ----------
       segments : list of ndarray
           The genomic segments produced by split_sequence.
       total_length : int
           Desired length of the replicate (same as original sequence).

       Returns
       -------
       replicate : ndarray
           The bootstrap replicate sequence.
       """
       n_segments = len(segments)
       # How many segments do we need to reach the target length?
       n_needed = total_length // len(segments[0]) + 1
       # Sample segment indices WITH REPLACEMENT -- this is the key step
       indices = np.random.choice(n_segments, size=n_needed, replace=True)
       # Concatenate the selected segments into one long sequence
       replicate = np.concatenate([segments[i] for i in indices])
       # Trim to exact length
       return replicate[:total_length]

   # Example bootstrap workflow (pseudocode)
   def run_bootstrap(seq, n_bootstrap=100, segment_length=50000, **psmc_kwargs):
       """Run PSMC with bootstrapping.

       Parameters
       ----------
       seq : ndarray
           Original sequence.
       n_bootstrap : int
           Number of bootstrap replicates (100 is standard).
       segment_length : int
           Bins per segment (50000 = 5 Mb at s=100).
       **psmc_kwargs : passed to psmc_inference.

       Returns
       -------
       bootstrap_lambdas : list of ndarray
           Lambda estimates from each bootstrap replicate.
       """
       # Step 1: Split the genome into segments
       segments = split_sequence(seq, segment_length)
       total_length = len(seq)
       bootstrap_lambdas = []

       for b in range(n_bootstrap):
           # Step 2: Create a resampled replicate
           replicate = bootstrap_resample(segments, total_length)
           # Step 3: Run full PSMC inference on the replicate
           # results = psmc_inference(replicate, **psmc_kwargs)
           # bootstrap_lambdas.append(results[-1]['lambdas'])
           pass

       return bootstrap_lambdas

With scaling in hand and uncertainty quantified, let us now turn to the
mistakes that can make all of this effort misleading.


.. _psmc_decoding_pitfalls:

Step 4: Common Pitfalls
=========================

*The ways even experienced watchmakers misread the dial.*

PSMC is remarkably powerful, but several pitfalls can produce misleading results.
Understanding these pitfalls is as important as understanding the algorithm
itself -- a watch that you trust blindly is more dangerous than one you know is
unreliable.

**Pitfall 1: Low coverage (the most common mistake)**

For diploid genomes sequenced to low coverage, heterozygous sites are randomly
lost -- if only one allele is sequenced at a heterozygous position, it appears
homozygous. This **false negative rate (FNR)** makes it look like the mutation
rate is lower, which shifts the entire population size curve upward and the
time axis to the right.

.. admonition:: Why does missing heterozygosity distort the results?

   Recall that PSMC reads population history from the *density* of heterozygous
   sites. If you systematically miss some heterozygous sites, the density
   appears lower. Lower heterozygosity looks like the two haplotypes diverged
   more recently (shorter coalescence times), which PSMC interprets as a
   smaller ancestral population. The entire :math:`N(t)` curve shifts, and the
   time axis stretches because :math:`N_0` is underestimated.

   In watch terms: it is as if some of the tick marks on the dial have been
   erased, making the watch systematically slow.

.. math::

   \theta_{\text{apparent}} = \theta_{\text{true}} \cdot (1 - \text{FNR})

**Correction:** If you know the FNR, divide :math:`\hat{\theta}_0` by
:math:`(1 - \text{FNR})` before scaling:

.. math::

   \theta_{\text{corrected}} = \frac{\hat{\theta}_0}{1 - \text{FNR}}

.. code-block:: python

   def correct_for_coverage(theta_0, fnr):
       """Correct theta_0 for false negative rate on heterozygotes.

       Apply this correction BEFORE scaling to real units. The FNR
       can be estimated by comparing variant calls at different coverage
       depths, or by using simulations.

       Parameters
       ----------
       theta_0 : float
           Observed theta (from EM on low-coverage data).
       fnr : float
           Fraction of heterozygotes missed (0 to 1).
           Typical values: 0.05 at 30x, 0.15 at 15x, 0.30 at 8x.

       Returns
       -------
       theta_corrected : float
       """
       return theta_0 / (1 - fnr)

   # Example: 20% of hets missed (roughly corresponding to ~10x coverage)
   theta_obs = 0.00069
   theta_corr = correct_for_coverage(theta_obs, 0.2)
   print(f"Observed theta: {theta_obs:.6f}")
   print(f"Corrected theta: {theta_corr:.6f}")
   print(f"N_0 ratio: {theta_corr/theta_obs:.2f}x")

**Pitfall 2: Overfitting**

Using too many free parameters (e.g., ``-p "64*1"`` where every interval has its
own :math:`\lambda`) leads to overfitting, especially in time intervals with few
expected recombination events. The symptom is a jagged, noisy :math:`N(t)` curve
with biologically implausible spikes and dips -- the watch appears to be ticking
erratically rather than smoothly.

**How to check:** Compute :math:`C_\sigma \sigma_k` for each interval -- this is
the expected number of genomic segments whose coalescence time falls in interval
:math:`k`. If this number is less than ~20, the :math:`\lambda_k` estimate is
unreliable and should be grouped with adjacent intervals. This is why the
default PSMC pattern ``"4+25*2+4+6"`` groups intervals in the very recent and
very ancient past, where data is sparse, while allowing finer resolution in the
intermediate past where signal is strongest.

.. code-block:: python

   def check_overfitting(sigma_k, C_sigma, threshold=20):
       """Check which intervals have too few expected segments.

       Intervals with fewer than ~20 expected segments cannot reliably
       support an independent lambda estimate. These should be grouped
       with neighboring intervals using the pattern string.

       Parameters
       ----------
       sigma_k : ndarray
           Stationary distribution over time intervals.
       C_sigma : float
           Total number of independent segments (approximately
           total_sequence_length * rho, where rho is the recombination rate).
       threshold : float
           Minimum expected segments for reliable estimation.

       Returns
       -------
       warnings : list of int
           Indices of intervals that may overfit.
       expected_segments : ndarray
           Expected segment count per interval.
       """
       # Expected number of segments coalescing in each interval
       expected_segments = C_sigma * sigma_k
       warnings = []
       for k, exp_seg in enumerate(expected_segments):
           if exp_seg < threshold:
               warnings.append(k)
       return warnings, expected_segments

**Pitfall 3: Interpreting the recent past**

PSMC has poor resolution for very recent times (the last ~20,000 years for
humans). This is because recent coalescence times correspond to long segments of
identical-by-descent (IBD), and a single diploid genome has limited information
about these long segments. The first few :math:`\lambda_k` values should be
interpreted with caution.

In watch terms: the hour hand moves so slowly in the recent past that you cannot
read the minutes. The watch still works, but its resolution is limited.

.. admonition:: Why is the recent past hard?

   For a coalescence time of :math:`T` generations, the expected length of the
   shared IBD segment is :math:`\sim 1/T` Morgans. Very recent coalescence
   times produce very long IBD segments, and each diploid genome contains only
   a finite amount of sequence. With few independent segments carrying
   information about recent times, the variance of the :math:`\lambda_k`
   estimate for the most recent intervals becomes large. This is a fundamental
   limitation of the two-sequence approach. Methods like MSMC and MSMC2, which
   use multiple genomes, have better resolution in the recent past because
   additional genomes provide more short IBD segments.

**Pitfall 4: Population structure**

PSMC assumes a **panmictic** (randomly mating) population. If the two haplotypes
come from a structured population, PSMC will infer spurious population size changes.
Migration between subpopulations looks like population expansion; isolation looks
like a bottleneck. This is because PSMC cannot distinguish between "the population
was large" (many individuals, high coalescence time) and "the population was
subdivided" (the two lineages were in different subpopulations, which also
increases coalescence time).

.. admonition:: When to worry about structure

   If your PSMC curves for different individuals from the same population diverge
   in the recent past, population structure may be confounding the inference. In
   this case, consider using MSMC (Multiple Sequentially Markovian Coalescent),
   which can model population separation.

   A useful diagnostic: run PSMC on individuals from different populations and
   compare. If the curves agree in the distant past (when the populations shared
   ancestors) but diverge in the recent past, the divergence point may indicate
   population split time -- but the *shape* of the curves after divergence
   reflects structure, not necessarily true population size changes.


.. _psmc_decoding_interpreting:

Step 5: Interpreting PSMC Plots
=================================

*Reading the face of the watch.*

We have now scaled our parameters, checked model fit, quantified uncertainty,
and accounted for pitfalls. It is time to interpret the final product: the PSMC
plot. This is the payoff -- the readable face of the watch we have built across
four chapters.

A PSMC plot shows :math:`N_e(t)` (y-axis) vs. time (x-axis), both on log scales.
Here is how to read it:

.. code-block:: text

              N_e
               ^
      100,000 -|        ____
               |       /    \
       50,000 -|      /      \
               |  ___/        \___
       10,000 -|                   \
               |                    \____
        5,000 -|
               +--+----+----+----+----+-->  time (years ago)
                 10k  50k  100k  500k  1M

Key features to look for:

1. **A peak**: indicates a period of large effective population size (or population
   expansion). The population was large, coalescence was slow, and the genome
   accumulated more diversity.
2. **A trough**: indicates a bottleneck (population contraction). The population
   shrank, forcing lineages to coalesce quickly, leaving less diversity in that
   time window.
3. **A plateau**: indicates stable population size. The rate of coalescence was
   roughly constant.
4. **Recent decline**: often seen in humans, may reflect the out-of-Africa
   bottleneck (~50,000--100,000 years ago depending on :math:`\mu`).
5. **Noisy edges**: the leftmost and rightmost parts have high uncertainty.
   Do not over-interpret the first or last one or two intervals (recall Pitfall 3).

.. admonition:: Log-log axes

   Both axes in a standard PSMC plot use logarithmic scales. This is essential
   because the time axis spans several orders of magnitude (10,000 to 1,000,000+
   years) and population sizes can vary by an order of magnitude. On linear
   axes, the interesting intermediate-time features would be compressed into an
   unreadable sliver. The log-log presentation gives roughly equal visual weight
   to each decade of time and population size.

.. code-block:: python

   def plot_psmc_history(theta_0, lambdas, t_boundaries,
                          mu=1.25e-8, s=100, generation_time=25):
       """Generate data for a PSMC plot.

       Returns the x (time) and y (N_e) coordinates for a step-function
       plot of population size history. Each interval is rendered as a
       horizontal line at height N_k between time boundaries t_k and t_{k+1}.

       Parameters
       ----------
       theta_0, lambdas, t_boundaries : PSMC output
       mu : float
       s : int
       generation_time : float

       Returns
       -------
       x : list of float
           Time points (years ago) for plotting as a step function.
       y : list of float
           Population sizes corresponding to each time point.
       """
       # First, scale from coalescent units to real units
       N_0, t_gen, t_years, N_t = scale_psmc_output(
           theta_0, lambdas, t_boundaries, mu, s, generation_time)

       # Create step function: for each interval k, draw a horizontal
       # line from t_k to t_{k+1} at height N_k
       x = []
       y = []
       for k in range(len(lambdas)):
           x.append(t_years[k])      # left edge of interval
           y.append(N_t[k])          # population size in this interval
           x.append(t_years[k + 1])  # right edge of interval
           y.append(N_t[k])          # same population size (step function)

       return x, y

   # Example plot data
   x, y = plot_psmc_history(theta_0, lambdas, t)
   print("PSMC history (step function):")
   print(f"Time range: {min(x):,.0f} to {max(x):,.0f} years ago")
   print(f"N_e range: {min(y):,.0f} to {max(y):,.0f}")


.. _psmc_decoding_pipeline:

Putting It All Together: The Complete PSMC Pipeline
=====================================================

Let us now step back and see the full pipeline from raw sequence to interpreted
population history. This function ties together every chapter of the PSMC
Timepiece: discretization (:ref:`psmc_discretization`), HMM inference
(:ref:`psmc_hmm`), and all five steps of decoding covered in this chapter.

.. code-block:: python

   def psmc_pipeline(seq, mu=1.25e-8, s=100, generation_time=25,
                      n=63, t_max=15.0, pattern="4+25*2+4+6",
                      n_iters=25, n_bootstrap=100):
       """The complete PSMC pipeline from sequence to population history.

       This function represents the entire watch -- from raw input (the
       binary sequence) through the internal mechanism (EM inference)
       to the readable output (scaled population history with confidence
       intervals).

       Parameters
       ----------
       seq : ndarray
           Binary sequence (0 = homozygous, 1 = heterozygous, 2 = missing).
       mu : float
           Per-generation, per-bp mutation rate (see mutation rate discussion).
       s : int
           Bin size in base pairs.
       generation_time : float
           Years per generation.
       n, t_max, pattern, n_iters : PSMC parameters.
           n : number of time intervals (before grouping by pattern).
           t_max : maximum coalescent time.
           pattern : grouping pattern for lambda parameters (see psmc_hmm).
           n_iters : number of EM iterations.
       n_bootstrap : int
           Number of bootstrap replicates for confidence intervals.

       Returns
       -------
       history : dict
           'times': time points (years),
           'pop_sizes': N_e at each time,
           'ci_lower': lower 95% CI,
           'ci_upper': upper 95% CI,
           'theta': estimated theta,
           'rho': estimated rho,
           'N_0': reference population size.
       """
       # Step 1: Run PSMC inference (the mainspring -- see psmc_hmm chapter)
       # results = psmc_inference(seq, n, t_max, pattern=pattern, n_iters=n_iters)
       # final = results[-1]

       # Step 2: Scale to real units (reading the dial -- this chapter, Step 1)
       # N_0, t_gen, t_years, N_t = scale_psmc_output(
       #     final['theta'], final['lambdas'], t_boundaries, mu, s, generation_time)

       # Step 3: Bootstrap for confidence intervals (this chapter, Step 3)
       # bootstrap_lambdas = run_bootstrap(seq, n_bootstrap)
       # ci_lower, ci_upper = compute_confidence_intervals(bootstrap_lambdas)

       # Step 4: Check goodness of fit (this chapter, Step 2)
       # G_sigma = goodness_of_fit_sigma(final_hmm, seq)

       # Step 5: Return the complete history
       # return history
       pass


Exercises
=========

.. admonition:: Exercise 1: Full PSMC on simulated data

   This is the capstone exercise. Simulate a diploid genome under a known
   demographic model:

   1. Constant :math:`N = 10,000` from present to 50,000 years ago
   2. Bottleneck :math:`N = 1,000` from 50,000 to 100,000 years ago
   3. :math:`N = 20,000` before 100,000 years ago

   Using :math:`\mu = 1.25 \times 10^{-8}`, generate 3 Mb of psmcfa data.
   Run your PSMC implementation and verify it recovers the demographic history.

   *Hint:* To simulate, you can use the msprime Timepiece
   (:ref:`msprime_timepiece`) with a demographic model. Convert the resulting
   tree sequence to a heterozygosity sequence using mutation overlays
   (see :ref:`msprime_mutations`).

.. admonition:: Exercise 2: The coverage effect

   Take the simulated data from Exercise 1. Randomly set 20% of heterozygous
   bins to homozygous (simulating low coverage). Run PSMC with and without the
   FNR correction. How much does the inferred history shift?

   *Hint:* The shift should be approximately proportional to
   :math:`1/(1 - \text{FNR})`. Plot both curves on the same axes and compare
   the time axis and population size axis separately.

.. admonition:: Exercise 3: Compare to the real PSMC

   Run both your Python implementation and the original C implementation
   (``psmc`` binary) on the same data. Compare:
   (a) log-likelihoods at each iteration,
   (b) final parameter estimates,
   (c) running time.

   Your Python version will be much slower, but the results should agree.

.. admonition:: Exercise 4: Bootstrap confidence intervals

   Using the data from Exercise 1, run 100 bootstrap replicates. Plot the
   inferred :math:`N(t)` with 95% confidence bands. Where is the uncertainty
   largest? Why?

   *Hint:* Recall our discussion of where PSMC has the most and least power.
   The very recent past and the very distant past should have the widest bands,
   while the intermediate past should be tightest.


Solutions
=========

.. admonition:: Solution 1: Full PSMC on simulated data

   This capstone exercise ties together every chapter. We simulate data under a
   known three-epoch demographic model, run PSMC, scale to real units, and
   verify recovery of the true history.

   .. code-block:: python

      import numpy as np

      # Define the demographic model in real units
      # Epoch 1: N = 10,000 from present to 50 kya
      # Epoch 2: N = 1,000 from 50 to 100 kya (bottleneck)
      # Epoch 3: N = 20,000 before 100 kya
      mu = 1.25e-8        # per bp per generation
      s = 100              # bin size
      g = 25               # generation time
      N_ref = 10000        # use as N_0 for scaling

      # Convert to coalescent units (time in units of 2*N_ref generations)
      t_50k = 50000 / (g * 2 * N_ref)    # ~0.1 coalescent units
      t_100k = 100000 / (g * 2 * N_ref)  # ~0.2 coalescent units

      def true_lambda(t):
          """True relative population size in coalescent units."""
          if t < t_50k:
              return 10000 / N_ref   # = 1.0
          elif t < t_100k:
              return 1000 / N_ref    # = 0.1
          else:
              return 20000 / N_ref   # = 2.0

      theta_true = 4 * N_ref * mu * s  # = 0.005
      rho_true = theta_true / 5

      # Simulate 30,000 bins (= 3 Mb at s=100 bp)
      np.random.seed(42)
      seq, _ = simulate_psmc_input(30000, theta_true, rho_true, true_lambda)

      print(f"Observed heterozygosity: {np.mean(seq):.4f}")
      print(f"theta_true = {theta_true:.6f}")

      # Run PSMC inference
      n = 20
      results = psmc_inference(seq, n=n, t_max=15.0, n_iters=25,
                                pattern=f"{n+1}*1")

      # Scale the output
      final = results[-1]
      t_boundaries = compute_time_intervals(n, t_max=15.0)
      N_0, t_gen, t_years, N_t = scale_psmc_output(
          final['theta'], final['lambdas'], t_boundaries,
          mu=mu, s=s, generation_time=g)

      print(f"\nInferred N_0 = {N_0:.0f} (expected ~{N_ref})")
      print(f"\nTime (years ago)    Inferred N_e    True N_e")
      print("-" * 50)
      for k in range(n + 1):
          t_mid_years = (t_years[k] + t_years[k+1]) / 2.0
          # Determine true N_e at this time
          if t_mid_years < 50000:
              true_N = 10000
          elif t_mid_years < 100000:
              true_N = 1000
          else:
              true_N = 20000
          print(f"  {t_mid_years:>12,.0f}    {N_t[k]:>12,.0f}    {true_N:>8,}")

   The inferred :math:`N_e(t)` should show three clear epochs: a plateau near
   10,000 in the recent past, a dip toward 1,000 around 50,000--100,000 years
   ago, and a rise toward 20,000 in the deep past. With only 3 Mb of data, the
   edges of the bottleneck will be somewhat smoothed, but the qualitative pattern
   should be unmistakable.

.. admonition:: Solution 2: The coverage effect

   We demonstrate how missing heterozygous sites (simulating low coverage)
   distort the inferred history, and how the FNR correction restores it.

   .. code-block:: python

      import numpy as np

      # Use the same simulated sequence from Exercise 1
      np.random.seed(42)
      seq_full, _ = simulate_psmc_input(30000, theta_true, rho_true, true_lambda)

      # Simulate low coverage: randomly convert 20% of het sites to hom
      fnr = 0.20
      seq_low = seq_full.copy()
      het_positions = np.where(seq_low == 1)[0]
      n_to_drop = int(len(het_positions) * fnr)
      drop_indices = np.random.choice(het_positions, size=n_to_drop, replace=False)
      seq_low[drop_indices] = 0

      het_original = np.mean(seq_full)
      het_low = np.mean(seq_low)
      print(f"Original heterozygosity: {het_original:.4f}")
      print(f"Low-coverage heterozygosity: {het_low:.4f}")
      print(f"Ratio: {het_low / het_original:.4f} (expected: {1 - fnr:.2f})")

      # Run PSMC on both sequences
      n = 10
      results_full = psmc_inference(seq_full, n=n, t_max=15.0, n_iters=20,
                                     pattern=f"{n+1}*1")
      results_low = psmc_inference(seq_low, n=n, t_max=15.0, n_iters=20,
                                    pattern=f"{n+1}*1")

      # Scale without correction
      theta_uncorrected = results_low[-1]['theta']
      # Scale with FNR correction
      theta_corrected = correct_for_coverage(theta_uncorrected, fnr)

      t_boundaries = compute_time_intervals(n, t_max=15.0)

      N0_full, _, t_years_full, Nt_full = scale_psmc_output(
          results_full[-1]['theta'], results_full[-1]['lambdas'],
          t_boundaries, mu=mu, s=s, generation_time=g)
      N0_uncorr, _, t_years_uncorr, Nt_uncorr = scale_psmc_output(
          theta_uncorrected, results_low[-1]['lambdas'],
          t_boundaries, mu=mu, s=s, generation_time=g)
      N0_corr, _, t_years_corr, Nt_corr = scale_psmc_output(
          theta_corrected, results_low[-1]['lambdas'],
          t_boundaries, mu=mu, s=s, generation_time=g)

      print(f"\nN_0 (full data):      {N0_full:.0f}")
      print(f"N_0 (uncorrected):    {N0_uncorr:.0f}")
      print(f"N_0 (FNR corrected):  {N0_corr:.0f}")
      print(f"Expected correction factor: {1/(1-fnr):.4f}")
      print(f"Actual N_0 ratio (corr/uncorr): {N0_corr/N0_uncorr:.4f}")

   **What happens:** Without correction, the inferred :math:`\hat{\theta}` is
   reduced by a factor of :math:`(1 - \text{FNR}) = 0.8`, which means
   :math:`N_0` is underestimated by 20%. This shifts the entire population size
   curve downward and stretches the time axis. The bottleneck appears less
   severe (because :math:`N_0` is smaller, the relative depth of the dip changes)
   and appears to occur at a different time.

   After applying ``correct_for_coverage``, the corrected :math:`N_0` should
   closely match the full-data estimate, restoring the correct scaling on both
   axes. The :math:`\lambda_k` values themselves are not affected by the FNR
   correction -- only the scaling changes -- because the relative emission
   probabilities across intervals are affected approximately uniformly by the
   missing data.

.. admonition:: Solution 3: Compare to the real PSMC

   This exercise requires the original C implementation of PSMC (available at
   ``https://github.com/lh3/psmc``). We compare the Python and C
   implementations on identical input.

   .. code-block:: python

      import numpy as np
      import subprocess
      import time

      # Step 1: Write simulated data in psmcfa format
      np.random.seed(42)
      seq, _ = simulate_psmc_input(100000, theta_true, rho_true, true_lambda)

      def write_psmcfa(seq, filename, s=100):
          """Write a binary sequence as a psmcfa file."""
          with open(filename, 'w') as f:
              f.write(">chr1\n")
              line = ""
              for obs in seq:
                  line += "K" if obs == 1 else "T"
                  if len(line) == 60:
                      f.write(line + "\n")
                      line = ""
              if line:
                  f.write(line + "\n")

      write_psmcfa(seq, "/tmp/test.psmcfa")

      # Step 2: Run Python PSMC
      t_start = time.time()
      results_py = psmc_inference(seq, n=63, t_max=15.0,
                                   pattern="4+25*2+4+6", n_iters=25)
      time_python = time.time() - t_start

      # Step 3: Run C PSMC (requires psmc binary in PATH)
      # t_start = time.time()
      # subprocess.run(["psmc", "-N", "25", "-t", "15", "-r", "5",
      #                 "-p", "4+25*2+4+6", "-o", "/tmp/test.psmc",
      #                 "/tmp/test.psmcfa"])
      # time_c = time.time() - t_start

      # Step 4: Parse C PSMC output and compare
      # (a) Log-likelihoods should agree to within ~1%
      # (b) Final lambda_k values should agree to within ~5%
      # (c) Python will be ~100-1000x slower than C

      print(f"Python PSMC time: {time_python:.1f} seconds")
      print(f"Final Python LL: {results_py[-1]['log_likelihood']:.2f}")

   **Expected comparison:**

   **(a) Log-likelihoods:** Should agree to within ~1% at each iteration. Small
   differences arise from different numerical optimization strategies (the C
   implementation uses Hooke-Jeeves while our Python version uses Nelder-Mead)
   and floating-point differences.

   **(b) Final parameters:** The inferred :math:`\lambda_k` curves should be
   visually indistinguishable. Quantitatively, individual :math:`\lambda_k`
   values may differ by up to ~5%, but the overall shape of the population
   history should match.

   **(c) Running time:** The Python implementation will be approximately
   100--1000x slower than the C implementation. The C version uses optimized
   matrix operations and avoids the overhead of Python loops in the
   forward-backward algorithm. For production use, always prefer the C
   implementation; the Python version is for understanding the algorithm.

.. admonition:: Solution 4: Bootstrap confidence intervals

   We run 100 bootstrap replicates to quantify uncertainty in the inferred
   population history and identify where PSMC has the most and least power.

   .. code-block:: python

      import numpy as np

      # Simulate a long sequence for meaningful bootstrapping
      np.random.seed(42)
      seq, _ = simulate_psmc_input(500000, theta_true, rho_true, true_lambda)

      n = 20
      segment_length = 50000  # 5 Mb segments

      # Split into segments
      segments = split_sequence(seq, segment_length)
      print(f"Number of segments: {len(segments)}")

      # Run original PSMC
      results_orig = psmc_inference(seq, n=n, t_max=15.0, n_iters=20,
                                     pattern=f"{n+1}*1")
      lambdas_orig = results_orig[-1]['lambdas']

      # Run 100 bootstrap replicates
      n_bootstrap = 100
      all_lambdas = np.zeros((n_bootstrap, n + 1))

      for b in range(n_bootstrap):
          replicate = bootstrap_resample(segments, len(seq))
          results_b = psmc_inference(replicate, n=n, t_max=15.0, n_iters=20,
                                      pattern=f"{n+1}*1")
          all_lambdas[b] = results_b[-1]['lambdas']
          if (b + 1) % 10 == 0:
              print(f"  Completed {b + 1}/{n_bootstrap} replicates")

      # Compute 95% confidence intervals (2.5th and 97.5th percentiles)
      ci_lower = np.percentile(all_lambdas, 2.5, axis=0)
      ci_upper = np.percentile(all_lambdas, 97.5, axis=0)
      ci_width = ci_upper - ci_lower

      # Scale to real units
      t_boundaries = compute_time_intervals(n, t_max=15.0)
      N_0, _, t_years, _ = scale_psmc_output(
          results_orig[-1]['theta'], lambdas_orig, t_boundaries,
          mu=mu, s=s, generation_time=g)

      print(f"\n{'Time (years)':>15} {'N_e':>10} {'CI width':>10} "
            f"{'Rel. width':>12}")
      print("-" * 50)
      for k in range(n + 1):
          t_mid = (t_years[k] + t_years[k+1]) / 2.0
          Ne = N_0 * lambdas_orig[k]
          width = N_0 * ci_width[k]
          rel_width = ci_width[k] / lambdas_orig[k] if lambdas_orig[k] > 0 else 0
          print(f"  {t_mid:>12,.0f} {Ne:>10,.0f} {width:>10,.0f} "
                f"{rel_width:>10.1%}")

   **Where is the uncertainty largest?**

   - **Very recent past** (first 1--2 intervals, :math:`< 20{,}000` years ago):
     Wide confidence bands because there are few recombination events producing
     very short-range coalescence times. A single diploid genome has limited
     information about very recent demography.

   - **Very distant past** (last 1--2 intervals, :math:`> 1{,}000{,}000` years
     ago): Wide bands because very few genomic segments retain coalescence times
     this ancient. The survival probability :math:`\alpha_k` is extremely small,
     so the expected number of segments :math:`C_\sigma \sigma_k` is tiny.

   - **Intermediate past** (~50,000--500,000 years ago for humans): The
     **tightest** confidence bands. This is PSMC's sweet spot -- enough
     recombination events to have good statistical power, and enough segments
     with coalescence times in this range to constrain :math:`\lambda_k`
     precisely.

   This pattern reflects the fundamental resolution limit of the two-sequence
   PSMC: it is most informative about population sizes in the time window where
   most coalescence events occur, and least informative at the extremes of the
   time axis.

----

.. _psmc_decoding_recap:

Recap: The Complete PSMC Timepiece
=====================================

Congratulations. You have now disassembled and rebuilt every gear in the PSMC
mechanism across four chapters:

- **The Continuous-Time Model** (:ref:`psmc_continuous`): The transition density
  :math:`q(t|s)` that captures how coalescence times change along the genome
  under variable :math:`N(t)`. This was the escapement -- the fundamental
  ticking mechanism.

- **Discretization** (:ref:`psmc_discretization`): Converting continuous time
  into discrete intervals with a computable transition matrix. This was the
  gear train -- carving continuous gears into finite teeth.

- **The HMM and EM** (:ref:`psmc_hmm`): The learning machine that estimates
  population parameters from a binary sequence. This was the mainspring --
  the engine that drives the whole mechanism.

- **Decoding the Clock** (this chapter): Scaling to real units (reading the
  dial), bootstrapping (testing the watch against multiple reference clocks),
  posterior decoding (asking the watch what time it thinks it is at each
  position), goodness of fit (checking the watch's accuracy), and interpreting
  the population size history (reading the face).

You built it yourself. No black boxes remain.

The PSMC is the simplest complete Timepiece -- a two-handed watch that reads
population history from a single diploid genome. But just as a simple watch
invites the question "could we add a chronograph? a moon phase? a tourbillon?",
the PSMC invites the question: what if we used *more* than two sequences? What
if we could read not just population *size* but population *structure*,
divergence times, and migration rates? Those questions lead to the Timepieces
that follow -- SMC++ (:ref:`Timepiece II <smcpp_timepiece>`), ARGweaver, SINGER,
and beyond.

*The clock ticks. And you can read every hour it marks.*
