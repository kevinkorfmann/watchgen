.. _clues_emissions:

===========================================
Emission Probabilities
===========================================

   *The gear train: how coalescence events and ancient genotypes constrain
   the allele frequency at each moment in time.*

The transition matrix (previous chapter) describes how the allele frequency
*evolves*. But the frequency trajectory is hidden -- we never observe it directly.
What we *do* observe are two kinds of evidence:

1. **Coalescence events in the gene tree** -- when lineages merge, and how many
   remain at each point in time.
2. **Ancient genotype likelihoods** -- direct (but noisy) observations of the
   allele in individuals sampled from the past.

This chapter derives the emission probabilities that connect these observations to
the hidden allele frequency. By the end, you will have built the complete
observation model of the CLUES HMM.


Step 1: Coalescent Emissions
==============================

The gene tree at the focal SNP has two kinds of lineages: those carrying the
**derived allele** (with the mutation) and those carrying the **ancestral allele**
(without it). CLUES observes the coalescence times of these lineages and asks:
given allele frequency :math:`x_t` at time :math:`t`, how likely are the observed
coalescence events?

The coalescent among derived-allele lineages
----------------------------------------------

Consider :math:`n_D` lineages carrying the derived allele. At time :math:`t`, if
the allele frequency is :math:`x_t`, these lineages are confined to a sub-population
of effective haploid size :math:`x_t \cdot N(t)`, where :math:`N(t)` is the total
haploid effective population size.

From the :ref:`coalescent theory prerequisite <coalescent_theory>`, :math:`k`
lineages in a population of size :math:`M` coalesce at rate :math:`\binom{k}{2}/M`.
For derived lineages at frequency :math:`x_t`:

.. math::

   \text{Coalescence rate of } k \text{ derived lineages} = \frac{\binom{k}{2}}{x_t \cdot N(t)}

.. admonition:: Why divide by :math:`x_t \cdot N(t)` and not :math:`2 \cdot x_t \cdot N(t)`?

   The convention matters. In CLUES, :math:`N(t)` is the **haploid** effective
   population size (twice the diploid :math:`N_e`). The derived lineages are
   confined to the fraction :math:`x_t` of this haploid population, giving an
   effective sub-population size of :math:`x_t \cdot N(t)`. With :math:`k`
   lineages, there are :math:`\binom{k}{2}` pairs, and each pair coalesces at
   rate :math:`1/(x_t \cdot N(t))`.

   In the CLUES2 code, you'll see :math:`k(k-1)/4` instead of
   :math:`\binom{k}{2} = k(k-1)/2`. This is because the code uses *diploid*
   :math:`N` (which is half the haploid size), so the factor of 2 is absorbed:
   :math:`\binom{k}{2}/(x \cdot N_{\text{haploid}}) = k(k-1)/2 \cdot 1/(x \cdot 2N_{\text{diploid}})
   = k(k-1)/(4 \cdot x \cdot N_{\text{diploid}})`.

Between two consecutive coalescence events, the number of surviving lineages is
constant, and the waiting time until the next coalescence follows an **exponential
distribution**. The density of observing a coalescence at time :math:`t_i` (given
:math:`k` remaining lineages and frequency :math:`x`) over an epoch :math:`[t_0, t_1]` is:

.. math::

   f(t_i) = \frac{\binom{k}{2}}{x \cdot N} \cdot
   \exp\left(-\frac{\binom{k}{2}}{x \cdot N} \cdot (t_i - t_{i-1})\right)

where :math:`t_{i-1}` is the time of the previous coalescence (or the start of the
epoch). After the last coalescence event, the surviving lineages must *not* coalesce
before the end of the epoch -- this is a **survival probability**:

.. math::

   P(\text{no coalescence in } [t_d, t_1]) =
   \exp\left(-\frac{\binom{k'}{2}}{x \cdot N} \cdot (t_1 - t_d)\right)

where :math:`k'` is the number of remaining lineages and :math:`t_d` is the last
coalescence time.

**Putting it together.** For :math:`d` coalescence events at times
:math:`t_1 < t_2 < \cdots < t_d` among :math:`n` lineages during epoch
:math:`[e_0, e_1]` at frequency :math:`x` and population size :math:`N`:

.. math::

   \log P(\text{coal events}) = \sum_{i=1}^{d} \left[
   -\log(x) + \log\left(\frac{k_i(k_i - 1)}{4 N}\right)
   - \frac{k_i(k_i-1)}{4 x N} (t_i - t_{i-1})
   \right]
   - \frac{k'(k'-1)}{4 x N}(e_1 - t_d)

where :math:`k_i = n - i + 1` is the number of lineages just before the
:math:`i`-th coalescence, :math:`t_0 = e_0`, and :math:`k' = n - d`.

The same formula applies to ancestral lineages, replacing :math:`x` with
:math:`1 - x` (the ancestral allele frequency).

.. code-block:: python

   import numpy as np

   def log_coalescent_density(coal_times, n_lineages, epoch_start, epoch_end,
                               freq, N_diploid, ancestral=False):
       """Compute the log-probability of coalescence events in one epoch.

       This is the core emission computation for the CLUES HMM. It
       computes the probability of observing a specific set of coalescence
       times given the allele frequency and population size.

       Parameters
       ----------
       coal_times : ndarray
           Sorted coalescence times within this epoch. May be empty.
       n_lineages : int
           Number of lineages at the start of the epoch.
       epoch_start : float
           Start time of the epoch (generations).
       epoch_end : float
           End time of the epoch (generations).
       freq : float
           Derived allele frequency (0 < freq < 1).
       N_diploid : float
           Diploid effective population size.
       ancestral : bool
           If True, use ancestral frequency (1 - freq) instead.

       Returns
       -------
       log_prob : float
           Log-probability of the coalescence events.
       """
       if n_lineages <= 1:
           # No coalescence possible with 0 or 1 lineages
           return 0.0

       xi = (1.0 - freq) if ancestral else freq

       if xi * N_diploid == 0.0:
           # Impossible: lineages exist but frequency is 0
           return -1e20

       logp = 0.0
       prev_t = epoch_start
       k = n_lineages

       for t in coal_times:
           # k choose 2, divided by 4 because N is diploid
           # (equivalent to k(k-1)/2 divided by 2*N_diploid = N_haploid)
           kchoose2_over_4 = k * (k - 1) / 4.0
           rate = kchoose2_over_4 / (xi * N_diploid)

           # Exponential density: rate * exp(-rate * dt)
           # In log: log(rate) - rate * dt
           # But we split: -log(xi) accounts for the 1/xi factor in the rate
           dt = t - prev_t
           logp += -np.log(xi) - kchoose2_over_4 / (xi * N_diploid) * dt

           prev_t = t
           k -= 1

       # Survival probability: no further coalescences until epoch end
       if k >= 2:
           kchoose2_over_4 = k * (k - 1) / 4.0
           logp += -kchoose2_over_4 / (xi * N_diploid) * (epoch_end - prev_t)

       return logp

   # Example: 3 derived lineages, one coalescence at t=0.5 in epoch [0, 1]
   coal_times = np.array([0.5])
   log_prob = log_coalescent_density(
       coal_times, n_lineages=3, epoch_start=0.0, epoch_end=1.0,
       freq=0.3, N_diploid=10000.0, ancestral=False)
   print(f"Log-probability of coalescence: {log_prob:.4f}")

   # How does the probability change with frequency?
   print("\nCoalescence probability vs. derived allele frequency:")
   for freq in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
       lp = log_coalescent_density(
           coal_times, n_lineages=3, epoch_start=0.0, epoch_end=1.0,
           freq=freq, N_diploid=10000.0)
       print(f"  freq = {freq:.1f}: log P = {lp:.4f}")


The Mixed Lineage Problem
---------------------------

There is a subtlety at the root of the gene tree. The mutation that created the
derived allele sits on a specific branch. Below this branch (closer to the present),
the lineage carries the derived allele. Above it (further into the past), the
lineage is ancestral. At the exact time of the mutation, one lineage transitions
from derived to ancestral.

When we reach the point where only :math:`n_D = 1` derived lineage and
:math:`n_A = 1` ancestral lineage remain, these two lineages must coalesce as
*ancestral* lineages (because the single derived lineage's ancestor is also
ancestral). CLUES handles this by treating the coalescence as occurring among
:math:`n_A = 2` ancestral lineages when :math:`n_D = 1`.

.. code-block:: python

   def compute_coalescent_emissions(coal_times_der, coal_times_anc,
                                     n_der, n_anc, epoch_start, epoch_end,
                                     freqs, N_diploid):
       """Compute coalescent emission probabilities for all frequency bins.

       Handles the special cases for mixed lineages:
       - If n_der > 1: standard derived + ancestral coalescences
       - If n_der == 1 and n_anc == 1: treat as 2 ancestral lineages (freq != 0)
       - If n_der == 1 and n_anc > 1: ancestral coalescences, but with n_anc+1
         lineages when freq = 0 (all lineages become ancestral)
       - If n_der == 0: all remaining lineages are ancestral

       Parameters
       ----------
       coal_times_der : ndarray
           Derived coalescence times in this epoch.
       coal_times_anc : ndarray
           Ancestral coalescence times in this epoch.
       n_der : int
           Number of remaining derived lineages.
       n_anc : int
           Number of remaining ancestral lineages.
       epoch_start : float
           Start of epoch.
       epoch_end : float
           End of epoch.
       freqs : ndarray of shape (K,)
           Frequency bins.
       N_diploid : float
           Diploid effective population size.

       Returns
       -------
       emissions : ndarray of shape (K,)
           Log-emission probabilities for each frequency bin.
       """
       K = len(freqs)
       emissions = np.zeros(K)

       for j in range(K):
           x = freqs[j]

           if n_der > 1:
               # Standard case: both derived and ancestral coalescences
               emissions[j] = log_coalescent_density(
                   coal_times_der, n_der, epoch_start, epoch_end,
                   x, N_diploid, ancestral=False)
               emissions[j] += log_coalescent_density(
                   coal_times_anc, n_anc, epoch_start, epoch_end,
                   x, N_diploid, ancestral=True)

           elif n_der == 0 and n_anc <= 1:
               # No lineages or single lineage: no coalescence possible
               if j != 0:
                   emissions[j] = -1e20  # freq must be 0 (allele lost)

           elif n_der == 0 and n_anc > 1:
               # All remaining lineages are ancestral
               if j != 0:
                   emissions[j] = -1e20  # freq must be 0
               else:
                   emissions[j] = log_coalescent_density(
                       coal_times_anc, n_anc, epoch_start, epoch_end,
                       x, N_diploid, ancestral=True)

           elif n_der == 1 and n_anc == 1:
               # Mixed lineage: the single derived + single ancestral
               # coalesce as 2 ancestral lineages
               if j != 0:
                   emissions[j] = 0.0  # no constraint from freq
               else:
                   emissions[j] = log_coalescent_density(
                       coal_times_anc, 2, epoch_start, epoch_end,
                       x, N_diploid, ancestral=True)

           elif n_der == 1 and n_anc > 1:
               # One derived lineage remains, multiple ancestral
               if j != 0:
                   emissions[j] = log_coalescent_density(
                       coal_times_anc, n_anc, epoch_start, epoch_end,
                       x, N_diploid, ancestral=True)
               else:
                   # At freq = 0, the derived lineage joins the ancestral pool
                   emissions[j] = log_coalescent_density(
                       coal_times_anc, n_anc + 1, epoch_start, epoch_end,
                       x, N_diploid, ancestral=True)

       return emissions


Step 2: Ancient Genotype Likelihood Emissions
===============================================

When an ancient individual is sampled at time :math:`t`, we observe their genotype
(or, more precisely, genotype likelihoods from sequencing reads). This provides a
direct constraint on the allele frequency at that time.

Diploid genotype likelihoods
-----------------------------

For a diploid individual sampled at time :math:`t`, the genotype likelihoods are
:math:`P(R \mid g)` for each genotype :math:`g \in \{AA, AD, DD\}`, where :math:`R`
is the sequencing read data. We don't know the true genotype, but we know the
probability of each genotype given the allele frequency:

.. math::

   P(g = AA \mid x) = (1-x)^2, \quad P(g = AD \mid x) = 2x(1-x), \quad P(g = DD \mid x) = x^2

The **emission probability** for this ancient sample at frequency :math:`x` is:

.. math::

   e(x) = (1-x)^2 \cdot P(R \mid AA) + 2x(1-x) \cdot P(R \mid AD) + x^2 \cdot P(R \mid DD)

In log space:

.. math::

   \log e(x) = \text{logsumexp}\Bigl(
   2\log(1-x) + \log P(R \mid AA), \;
   \log 2 + \log x + \log(1-x) + \log P(R \mid AD), \;
   2\log x + \log P(R \mid DD)
   \Bigr)

.. admonition:: Why genotype *likelihoods* instead of genotype *calls*?

   Ancient DNA is often degraded and low-coverage. A site might have only 1-2
   sequencing reads, making the true genotype uncertain. Using genotype likelihoods
   (the probability of the reads given each possible genotype) rather than hard
   genotype calls preserves this uncertainty and avoids bias from calling errors.

   For example, if a site has 3 reads showing the derived allele and 1 showing
   ancestral, the genotype likelihoods might be :math:`P(R \mid AA) = 0.01`,
   :math:`P(R \mid AD) = 0.24`, :math:`P(R \mid DD) = 0.75`. A hard call would
   say "DD", but the likelihoods correctly represent the 24% chance of being
   heterozygous.

.. code-block:: python

   def genotype_likelihood_emission(anc_gl, log_freq, log_1minus_freq):
       """Compute the log-emission probability for a diploid ancient sample.

       Parameters
       ----------
       anc_gl : ndarray of shape (3,)
           Log genotype likelihoods: [log P(R|AA), log P(R|AD), log P(R|DD)].
       log_freq : float
           log(x), where x is the derived allele frequency.
       log_1minus_freq : float
           log(1 - x).

       Returns
       -------
       log_emission : float
           log P(R | freq = x), marginalizing over genotypes.
       """
       # Hardy-Weinberg genotype frequencies (in log space)
       log_geno_freqs = np.array([
           log_1minus_freq + log_1minus_freq,          # log((1-x)^2) = AA
           np.log(2) + log_freq + log_1minus_freq,     # log(2x(1-x)) = AD
           log_freq + log_freq                          # log(x^2) = DD
       ])

       # Combine: P(R|x) = sum_g P(g|x) * P(R|g)
       log_emission = logsumexp(log_geno_freqs + anc_gl)

       if np.isnan(log_emission):
           return -np.inf
       return log_emission

   # Example: ancient individual with 3 derived reads, 1 ancestral read
   # Genotype likelihoods (in log space)
   gl = np.log(np.array([0.01, 0.24, 0.75]))  # P(R|AA), P(R|AD), P(R|DD)

   print("Ancient genotype emission vs. frequency:")
   for freq in [0.1, 0.3, 0.5, 0.7, 0.9]:
       log_em = genotype_likelihood_emission(
           gl, np.log(freq), np.log(1 - freq))
       print(f"  freq = {freq:.1f}: log P(R|x) = {log_em:.4f}, "
             f"P(R|x) = {np.exp(log_em):.6f}")


Haploid genotype likelihoods
------------------------------

For haploid samples (or phased data), each individual carries exactly one allele:

.. math::

   e(x) = x \cdot P(R \mid D) + (1-x) \cdot P(R \mid A)

.. code-block:: python

   def haplotype_likelihood_emission(anc_gl, log_freq, log_1minus_freq):
       """Compute the log-emission probability for a haploid ancient sample.

       Parameters
       ----------
       anc_gl : ndarray of shape (2,)
           Log haplotype likelihoods: [log P(R|ancestral), log P(R|derived)].
       log_freq : float
           log(x), derived allele frequency.
       log_1minus_freq : float
           log(1 - x).

       Returns
       -------
       log_emission : float
       """
       # Haplotype frequencies: P(ancestral) = 1-x, P(derived) = x
       log_hap_freqs = np.array([log_1minus_freq, log_freq])

       log_emission = logsumexp(log_hap_freqs + anc_gl)
       if np.isnan(log_emission):
           return -np.inf
       return log_emission


Known haplotype emissions
--------------------------

When a haplotype's allelic state is known with certainty (as when lineages from the
gene tree are sampled at specific times in the past), the emission is simply
:math:`\log(x)` for a derived haplotype or :math:`\log(1-x)` for an ancestral
haplotype:

.. math::

   e_D(x) = x, \quad e_A(x) = 1-x

This represents the probability that a randomly sampled individual at frequency
:math:`x` carries the specified allele.


Step 3: Combining Emissions Within an Epoch
=============================================

At each generation :math:`t`, the total emission probability combines all evidence:

1. Coalescent emissions from derived and ancestral lineages.
2. Ancient diploid genotype likelihoods (if any samples fall in this generation).
3. Ancient haploid genotype likelihoods (if any).
4. Known haplotype emissions from ARG samples.

These are all independent (given the frequency), so their log-probabilities **add**:

.. math::

   \log e_{\text{total}}(t, x_k) = \log e_{\text{coal}}(t, x_k)
   + \sum_{i \in \text{diploid}} \log e_{\text{GL}}^{(i)}(x_k)
   + \sum_{j \in \text{haploid}} \log e_{\text{hap}}^{(j)}(x_k)
   + n_D^{(t)} \log(x_k) + n_A^{(t)} \log(1 - x_k)

where :math:`n_D^{(t)}` and :math:`n_A^{(t)}` are the numbers of derived and
ancestral haplotypes sampled from the ARG at generation :math:`t`.

.. code-block:: python

   def compute_total_emissions(freq_bins, logfreqs, log1minusfreqs,
                                coal_times_der, coal_times_anc,
                                n_der, n_anc, epoch_start, epoch_end,
                                N_diploid,
                                diploid_gls=None, haploid_gls=None,
                                n_der_sampled=0, n_anc_sampled=0):
       """Compute total emission probabilities for all frequency bins.

       Combines coalescent emissions, ancient genotype likelihoods,
       and known haplotype emissions.

       Parameters
       ----------
       freq_bins : ndarray of shape (K,)
           Frequency bins.
       logfreqs : ndarray of shape (K,)
           log(freq_bins).
       log1minusfreqs : ndarray of shape (K,)
           log(1 - freq_bins).
       coal_times_der : ndarray
           Derived coalescence times in this epoch.
       coal_times_anc : ndarray
           Ancestral coalescence times in this epoch.
       n_der : int
           Remaining derived lineages.
       n_anc : int
           Remaining ancestral lineages.
       epoch_start : float
           Start of epoch (generations).
       epoch_end : float
           End of epoch (generations).
       N_diploid : float
           Diploid effective population size.
       diploid_gls : list of ndarray, optional
           Each element is [log P(R|AA), log P(R|AD), log P(R|DD)] for one
           ancient diploid sample in this epoch.
       haploid_gls : list of ndarray, optional
           Each element is [log P(R|A), log P(R|D)] for one ancient haploid
           sample in this epoch.
       n_der_sampled : int
           Number of known derived haplotypes sampled in this epoch.
       n_anc_sampled : int
           Number of known ancestral haplotypes sampled in this epoch.

       Returns
       -------
       total_emissions : ndarray of shape (K,)
           Log-emission probability for each frequency bin.
       """
       K = len(freq_bins)

       # 1. Coalescent emissions
       coal_emissions = compute_coalescent_emissions(
           coal_times_der, coal_times_anc, n_der, n_anc,
           epoch_start, epoch_end, freq_bins, N_diploid)

       # 2. Ancient genotype likelihoods
       gl_emissions = np.zeros(K)
       if diploid_gls is not None:
           for gl in diploid_gls:
               for j in range(K):
                   gl_emissions[j] += genotype_likelihood_emission(
                       gl, logfreqs[j], log1minusfreqs[j])

       if haploid_gls is not None:
           for gl in haploid_gls:
               for j in range(K):
                   gl_emissions[j] += haplotype_likelihood_emission(
                       gl, logfreqs[j], log1minusfreqs[j])

       # 3. Known haplotype emissions from ARG samples
       hap_emissions = np.zeros(K)
       for j in range(K):
           if n_der_sampled > 0:
               hap_emissions[j] += n_der_sampled * logfreqs[j]
           if n_anc_sampled > 0:
               hap_emissions[j] += n_anc_sampled * log1minusfreqs[j]

       return coal_emissions + gl_emissions + hap_emissions

   # Example: no ancient samples, just coalescent emissions
   freqs, logfreqs, log1minusfreqs = build_frequency_bins(K=50)
   coal_der = np.array([0.3])  # one derived coalescence at t=0.3
   coal_anc = np.array([])     # no ancestral coalescences
   emissions = compute_total_emissions(
       freqs, logfreqs, log1minusfreqs,
       coal_der, coal_anc, n_der=3, n_anc=2,
       epoch_start=0.0, epoch_end=1.0, N_diploid=10000.0)

   # The emissions should peak where the frequency makes the coalescence
   # times most likely
   peak_bin = np.argmax(emissions)
   print(f"Peak emission at freq = {freqs[peak_bin]:.4f}")
   print(f"Log-emission range: [{emissions.min():.1f}, {emissions.max():.4f}]")


Step 4: How Emissions Constrain the Frequency
===============================================

Let's build intuition for how different types of evidence constrain the allele
frequency.

**Coalescent emissions alone.** If all derived lineages coalesce rapidly (short
coalescence times), the emission probability peaks at *low* frequency -- because
at low frequency, the effective population of derived lineages is small, forcing
rapid coalescence. Conversely, if coalescence is slow, the frequency was likely
high.

**Ancient genotype likelihoods.** An ancient individual with genotype :math:`DD`
(two derived copies) provides evidence for high frequency; genotype :math:`AA`
provides evidence for low frequency. Heterozygotes :math:`AD` favor intermediate
frequencies.

**The two types of evidence complement each other.** Coalescent emissions come from
the *shape* of the gene tree (deep in the past, many generations back). Ancient
genotypes provide direct snapshots of frequency at specific points in time. Together,
they constrain the trajectory far more tightly than either alone.

.. code-block:: python

   # Demonstrate how different data constrain the frequency differently

   freqs, logfreqs, log1minusfreqs = build_frequency_bins(K=100)

   # Scenario 1: Rapid derived coalescence (3 lineages coalescing at t=0.1 and t=0.2)
   coal_der_fast = np.array([0.1, 0.2])
   e_fast = compute_coalescent_emissions(
       coal_der_fast, np.array([]), n_der=3, n_anc=2,
       epoch_start=0.0, epoch_end=1.0, freqs=freqs, N_diploid=10000.0)

   # Scenario 2: Slow derived coalescence (coalescing at t=0.8 and t=0.9)
   coal_der_slow = np.array([0.8, 0.9])
   e_slow = compute_coalescent_emissions(
       coal_der_slow, np.array([]), n_der=3, n_anc=2,
       epoch_start=0.0, epoch_end=1.0, freqs=freqs, N_diploid=10000.0)

   # Normalize for comparison
   e_fast_norm = np.exp(e_fast - logsumexp(e_fast))
   e_slow_norm = np.exp(e_slow - logsumexp(e_slow))

   peak_fast = freqs[np.argmax(e_fast_norm)]
   peak_slow = freqs[np.argmax(e_slow_norm)]
   print(f"Fast coalescence: peak at freq = {peak_fast:.4f} (low freq expected)")
   print(f"Slow coalescence: peak at freq = {peak_slow:.4f} (high freq expected)")

   # Scenario 3: Ancient DD genotype (strong evidence for high frequency)
   gl_DD = np.log(np.array([0.001, 0.01, 0.989]))  # clearly DD
   e_ancient = np.array([
       genotype_likelihood_emission(gl_DD, logfreqs[j], log1minusfreqs[j])
       for j in range(len(freqs))])
   e_ancient_norm = np.exp(e_ancient - logsumexp(e_ancient))
   peak_ancient = freqs[np.argmax(e_ancient_norm)]
   print(f"Ancient DD sample: peak at freq = {peak_ancient:.4f}")


Exercises
=========

.. admonition:: Exercise 1: Coalescence rate vs. frequency

   For :math:`n = 5` derived lineages and :math:`N = 10000` (diploid):

   (a) Compute the pairwise coalescence rate as a function of frequency
       :math:`x \in \{0.01, 0.05, 0.1, 0.3, 0.5, 0.9\}`.
   (b) At what frequency does the expected time to the first coalescence equal 1
       generation? (Solve :math:`\binom{5}{2}/(x \cdot 2N) = 1`.)
   (c) If the actual first coalescence time is 0.001 generations, what frequency
       does this most strongly support? Why?

.. admonition:: Exercise 2: Ancient sample information content

   Compare the information content (the "peakedness" of the emission distribution)
   for three ancient sample scenarios, all at the same frequency bins:

   (a) A high-confidence DD call: :math:`P(R|AA) = 0.001, P(R|AD) = 0.01, P(R|DD) = 0.989`.
   (b) An ambiguous call: :math:`P(R|AA) = 0.3, P(R|AD) = 0.4, P(R|DD) = 0.3`.
   (c) No ancient sample (uniform emission).

   For each, compute the emission across all frequency bins and measure the
   entropy. Lower entropy = more informative.

.. admonition:: Exercise 3: The mixed lineage transition

   Consider a gene tree with 3 derived and 2 ancestral lineages. Walk through the
   coalescent process epoch by epoch as lineages coalesce:

   (a) In epoch 1: 2 derived lineages coalesce. After: 2 derived, 2 ancestral.
   (b) In epoch 2: 1 derived coalesces with 1 ancestral. After: 1 derived, 1 ancestral.
       But this is the mixed lineage case! What happens?
   (c) Implement the emission computation for each epoch and verify that the log-
       probability is finite (not :math:`-\infty`) for intermediate frequencies.


Solutions
=========

.. admonition:: Solution 1: Coalescence rate vs. frequency

   .. code-block:: python

      n = 5
      N_dip = 10000
      N_hap = 2 * N_dip  # haploid = 20,000

      print("(a) Coalescence rate vs. frequency:")
      for x in [0.01, 0.05, 0.1, 0.3, 0.5, 0.9]:
          rate = n * (n - 1) / 2 / (x * N_hap)
          expected_time = 1.0 / rate
          print(f"  x = {x:.2f}: rate = {rate:.4f}/gen, "
                f"E[T_first] = {expected_time:.2f} gen")

      # (b) Solve: C(5,2) / (x * 2N) = 1
      # => x = C(5,2) / (2N) = 10 / 20000 = 0.0005
      x_threshold = 10.0 / N_hap
      print(f"\n(b) E[T_first] = 1 gen when x = {x_threshold:.6f}")

      # (c) Very fast coalescence (0.001 gen) supports very low frequency
      # because the coalescence rate must be ~1000/gen = C(5,2)/(x*2N)
      x_fast = 10.0 / (N_hap * 1000)
      print(f"\n(c) Coal at t=0.001 supports x ~ {x_fast:.6f}")

   Very fast coalescence strongly supports low allele frequency, because the
   derived lineages must be crammed into a tiny sub-population.

.. admonition:: Solution 2: Ancient sample information content

   .. code-block:: python

      from scipy.stats import entropy as scipy_entropy

      freqs, logfreqs, log1minusfreqs = build_frequency_bins(K=100)

      scenarios = {
          "(a) High-confidence DD": np.log(np.array([0.001, 0.01, 0.989])),
          "(b) Ambiguous call": np.log(np.array([0.3, 0.4, 0.3])),
          "(c) No ancient sample": None,
      }

      for name, gl in scenarios.items():
          if gl is not None:
              emissions = np.array([
                  genotype_likelihood_emission(gl, logfreqs[j], log1minusfreqs[j])
                  for j in range(len(freqs))])
          else:
              emissions = np.zeros(len(freqs))  # uniform (no information)

          probs = np.exp(emissions - logsumexp(emissions))
          ent = scipy_entropy(probs)
          peak = freqs[np.argmax(probs)]
          print(f"{name}: entropy = {ent:.4f}, peak at x = {peak:.4f}")

   The high-confidence DD sample has the lowest entropy (most informative),
   peaking near :math:`x = 1`. The ambiguous call has higher entropy but still
   provides some constraint. No ancient sample gives uniform emissions (maximum
   entropy).

.. admonition:: Solution 3: The mixed lineage transition

   .. code-block:: python

      freqs, logfreqs, log1minusfreqs = build_frequency_bins(K=50)
      N_dip = 10000.0

      # Epoch 1: 2 derived lineages coalesce at t=0.5
      # Before: 3 derived, 2 ancestral. After: 2 derived, 2 ancestral.
      e1 = compute_coalescent_emissions(
          np.array([0.5]), np.array([]),
          n_der=3, n_anc=2, epoch_start=0.0, epoch_end=1.0,
          freqs=freqs, N_diploid=N_dip)
      print(f"Epoch 1 (3D, 2A, 1 der coal): max log-emission = {e1.max():.4f}")

      # Epoch 2: In this epoch, n_der=2, n_anc=2.
      # One derived and one ancestral coalesce.
      # After: n_der=1, n_anc=1. This is the mixed lineage case.
      e2 = compute_coalescent_emissions(
          np.array([0.3]), np.array([0.7]),
          n_der=2, n_anc=2, epoch_start=1.0, epoch_end=2.0,
          freqs=freqs, N_diploid=N_dip)
      print(f"Epoch 2 (2D, 2A, coals): max log-emission = {e2.max():.4f}")

      # Epoch 3: n_der=1, n_anc=1 (mixed lineage case)
      # No coalescences in this epoch
      e3 = compute_coalescent_emissions(
          np.array([]), np.array([]),
          n_der=1, n_anc=1, epoch_start=2.0, epoch_end=3.0,
          freqs=freqs, N_diploid=N_dip)
      n_finite = np.sum(e3 > -1e19)
      print(f"Epoch 3 (1D, 1A, mixed): {n_finite}/{len(freqs)} "
            f"bins have finite emission")

   In epoch 3, the mixed lineage case allows finite emissions for all frequency
   bins except :math:`x = 0` (where the code correctly handles the special case
   of 2 ancestral lineages). The derived allele has not been lost, so :math:`x > 0`
   is required.
