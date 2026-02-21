.. _linkage_disequilibrium:

=============================
Linkage Disequilibrium
=============================

   *A second pendulum: reading demographic history from correlations between loci.*

The SFS captures how often alleles appear at each frequency, but it ignores the
**relationships between loci**. Two sites that are physically close on a chromosome
tend to be inherited together -- they are in **linkage disequilibrium** (LD). The
rate at which LD decays with physical distance depends on population history,
providing information that the SFS alone cannot.

``moments.LD`` extends the moment equations framework to two-locus statistics,
opening a second window onto demographic history -- one that is especially powerful
for detecting recent events like admixture.

In the watch metaphor developed in :ref:`moments_overview`, the SFS is the main
**dial face** -- the primary read-out of population history.  LD is a **second
pendulum** mounted alongside the first.  Both are driven by the same gear train
(drift, mutation, migration), but the second pendulum also responds to a force the
first cannot feel: **recombination**.  Because recombination rate scales with
:math:`N_e` (via :math:`\rho = 4N_e r`), the LD pendulum constrains the absolute
population size -- something the SFS, which sees only :math:`\theta = 4N_e \mu`,
cannot do on its own.  Together, the two dials break the degeneracy and pin down
both :math:`N_e` and :math:`\mu` separately.


Step 1: What Is Linkage Disequilibrium?
========================================

Consider two biallelic loci, A and B. Locus A has alleles :math:`A_0, A_1` at
frequencies :math:`1 - p, p`. Locus B has alleles :math:`B_0, B_1` at
frequencies :math:`1 - q, q`.

There are four possible **haplotypes**:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Haplotype
     - Notation
     - Frequency
   * - :math:`A_0 B_0`
     - :math:`x_{00}`
     - Expected: :math:`(1-p)(1-q)` if independent
   * - :math:`A_0 B_1`
     - :math:`x_{01}`
     - Expected: :math:`(1-p)q` if independent
   * - :math:`A_1 B_0`
     - :math:`x_{10}`
     - Expected: :math:`p(1-q)` if independent
   * - :math:`A_1 B_1`
     - :math:`x_{11}`
     - Expected: :math:`pq` if independent

If the two loci are **independent** (in linkage equilibrium), haplotype
frequencies are just the product of allele frequencies. **Linkage disequilibrium**
is any departure from this independence:

.. math::

   D = x_{11} - pq = x_{11}x_{00} - x_{01}x_{10}

:math:`D > 0` means alleles :math:`A_1` and :math:`B_1` co-occur more often than
expected by chance. :math:`D < 0` means they co-occur less often.

.. admonition:: Probability Aside -- LD as covariance

   If you assign the value 1 to allele :math:`A_1` and 0 to :math:`A_0`
   (and similarly for locus B), then for a randomly chosen chromosome:

   .. math::

      D = E[A \cdot B] - E[A] \cdot E[B] = \text{Cov}(A, B)

   LD is literally the **covariance** between the allelic states at two
   loci.  Just as the covariance between two random variables measures their
   linear association, :math:`D` measures how much knowing the allele at one
   locus tells you about the allele at the other.  When :math:`D = 0`, the
   loci are statistically independent (at the population level); when
   :math:`D \neq 0`, they are associated.

.. code-block:: python

   import numpy as np

   def compute_D(haplotypes):
       """Compute D for two biallelic loci from haplotype data.

       Parameters
       ----------
       haplotypes : ndarray of shape (n, 2)
           Each row is a haplotype. Columns are the two loci (0/1).

       Returns
       -------
       D : float
       """
       n = len(haplotypes)
       p = haplotypes[:, 0].mean()  # frequency of allele 1 at locus A
       q = haplotypes[:, 1].mean()  # frequency of allele 1 at locus B
       x11 = ((haplotypes[:, 0] == 1) & (haplotypes[:, 1] == 1)).mean()  # freq of (1,1) haplotype
       D = x11 - p * q  # departure from independence
       return D

   # Example: 100 haplotypes, two loci in strong LD
   np.random.seed(42)
   n = 100
   # All haplotypes are either (0,0) or (1,1) -- perfect LD
   hap_ld = np.array([[0, 0]] * 60 + [[1, 1]] * 40)
   np.random.shuffle(hap_ld)

   D_strong = compute_D(hap_ld)
   p = hap_ld[:, 0].mean()
   q = hap_ld[:, 1].mean()
   D_max = min(p * (1 - q), q * (1 - p))  # maximum possible D given allele freqs

   print(f"p = {p:.2f}, q = {q:.2f}")
   print(f"D = {D_strong:.4f}")
   print(f"D_max = {D_max:.4f}")
   print(f"D' = D/D_max = {D_strong/D_max:.4f}")

**Intuition**: LD is the covariance between allelic states at two loci. Just as
correlation measures association between two random variables, LD measures
association between alleles. Recombination breaks up these associations over
time, so LD decays with distance and with time.

Now that we know what LD *is*, let us see how recombination erodes it.


Step 2: LD Decay and Recombination
====================================

In each generation, recombination between two loci occurs with probability
:math:`r` (the recombination rate). Each recombination event breaks up the
existing LD:

.. math::

   E[D_{t+1}] = (1 - r) \cdot D_t

After :math:`t` generations without new LD being created:

.. math::

   E[D_t] = (1 - r)^t \cdot D_0 \approx e^{-rt} \cdot D_0

LD decays **exponentially** with time at a rate proportional to :math:`r`. This is
the deterministic approximation. In a finite population, genetic drift constantly
creates new LD (randomly), which balances the decay and produces a nonzero
equilibrium level.

.. admonition:: Calculus Aside -- Exponential decay and half-life

   The recurrence :math:`D_{t+1} = (1-r) D_t` is a first-order linear
   difference equation with solution :math:`D_t = (1-r)^t D_0`.  For small
   :math:`r`, the approximation :math:`(1-r)^t \approx e^{-rt}` converts
   this into the continuous exponential decay
   :math:`D(t) = D_0 \, e^{-rt}`.  The **half-life** -- the time for LD
   to drop to half its initial value -- is:

   .. math::

      t_{1/2} = \frac{\ln 2}{r}

   For :math:`r = 0.01` (about 1 centimorgan), the half-life is roughly 69
   generations.  For :math:`r = 10^{-4}` (about 0.01 cM, or roughly 10 kb
   in humans), the half-life is about 6,900 generations -- long enough that
   LD from ancient events can still be detected.

.. code-block:: python

   def ld_decay_deterministic(D0, r, t_generations):
       """Deterministic LD decay over t generations.

       Parameters
       ----------
       D0 : float
           Initial D.
       r : float
           Recombination rate between loci.
       t_generations : int
           Number of generations.

       Returns
       -------
       D_t : float
       """
       return D0 * (1 - r) ** t_generations  # exponential decay

   # Example: LD decays to half in log(2)/r generations
   r = 0.01  # 1% recombination rate (= ~1 cM distance)
   half_life = np.log(2) / r
   print(f"Half-life of LD at r={r}: {half_life:.0f} generations")

   # Plot LD decay
   D0 = 0.1
   t_range = range(0, 500)
   D_vals = [ld_decay_deterministic(D0, r, t) for t in t_range]
   print(f"D at t=0: {D_vals[0]:.4f}")
   print(f"D at t=69: {D_vals[69]:.4f} (approximately half-life)")
   print(f"D at t=500: {D_vals[-1]:.6f}")

With the basic dynamics in place, we can now introduce the specific LD
statistics that ``moments.LD`` tracks.


Step 3: The Three LD Statistics in moments
============================================

Raw :math:`D` is hard to work with because it depends on allele frequencies.
``moments.LD`` tracks three **expected values** of normalized LD statistics,
computed as moments of the two-locus haplotype distribution:

Statistic 1: :math:`E[D^2]`
-----------------------------

The expected squared LD coefficient. This is the numerator of the commonly used
:math:`r^2`:

.. math::

   r^2 = \frac{D^2}{p(1-p)q(1-q)}

:math:`E[D^2]` measures the overall *magnitude* of LD, regardless of sign.

Statistic 2: :math:`E[Dz]` where :math:`z = (1 - 2p)(1 - 2q)`
-----------------------------------------------------------------

.. math::

   E[Dz] = E[D \cdot (1 - 2p)(1 - 2q)]

This is :math:`D` weighted by how far allele frequencies are from 0.5. It has a
remarkable property: **it is strongly elevated by admixture** between
diverged populations, even at low admixture proportions.

**Intuition**: When two populations with different allele frequencies admix, the
resulting LD has a specific pattern: :math:`D` and :math:`(1-2p)(1-2q)` tend to
have the same sign (because alleles that are common in one population and rare
in the other create positive :math:`D` between loci where the same population is
at high frequency, and these loci also tend to have :math:`(1-2p)` of the same
sign). This makes :math:`E[Dz]` an exquisitely sensitive detector of admixture.

.. admonition:: Probability Aside -- Why :math:`E[Dz]` detects admixture

   Consider two source populations that have diverged long enough that their
   allele frequencies differ appreciably.  At a pair of loci where population
   1 has frequencies :math:`(p_1, q_1)` and population 2 has
   :math:`(p_2, q_2)`, the admixed population (mixing fraction
   :math:`\alpha`) inherits LD of the form:

   .. math::

      D_{\text{admix}} \approx \alpha(1-\alpha)(p_1 - p_2)(q_1 - q_2)

   The factor :math:`(1-2p)(1-2q)` has the same sign structure as
   :math:`(p_1 - p_2)(q_1 - q_2)` when :math:`p` and :math:`q` lie between
   the two source frequencies.  As a result, :math:`D \cdot (1-2p)(1-2q)` is
   systematically positive across locus pairs, producing a large positive
   :math:`E[Dz]`.  Under a purely drifting (non-admixed) population, the
   sign of :math:`D` is random, so :math:`E[Dz] \approx 0`.  This is why
   :math:`E[Dz]` serves as a near-zero-background signal for admixture.

Statistic 3: :math:`\pi_2 = E[p(1-p)q(1-q)]`
------------------------------------------------

The product of heterozygosities at two loci -- a normalization factor:

.. math::

   \pi_2 = E[p(1-p) \cdot q(1-q)]

This does not carry LD information per se; it serves as the denominator for
normalized statistics.

Normalized statistics
----------------------

The statistics used for inference are the ratios:

.. math::

   \sigma_d^2 = \frac{E[D^2]}{E[\pi_2]}, \qquad
   \sigma_{Dz} = \frac{E[Dz]}{E[\pi_2]}

These are analogous to :math:`r^2` but defined as ratios of expectations (not
expectations of ratios), which is mathematically cleaner and avoids singularities
when individual-site heterozygosities are small.

.. admonition:: Calculus Aside -- Ratio of expectations vs. expectation of ratio

   For two random variables :math:`X` and :math:`Y`, the ratio of
   expectations :math:`E[X]/E[Y]` is generally *not* equal to the
   expectation of the ratio :math:`E[X/Y]` (Jensen's inequality).  The
   former is well-defined even when :math:`Y` can be zero for some
   realizations; the latter can diverge.  By defining :math:`\sigma_d^2` as
   :math:`E[D^2] / E[\pi_2]` rather than :math:`E[D^2 / (p(1-p)q(1-q))]`,
   ``moments.LD`` avoids the singularity that arises when one locus is
   nearly monomorphic (heterozygosity near zero).  This is a common
   technique in survey sampling and meta-analysis.

.. code-block:: python

   def compute_ld_statistics(haplotype_matrix):
       """Compute E[D^2], E[Dz], and pi_2 from a haplotype matrix.

       Parameters
       ----------
       haplotype_matrix : ndarray of shape (n_haplotypes, n_loci)
           Binary matrix (0/1) for each locus.

       Returns
       -------
       D2_mean, Dz_mean, pi2_mean : float
           Average D^2, Dz, and pi_2 over all pairs of loci.
       """
       n_haps, n_loci = haplotype_matrix.shape
       D2_sum, Dz_sum, pi2_sum = 0.0, 0.0, 0.0
       n_pairs = 0

       for i in range(n_loci):
           for j in range(i + 1, n_loci):
               p = haplotype_matrix[:, i].mean()  # allele freq at locus i
               q = haplotype_matrix[:, j].mean()  # allele freq at locus j

               # Skip monomorphic loci (no LD can be computed)
               if p == 0 or p == 1 or q == 0 or q == 1:
                   continue

               # Compute D = freq(1,1) - p*q
               x11 = ((haplotype_matrix[:, i] == 1) &
                       (haplotype_matrix[:, j] == 1)).mean()
               D = x11 - p * q

               D2_sum += D ** 2                         # squared LD
               Dz_sum += D * (1 - 2*p) * (1 - 2*q)     # admixture-sensitive statistic
               pi2_sum += p * (1 - p) * q * (1 - q)     # product of heterozygosities
               n_pairs += 1

       if n_pairs == 0:
           return 0, 0, 0

       return D2_sum / n_pairs, Dz_sum / n_pairs, pi2_sum / n_pairs

Having defined the LD statistics, we now derive the ODEs that govern their
evolution -- the two-locus analogue of the moment equations in
:ref:`moment_equations`.


Step 4: Two-Locus Moment Equations
=====================================

Just as the SFS entries satisfy ODEs under drift/mutation/selection, the
two-locus statistics :math:`E[D^2]`, :math:`E[Dz]`, and :math:`\pi_2` satisfy
their own system of ODEs.

The key forces acting on two-locus statistics:

**Drift** creates LD (randomly) and changes allele frequencies:

.. math::

   \left.\frac{dE[D^2]}{dt}\right|_{\text{drift}} =
   \text{(terms involving } E[D^2], E[Dz], \pi_2, \text{ and heterozygosities)}

**Recombination** destroys LD:

.. math::

   \left.\frac{dE[D^2]}{dt}\right|_{\text{recomb}} = -2\rho \cdot E[D^2]

where :math:`\rho = 4N_e r` is the scaled recombination rate. The factor of 2
comes from the fact that each recombination event reduces :math:`D` by a factor
:math:`(1-r)`, so :math:`D^2` is reduced by :math:`(1-r)^2 \approx 1 - 2r`.

.. admonition:: Calculus Aside -- Why :math:`D^2` decays at rate :math:`2\rho`

   Recall from Step 2 that :math:`D_{t+1} = (1-r) D_t`.  Squaring both sides:
   :math:`D_{t+1}^2 = (1-r)^2 D_t^2`.  Taking expectations and using the
   approximation :math:`(1-r)^2 \approx 1 - 2r` for small :math:`r`:

   .. math::

      E[D_{t+1}^2] \approx (1 - 2r) \, E[D_t^2]

   Converting to diffusion time (:math:`dt = 1/(2N_e)` generations) and
   using :math:`\rho = 4N_e r`:

   .. math::

      \frac{dE[D^2]}{dt}\bigg|_{\text{recomb}} = -2\rho \, E[D^2]

   This is exponential decay with rate constant :math:`2\rho`.  The factor
   of 2 arises because :math:`D^2` is a *second-order* statistic: each
   recombination event reduces :math:`D` by a factor :math:`(1-r)`, so
   :math:`D^2` is reduced by :math:`(1-r)^2`, which contributes *twice* the
   decay rate.

**Mutation** generates new variation that contributes to heterozygosity and
indirectly to LD:

.. math::

   \left.\frac{d\pi_2}{dt}\right|_{\text{mutation}} = \theta \cdot E[q(1-q)] + \theta \cdot E[p(1-p)]

The full system is a set of coupled ODEs -- more complex than the SFS system
because it involves higher-order moments of two allele frequencies, but the
same moment-closure techniques (the jackknife approximation from
:ref:`moment_equations`) apply.

.. code-block:: python

   def ld_equilibrium(theta, rho, n_pops=1):
       """Compute equilibrium LD statistics for one population.

       At equilibrium, the rate of LD creation by drift equals the rate
       of LD decay by recombination.

       Parameters
       ----------
       theta : float
           Scaled mutation rate (4*Ne*mu per site).
       rho : float
           Scaled recombination rate (4*Ne*r between the two loci).

       Returns
       -------
       sigma_d2 : float
           Equilibrium sigma_d^2 = E[D^2] / pi_2.
       """
       # Under the neutral model with constant population size,
       # sigma_d^2 at equilibrium is approximately:
       #
       #   sigma_d^2 ~ 1 / (1 + rho)  (for large n, low theta)
       #
       # This is because:
       # - Drift creates D^2 at a rate proportional to pi_2
       #   (roughly 1/(2N) per generation, or 1 in diffusion time)
       # - Recombination destroys D^2 at rate 2*rho
       # - At equilibrium: creation = destruction
       # - D^2 ~ pi_2 / (1 + rho) [simplified]
       return 1.0 / (1.0 + rho)

   # Compare with moments
   import moments.LD

   rho_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
   theta = 0.001

   print(f"{'rho':>6} {'sigma_d2 (approx)':>18} {'sigma_d2 (moments)':>20}")
   print("-" * 46)
   for rho in rho_values:
       # Compute with moments.LD: solve the two-locus moment equations at equilibrium
       y = moments.LD.LDstats(
           moments.LD.Numerics.steady_state([rho], theta=theta),
           num_pops=1, pop_ids=["pop0"]
       )
       # sigma_d2 = E[D^2] / pi_2
       sigma_d2_moments = y.D2() / y.pi2()

       sigma_d2_approx = ld_equilibrium(theta, rho)
       print(f"{rho:6.1f} {sigma_d2_approx:18.6f} {sigma_d2_moments:20.6f}")

.. admonition:: The recombination-LD tradeoff

   :math:`\sigma_d^2 \approx 1/(1 + \rho)` is a profound result. It says that
   the equilibrium level of LD between two loci is determined solely by the
   **ratio of drift to recombination**. Closer loci (lower :math:`\rho`) have
   more LD. This is why LD decay curves -- plots of :math:`\sigma_d^2` vs.
   recombination distance -- encode demographic information: the shape of the
   decay depends on population history.

With the equilibrium baseline established, we now examine how specific
demographic events distort the LD decay curve.


Step 5: How Demography Affects LD
====================================

Different demographic events leave characteristic signatures in the LD decay
curve:

Population expansion
---------------------

After an expansion, drift is weak (large :math:`N`). Existing LD decays by
recombination but new LD is created slowly. **Result**: lower :math:`\sigma_d^2`
than expected under constant size.

Bottleneck
-----------

During a bottleneck, drift is strong. Lots of LD is created. Even after the
population recovers, the excess LD at short distances persists for many
generations. **Result**: elevated :math:`\sigma_d^2`, especially at short
distances.

Admixture
----------

When two diverged populations admix, they bring together different haplotypes.
This creates LD between *any* pair of loci where the source populations had
different allele frequencies. The resulting "admixture LD" decays with
recombination distance because recombination breaks up the immigrant haplotypes.
**Result**: elevated :math:`\sigma_{Dz}` that decays with distance -- the
hallmark of admixture.

.. admonition:: Probability Aside -- Admixture LD and the ROLLOFF/ALDER signal

   The LD created by admixture at generation :math:`t_a` in the past decays
   as :math:`e^{-r t_a}` with recombination distance :math:`r`.  Plotting
   admixture LD against genetic distance therefore produces an exponential
   curve whose decay constant is the number of generations since admixture.
   This is the principle behind the ROLLOFF and ALDER methods.
   ``moments.LD`` captures the same information through its
   :math:`\sigma_{Dz}` statistic, but models it via the moment equations
   rather than fitting an empirical exponential.  The moment-equation
   approach naturally handles complex scenarios -- e.g., multiple pulses of
   admixture, or continuous gene flow -- that would require ad hoc
   extensions in an exponential-fitting framework.

.. code-block:: python

   import moments.LD

   def ld_after_bottleneck(nu_B, T_B, rho_values, theta=0.001):
       """Compute LD statistics after a bottleneck.

       Parameters
       ----------
       nu_B : float
           Bottleneck population size (relative to N_ref).
       T_B : float
           Duration of bottleneck (2*N_ref generations).
       rho_values : list of float
           Scaled recombination rates to compute.
       theta : float
           Scaled mutation rate.

       Returns
       -------
       sigma_d2 : list of float
           sigma_d^2 at each rho value.
       """
       sigma_d2 = []
       for rho in rho_values:
           # Start from equilibrium (steady-state two-locus statistics)
           y = moments.LD.LDstats(
               moments.LD.Numerics.steady_state([rho], theta=theta),
               num_pops=1, pop_ids=["pop0"]
           )
           # Bottleneck: strong drift creates excess LD
           y.integrate([nu_B], T_B, rho=[rho], theta=theta)
           # Recovery to original size
           y.integrate([1.0], 0.1, rho=[rho], theta=theta)

           sigma_d2.append(y.D2() / y.pi2())  # normalized LD statistic
       return sigma_d2

   # Compare equilibrium vs post-bottleneck LD
   rho_values = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

   ld_eq = [ld_equilibrium(0.001, r) for r in rho_values]
   ld_bn = ld_after_bottleneck(0.1, 0.05, rho_values)

   print(f"{'rho':>6} {'Equilibrium':>14} {'Post-bottleneck':>16} {'Ratio':>8}")
   print("-" * 48)
   for rho, eq, bn in zip(rho_values, ld_eq, ld_bn):
       print(f"{rho:6.1f} {eq:14.6f} {bn:16.6f} {bn/eq:8.3f}")
   print("(Ratio > 1 = bottleneck elevates LD)")

The LD decay curve is our second dial reading.  To use it for inference, we
need a likelihood -- and because LD statistics are averages rather than counts,
the appropriate likelihood is Gaussian, not Poisson.


Step 6: LD-Based Demographic Inference
=========================================

The inference framework for LD parallels the SFS framework, with a key difference:
the likelihood is **Gaussian** rather than Poisson.

The Gaussian composite likelihood
-----------------------------------

LD statistics are computed by averaging over many pairs of SNPs within each
recombination distance bin. By the central limit theorem, these averages are
approximately normally distributed. The log-likelihood for a set of LD
statistics :math:`\mathbf{d}` in recombination bin :math:`k` is:

.. math::

   \ell_k = -\frac{1}{2}(\mathbf{d}_k - \boldsymbol{\mu}_k)^T
   \Sigma_k^{-1}
   (\mathbf{d}_k - \boldsymbol{\mu}_k)

where:

- :math:`\mathbf{d}_k` = observed LD statistics in bin :math:`k` (a vector
  containing :math:`\sigma_d^2` and optionally :math:`\sigma_{Dz}`)
- :math:`\boldsymbol{\mu}_k` = model predictions from the moment equations
- :math:`\Sigma_k` = variance-covariance matrix (estimated from bootstrap)

The total log-likelihood is the sum over bins:

.. math::

   \ell = \sum_k \ell_k

.. admonition:: Probability Aside -- From Poisson (SFS) to Gaussian (LD)

   The SFS consists of **counts** (how many sites have each allele frequency),
   which are naturally Poisson. LD statistics are **averages** over pairs of
   sites within recombination distance bins. As averages of many independent-ish
   terms, they follow a Gaussian distribution by the central limit theorem.

   Formally, if you average :math:`N` independent observations each with
   variance :math:`\sigma^2`, the sample mean has variance
   :math:`\sigma^2 / N` and is approximately Gaussian for large :math:`N`.
   Each recombination bin typically contains thousands to millions of SNP
   pairs, so the Gaussian approximation is excellent.  The covariance matrix
   :math:`\Sigma_k` accounts for the fact that the same SNP can appear in
   multiple pairs (and that nearby pairs share haplotype information),
   inflating the effective variance above the naive :math:`\sigma^2 / N`.

.. code-block:: python

   def gaussian_composite_ll(data_ld, model_ld, varcov_matrices):
       """Compute Gaussian composite log-likelihood for LD statistics.

       Parameters
       ----------
       data_ld : list of ndarray
           Observed LD statistics, one array per recombination bin.
       model_ld : list of ndarray
           Model-predicted LD statistics.
       varcov_matrices : list of ndarray
           Variance-covariance matrix for each bin (from bootstrap).

       Returns
       -------
       ll : float
       """
       ll = 0.0
       for d, mu, sigma in zip(data_ld, model_ld, varcov_matrices):
           residual = d - mu                    # data minus model prediction
           sigma_inv = np.linalg.inv(sigma)      # precision matrix
           ll -= 0.5 * residual @ sigma_inv @ residual  # quadratic form
       return ll

.. admonition:: Why Gaussian instead of Poisson?

   The SFS consists of **counts** (how many sites have each allele frequency),
   which are naturally Poisson. LD statistics are **averages** over pairs of
   sites within recombination distance bins. As averages of many independent-ish
   terms, they follow a Gaussian distribution by the central limit theorem.


Step 7: Multi-Population LD and Admixture Detection
======================================================

For multiple populations, LD statistics include **cross-population** terms:

- :math:`E[D_i \cdot D_j]`: covariance of LD between populations :math:`i` and
  :math:`j`

Right after a population split, :math:`E[D_i D_j] = E[D^2]` (perfect correlation).
Over time, independent drift in each population decorrelates their LD:
:math:`E[D_i D_j]` decays toward zero.

.. admonition:: Probability Aside -- Cross-population LD as shared ancestry

   Two populations that recently shared an ancestor have correlated LD: the
   same ancestral haplotypes, and therefore the same LD patterns, were
   present in both at the time of the split.  As independent drift in each
   population creates new, uncorrelated LD and recombination erodes the
   shared signal, :math:`E[D_i D_j]` decays.  The rate of decay depends on
   both :math:`\rho` (how fast recombination destroys old LD) and the
   population sizes (how fast new drift creates uncorrelated LD).  By
   measuring :math:`E[D_i D_j]` at multiple recombination distances, one
   can therefore estimate both the split time and the population sizes --
   providing information complementary to :math:`F_{ST}` from the SFS
   (see :ref:`the_frequency_spectrum`).

.. code-block:: python

   import moments.LD

   def two_pop_ld(nu1, nu2, T, m, rho_values, theta=0.001):
       """Compute LD statistics for two populations after a split.

       Parameters
       ----------
       nu1, nu2 : float
           Relative population sizes.
       T : float
           Time since split.
       m : float
           Symmetric migration rate.
       rho_values : list of float
       theta : float

       Returns
       -------
       results : dict
           LD statistics for each rho value.
       """
       results = {'rho': rho_values, 'DD_0_0': [], 'DD_0_1': [], 'DD_1_1': []}

       for rho in rho_values:
           # Equilibrium for one population
           y = moments.LD.LDstats(
               moments.LD.Numerics.steady_state([rho], theta=theta),
               num_pops=1, pop_ids=["anc"]
           )

           # Split into two populations
           y = y.split(0, new_ids=["pop0", "pop1"])

           # Diverge with migration
           mig = np.array([[0, m], [m, 0]])
           y.integrate([nu1, nu2], T, rho=[rho], theta=theta, m=mig)

           # Collect within-population and cross-population LD
           results['DD_0_0'].append(y.D2(pops="pop0"))    # LD within pop 0
           results['DD_0_1'].append(y.D2(pops=["pop0", "pop1"]))  # cross-pop LD
           results['DD_1_1'].append(y.D2(pops="pop1"))    # LD within pop 1

       return results

   # Example: recent split with moderate migration
   rho_values = [0.5, 1.0, 2.0, 5.0, 10.0]
   res = two_pop_ld(1.0, 1.0, 0.1, 0.5, rho_values)

   print("Cross-population LD correlation:")
   print(f"{'rho':>6} {'DD_00':>10} {'DD_01':>10} {'DD_11':>10} {'Corr':>8}")
   print("-" * 48)
   for i, rho in enumerate(rho_values):
       corr = res['DD_0_1'][i] / np.sqrt(res['DD_0_0'][i] * res['DD_1_1'][i])
       print(f"{rho:6.1f} {res['DD_0_0'][i]:10.6f} {res['DD_0_1'][i]:10.6f} "
             f"{res['DD_1_1'][i]:10.6f} {corr:8.4f}")
   print("(Higher correlation = more recent split or more migration)")


Step 8: Parsing LD from Real Data
====================================

Computing LD statistics from VCF files involves:

1. **Pair up SNPs** within recombination distance bins
2. **Compute** :math:`D^2`, :math:`Dz`, :math:`\pi_2` for each pair
3. **Average** within bins
4. **Bootstrap** across genomic regions for uncertainty

.. code-block:: python

   # Pseudocode for the parsing pipeline
   import moments.LD

   def parse_ld_workflow(vcf_file, pop_file, recomb_map, bed_file):
       """Complete workflow for parsing LD from VCF data.

       Parameters
       ----------
       vcf_file : str
           Path to VCF file.
       pop_file : str
           Path to population assignment file (sample -> population).
       recomb_map : str
           Path to recombination map file (position -> cumulative cM).
       bed_file : str
           Path to BED file defining genomic regions.
       """
       # Define recombination distance bins (in Morgans)
       r_bins = np.array([0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5,
                           5e-5, 1e-4, 2e-4, 5e-4, 1e-3])

       # Compute LD statistics per region (moments.LD handles the averaging)
       ld_stats = moments.LD.Parsing.compute_ld_statistics(
           vcf_file,
           rec_map_file=recomb_map,
           pop_file=pop_file,
           bed_file=bed_file,
           r_bins=r_bins,
           report=False
       )

       return ld_stats

   # In practice, you compute LD for many regions and bootstrap:
   # region_stats = {i: parse_ld_workflow(..., bed=f"region_{i}.bed")
   #                 for i in range(n_regions)}
   # means_and_varcov = moments.LD.Parsing.bootstrap_data(region_stats)

.. admonition:: Phased vs. unphased data

   ``moments.LD`` defaults to using **genotype-based** statistics (unphased)
   rather than haplotype-based (phased). This is deliberate: phasing errors
   create artificial LD that biases results, while using genotypes only
   slightly increases variance. Unless your data has very high-quality phasing,
   keep the default ``use_genotypes=True``.

With observed LD statistics and a demographic model, the last ingredient is
the conversion between physical recombination rates and scaled :math:`\rho`
values.


Step 9: Recombination Bins and N_e
====================================

A crucial link: the model operates in **scaled** units (:math:`\rho = 4N_e r`),
but the data is binned by **physical** recombination rates (:math:`r`). The
effective population size :math:`N_e` provides the conversion:

.. math::

   \rho = 4 N_e \cdot r

This means :math:`N_e` can be **jointly estimated** from the LD decay curve: it
determines how the physical-distance bins map onto the scaled :math:`\rho` values
where the model makes predictions.

.. admonition:: Calculus Aside -- Why LD resolves the :math:`N_e`--:math:`\mu` degeneracy

   The SFS depends on :math:`\theta = 4N_e \mu` -- it constrains the
   *product* of :math:`N_e` and :math:`\mu` but cannot separate them.  The
   LD decay curve depends on :math:`\rho = 4N_e r` -- it constrains the
   *product* of :math:`N_e` and :math:`r`.  If :math:`r` is known from a
   genetic map (and :math:`\mu` is independently estimated, or vice versa),
   combining SFS and LD data pins down :math:`N_e` *and* :math:`\mu`
   separately:

   .. math::

      N_e = \frac{\rho}{4r}, \qquad \mu = \frac{\theta}{4 N_e}

   This is the power of the "second pendulum" -- it provides an independent
   constraint that breaks the degeneracy inherent in any single statistic.

.. code-block:: python

   def map_r_bins_to_rho(r_bins, Ne):
       """Convert physical recombination bins to scaled rho values.

       Parameters
       ----------
       r_bins : ndarray
           Physical recombination rates (per generation).
       Ne : float
           Effective population size.

       Returns
       -------
       rho_bins : ndarray
           Scaled recombination rates (4*Ne*r).
       """
       return 4 * Ne * r_bins  # the conversion that links LD to absolute N_e

   # Example: for Ne = 10,000, r = 1e-4 maps to rho = 4
   Ne_values = [5000, 10000, 20000]
   r = 1e-4

   for Ne in Ne_values:
       rho = 4 * Ne * r
       sigma_d2 = 1 / (1 + rho)
       print(f"Ne = {Ne:6d}: rho = {rho:5.1f}, "
             f"sigma_d^2 ~ {sigma_d2:.4f}")


Step 10: Putting It All Together -- LD Inference
==================================================

The complete LD inference workflow mirrors the SFS workflow from
:ref:`demographic_inference`:

1. Compute observed LD statistics from data (Step 8)
2. Define a demographic model
3. For candidate parameters, solve the two-locus moment equations to predict LD
4. Compare predictions to observations via the Gaussian likelihood (Step 6)
5. Optimize parameters; quantify uncertainty via bootstrap

.. code-block:: python

   import moments.LD

   def ld_inference_demo():
       """Demonstrate LD-based demographic inference."""

       # --- 1. Define the demographic model ---
       def model_func(params, rho, theta):
           """Two-epoch model for LD statistics."""
           nu, T = params
           # Start from equilibrium (two-locus steady state)
           y = moments.LD.LDstats(
               moments.LD.Numerics.steady_state(rho, theta=theta),
               num_pops=1, pop_ids=["pop0"]
           )
           # Integrate the two-locus moment equations through the size change
           y.integrate([nu], T, rho=rho, theta=theta)
           return y

       # --- 2. Generate "data" ---
       theta = 0.001
       rho_values = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

       # True parameters: a bottleneck
       nu_true, T_true = 0.1, 0.05

       y_data = model_func([nu_true, T_true], rho_values, theta)

       # --- 3. Compute model for candidate parameters ---
       # (Grid search for demonstration; in practice use gradient-based optimizer)
       print("Searching for best-fit parameters...")
       best_ll = -np.inf
       best_params = None

       for nu in [0.05, 0.1, 0.2, 0.5, 1.0]:
           for T in [0.01, 0.02, 0.05, 0.1, 0.2]:
               y_model = model_func([nu, T], rho_values, theta)

               # Simple sum-of-squares comparison
               # (In practice, use Gaussian likelihood with bootstrap covariance)
               ss = sum((y_data.D2(pops="pop0", r_bin=i) -
                         y_model.D2(pops="pop0", r_bin=i))**2
                        for i in range(len(rho_values)))
               ll = -ss  # proxy for likelihood (lower SS = better fit)

               if ll > best_ll:
                   best_ll = ll
                   best_params = (nu, T)

       print(f"True:      nu={nu_true}, T={T_true}")
       print(f"Best grid: nu={best_params[0]}, T={best_params[1]}")

   ld_inference_demo()

.. admonition:: Probability Aside -- Joint SFS + LD inference

   The most powerful analyses combine *both* the SFS and LD.  Since the SFS
   likelihood (Poisson) and the LD likelihood (Gaussian) involve different
   data, they can be summed:

   .. math::

      \ell_{\text{joint}} = \ell_{\text{SFS}} + \ell_{\text{LD}}

   This is a **composite likelihood**: it treats the SFS and LD as
   independent (they are not perfectly so, but the approximation works well
   in practice).  The joint approach is particularly valuable because the
   SFS constrains :math:`\theta`-dependent parameters while LD constrains
   :math:`\rho`-dependent parameters, and together they resolve degeneracies
   that neither can break alone.


Exercises
=========

.. admonition:: Exercise 1: LD decay curves

   Using ``moments.LD``, compute the LD decay curve (:math:`\sigma_d^2` vs.
   :math:`\rho`) for: (a) constant size, (b) 10x expansion 0.1 time units ago,
   (c) 10x bottleneck 0.05 time units ago. Plot all three on the same axes.
   How do the shapes differ?

.. admonition:: Exercise 2: Admixture detection

   Simulate two populations that diverged at :math:`T = 0.5`, then one received
   10% admixture from the other at :math:`T = 0.01`. Compute :math:`\sigma_{Dz}`
   for the admixed population. Compare to a population that received no admixture.
   At what recombination distances is admixture most detectable?

.. admonition:: Exercise 3: Joint SFS + LD inference

   For a two-epoch model, show that LD constrains :math:`N_e` (through the
   :math:`r \to \rho` mapping) while the SFS constrains :math:`\theta / N_e`.
   Demonstrate that jointly fitting both gives tighter parameter estimates than
   either alone.

.. admonition:: Exercise 4: Cross-population LD

   For two populations that split at different times (:math:`T = 0.01, 0.1, 1.0`),
   compute the cross-population LD correlation
   :math:`E[D_0 D_1] / \sqrt{E[D_0^2] E[D_1^2]}` as a function of :math:`\rho`.
   How does split time affect the correlation?
