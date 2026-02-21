.. _the_frequency_spectrum:

===========================
The Site Frequency Spectrum
===========================

   *The escapement of the mechanism: the summary statistic that makes everything tick.*

The site frequency spectrum (SFS) is the foundation on which all of ``moments``
is built. Before we can infer demographic history, we need to deeply understand
what the SFS is, what it looks like under simple models, and how to manipulate it.

In the watch metaphor introduced in :ref:`moments_overview`, the SFS is **the
dial face** -- the visible read-out from which a skilled horologist (or a
likelihood optimizer) can deduce the entire hidden gear train of population
history.  ``moments`` reads time from the hand positions on this dial without
ever opening the case.


Step 1: What Is the SFS?
==========================

Imagine you've sequenced :math:`n` haploid chromosomes from a population. At each
**segregating site** (a position where not all chromosomes carry the same allele),
you count how many chromosomes carry the **derived** (mutant) allele. This count
:math:`j` can range from 1 to :math:`n-1` (if :math:`j = 0` or :math:`j = n`,
the site is monomorphic -- not segregating).

The SFS is a histogram: :math:`\text{SFS}[j]` = number of segregating sites with
derived allele count :math:`j`.

.. admonition:: Probability Aside -- Why a histogram captures everything

   Under the infinite-sites mutation model, every mutation occurs at a fresh
   genomic position, so each segregating site is the product of exactly one
   mutation event.  Because mutations at different sites are independent, the
   *joint* probability of the whole data set factors into a product of
   per-site terms, each depending only on the allele count :math:`j` at that
   site.  Summing the per-site log-likelihoods groups sites by their count,
   producing the SFS.  Formally, the SFS is a **sufficient statistic** for
   the demographic parameters under the Poisson Random Field model (see
   :ref:`demographic_inference`).  No information is lost by collapsing the
   full data into this histogram.

.. code-block:: python

   import numpy as np

   def compute_sfs(genotype_matrix, n):
       """Compute the SFS from a genotype matrix.

       Parameters
       ----------
       genotype_matrix : ndarray of shape (L, n)
           Each row is a site, each column a haploid chromosome.
           Entries are 0 (ancestral) or 1 (derived).
       n : int
           Number of chromosomes.

       Returns
       -------
       sfs : ndarray of shape (n+1,)
           sfs[j] = number of sites with derived allele count j.
       """
       sfs = np.zeros(n + 1, dtype=int)
       for site in genotype_matrix:
           j = int(site.sum())  # derived allele count at this site
           sfs[j] += 1          # increment the histogram bin for count j
       return sfs

   # Example: 5 chromosomes, 10 segregating sites
   np.random.seed(42)
   n = 5
   # Simulate: each site has a random derived allele count from 1 to n-1
   genotypes = np.zeros((10, n), dtype=int)
   for i in range(10):
       j = np.random.randint(1, n)  # pick a random derived count
       positions = np.random.choice(n, j, replace=False)  # choose which chromosomes carry it
       genotypes[i, positions] = 1   # mark those chromosomes as derived

   sfs = compute_sfs(genotypes, n)
   print(f"SFS: {sfs}")
   print(f"Segregating sites: {sfs[1:-1].sum()}")
   # sfs[0] = monomorphic ancestral, sfs[n] = monomorphic derived

**Intuition**: The SFS is a fingerprint of the evolutionary process. Every force
that shapes genetic variation -- drift, mutation, selection, demography -- leaves
a characteristic signature in the SFS. Learning to read these signatures is what
demographic inference is all about.

The Multi-Population SFS
-------------------------

When you have samples from :math:`p` populations, the SFS becomes a :math:`p`-dimensional
histogram. For two populations with sample sizes :math:`n_1` and :math:`n_2`:

.. math::

   \text{SFS}[j_1, j_2] = \text{number of sites with derived allele count }
   j_1 \text{ in pop 1 and } j_2 \text{ in pop 2}

The shape is :math:`(n_1 + 1) \times (n_2 + 1)`. The entry :math:`\text{SFS}[3, 7]`
means "3 derived copies in population 1, 7 derived copies in population 2."

Think of a multi-population SFS as a watch with *multiple dials* -- one axis
per population.  A single-population SFS tells you how one hand is positioned;
the joint SFS tells you how *all* hands relate to each other, revealing shared
ancestry and migration.

.. code-block:: python

   def compute_joint_sfs(genotypes_pop1, genotypes_pop2, n1, n2):
       """Compute the joint SFS for two populations.

       Parameters
       ----------
       genotypes_pop1 : ndarray of shape (L, n1)
           Binary genotype matrix for population 1.
       genotypes_pop2 : ndarray of shape (L, n2)
           Binary genotype matrix for population 2.
       n1, n2 : int
           Sample sizes.

       Returns
       -------
       sfs : ndarray of shape (n1+1, n2+1)
           Joint SFS: sfs[j1, j2] = number of sites with
           j1 derived in pop1 and j2 derived in pop2.
       """
       L = genotypes_pop1.shape[0]
       sfs = np.zeros((n1 + 1, n2 + 1), dtype=int)
       for i in range(L):
           j1 = int(genotypes_pop1[i].sum())  # derived count in pop 1
           j2 = int(genotypes_pop2[i].sum())  # derived count in pop 2
           sfs[j1, j2] += 1                   # increment the 2D histogram
       return sfs

With the dial face defined, we now turn to the marks etched on it -- the
expected SFS under the simplest possible model.


Step 2: The Neutral Expectation
================================

Under the simplest model -- constant population size, no selection, no migration,
infinite sites mutation model -- the expected SFS has a beautiful closed form.

The expected number of segregating sites with derived allele count :math:`j` in a
sample of :math:`n` chromosomes is:

.. math::

   E[\text{SFS}[j]] = \frac{\theta}{j}, \quad j = 1, 2, \ldots, n-1

where :math:`\theta = 4N_e \mu L` is the population-scaled mutation rate across
:math:`L` sites.

.. admonition:: Probability Aside -- The :math:`1/j` law intuitively

   Why does the neutral SFS fall off as :math:`1/j`?  Picture a population of
   :math:`2N` chromosomes evolving forward in time.  A new mutation begins
   life on a single chromosome (:math:`j = 1`).  For it to ever reach count
   :math:`j`, it must survive drift for many generations.  The probability
   that a neutral mutation ever reaches frequency :math:`j/n` is roughly
   proportional to :math:`1/j` (this is a consequence of the harmonic
   structure of the coalescent, as derived below).  Because most mutations
   are quickly lost, the SFS is heavily tilted toward singletons (:math:`j=1`),
   with each subsequent bin holding progressively fewer sites.

**Why** :math:`1/j` **?** This is one of the most fundamental results in population
genetics, and it's worth deriving carefully.

Derivation from the coalescent
-------------------------------

Consider a single segregating site. It arose as a mutation somewhere on the
genealogy of the :math:`n` sampled chromosomes. Under the infinite-sites model,
each mutation occurs on exactly one branch of the genealogy.

The key insight: **a mutation that occurred on a branch subtending** :math:`j`
**leaves produces a derived allele count of** :math:`j`.

In a coalescent tree with :math:`n` leaves, the total branch length at the level
where there are :math:`k` lineages is :math:`k \cdot T_k`, where :math:`T_k` is
the waiting time for the next coalescence when there are :math:`k` lineages:

.. math::

   T_k \sim \text{Exp}\left(\binom{k}{2}\right), \quad E[T_k] = \frac{2}{k(k-1)}

The total branch length subtending exactly :math:`j` leaves: we need to count how
many branches at each level subtend exactly :math:`j` leaves. This is complex in
general, but the total **expected** branch length leading to allele count :math:`j`
has a remarkably simple form: :math:`2/j` (in coalescent time units of :math:`2N_e`
generations).

Since mutations arrive as a Poisson process with rate :math:`\theta/2` per unit of
coalescent branch length, the expected number of mutations creating allele count
:math:`j` is:

.. math::

   E[\text{SFS}[j]] = \frac{\theta}{2} \cdot \frac{2}{j} = \frac{\theta}{j}

**Verify**: The total expected number of segregating sites is:

.. math::

   S = \sum_{j=1}^{n-1} \frac{\theta}{j} = \theta \cdot H_{n-1}

where :math:`H_{n-1} = \sum_{j=1}^{n-1} \frac{1}{j}` is the :math:`(n-1)`-th
harmonic number. This is the classical result :math:`E[S] = \theta \cdot a_n`
where :math:`a_n = H_{n-1}` is Watterson's :math:`a`.

.. admonition:: Calculus Aside -- The harmonic number and its logarithmic growth

   The harmonic series :math:`H_n = \sum_{k=1}^{n} 1/k` grows like
   :math:`\ln n + \gamma`, where :math:`\gamma \approx 0.5772` is the
   Euler--Mascheroni constant.  This means the total number of segregating
   sites :math:`S \approx \theta \ln n` -- it grows only *logarithmically*
   with sample size.  Doubling your sample from 50 to 100 chromosomes adds
   roughly :math:`\theta \ln 2 \approx 0.69\,\theta` new segregating sites,
   not twice as many.  This slow growth is why even modest sample sizes
   capture much of the variation in a population.

.. code-block:: python

   def expected_sfs_neutral(n, theta=1.0):
       """Expected SFS under the standard neutral model.

       Parameters
       ----------
       n : int
           Haploid sample size.
       theta : float
           Population-scaled mutation rate (4*Ne*mu*L).

       Returns
       -------
       sfs : ndarray of shape (n+1,)
           Expected counts. sfs[0] and sfs[n] are 0 (no fixed sites
           under the infinite-sites model with theta as total rate).
       """
       sfs = np.zeros(n + 1)
       for j in range(1, n):
           sfs[j] = theta / j  # the 1/j law: each bin j gets theta/j expected sites
       return sfs

   # Plot the neutral SFS
   n = 50
   theta = 1000  # genome-wide (say, 10 Mb at mu = 1e-8, Ne = 10000)
   sfs_neutral = expected_sfs_neutral(n, theta)

   print("Expected neutral SFS (first 10 entries):")
   for j in range(1, 11):
       print(f"  SFS[{j:2d}] = {sfs_neutral[j]:.1f}")

   total_seg = sfs_neutral[1:n].sum()
   harmonic = sum(1/j for j in range(1, n))
   print(f"\nTotal segregating sites: {total_seg:.1f}")
   print(f"theta * H_(n-1) = {theta * harmonic:.1f}")
   print(f"Match: {np.isclose(total_seg, theta * harmonic)}")

Now that we know what the "undisturbed" dial looks like, let us see how
demographic events shift the hands.


Step 3: How Demography Distorts the SFS
=========================================

The :math:`1/j` spectrum is the baseline. Real populations don't have constant size,
so their SFS deviates from this baseline in informative ways.

Population expansion
---------------------

After an expansion, the population is large. New mutations arise frequently (because
there are many individuals), but they haven't had time to drift to high frequency.
**Result**: excess of rare variants, SFS shifted toward low :math:`j`.

.. code-block:: python

   def sfs_after_expansion_simulation(n, theta, nu, T, num_reps=100000):
       """Simulate the SFS after a population expansion using msprime-style logic.

       This uses the coalescent with piecewise-constant population size.

       Parameters
       ----------
       n : int
           Sample size.
       theta : float
           Per-site mutation rate (4*Ne*mu) for the REFERENCE population.
       nu : float
           Expansion factor (new size = nu * N_ref).
       T : float
           Time since expansion (in units of 2*N_ref generations).
       num_reps : int
           Number of independent loci to simulate.

       Returns
       -------
       sfs : ndarray of shape (n+1,)
           Simulated SFS (averaged over loci).
       """
       sfs = np.zeros(n + 1)

       for _ in range(num_reps):
           # Simulate a coalescent tree with expansion
           # Time 0 to T: population size nu*N_ref
           # Time > T: population size N_ref
           times = [0.0]  # coalescence times
           k = n  # current number of lineages

           t = 0.0
           while k > 1:
               # Coalescence rate: k*(k-1)/2, scaled by 1/nu during expansion
               if t < T:
                   rate = k * (k - 1) / (2 * nu)  # larger nu => slower coalescence
                   # Time to next event in current epoch
                   wait = np.random.exponential(1 / rate)
                   if t + wait < T:
                       t += wait
                       k -= 1
                       times.append(t)
                   else:
                       t = T  # move to ancestral epoch
               else:
                   rate = k * (k - 1) / 2  # ancestral size = 1 (reference)
                   wait = np.random.exponential(1 / rate)
                   t += wait
                   k -= 1
                   times.append(t)

           # Place a mutation on a random branch
           # Each branch subtends some number of leaves j
           # For simplicity, pick j proportional to branch length
           # (This is a simplified version -- for the full version, you'd
           # need to track the tree topology)
           j = np.random.randint(1, n)  # placeholder
           sfs[j] += 1

       return sfs / num_reps * theta

**Intuition**: Think of it like a city that suddenly grew. Lots of new families
(mutations) appeared recently, but none of them are widespread yet. The phone book
has many names that appear only once or twice.

Population bottleneck
----------------------

During a bottleneck, the population is small. Genetic drift is strong: alleles
either get lost or drift to high frequency quickly. Rare variants are removed,
intermediate-frequency variants become over-represented.
**Result**: flattened SFS, relatively more variants at intermediate :math:`j`.

**Intuition**: Like a small village where everyone is related. There's less variety,
but what variety exists tends to be at moderate frequency because drift pushed
alleles away from the extremes.

.. admonition:: Probability Aside -- Reading demography from the SFS shape

   The :math:`1/j` spectrum is perfectly monotone-decreasing.  Any deviation
   from this monotone shape carries demographic signal:

   * **Concave-up** (steeper than :math:`1/j` at low :math:`j`) suggests a
     recent expansion -- an excess of singletons.
   * **Concave-down** (flatter than :math:`1/j`) suggests a recent
     bottleneck -- singletons were lost to drift, leaving proportionally
     more intermediate-frequency variants.
   * A U-shaped uptick at *both* tails hints at population structure or
     ancestral misidentification (see :ref:`demographic_inference`).

   These shape differences are what the likelihood optimizer quantifies:
   it searches for the demographic parameters whose predicted SFS shape
   best matches the observed one.

These signatures are the reason the SFS -- our dial face -- can distinguish
so many different histories.  Next, we address a practical complication:
what if we cannot determine which allele is ancestral?


Step 4: Folding the SFS
=========================

Sometimes we don't know which allele is ancestral and which is derived. Without an
outgroup genome to polarize mutations, we can only tell the **minor** allele (less
common) from the **major** allele (more common). The **folded SFS** uses minor
allele counts:

.. math::

   \text{SFS}_{\text{folded}}[j] = \text{SFS}[j] + \text{SFS}[n - j], \quad j = 1, \ldots, \lfloor n/2 \rfloor

For even :math:`n`, the entry at :math:`j = n/2` is not doubled (it's its own mirror).

.. code-block:: python

   def fold_sfs(sfs):
       """Fold an unfolded SFS into a minor allele frequency spectrum.

       Parameters
       ----------
       sfs : ndarray of shape (n+1,)
           Unfolded SFS.

       Returns
       -------
       folded : ndarray of shape (n//2 + 1,)
           Folded SFS. folded[0] is unused (monomorphic).
       """
       n = len(sfs) - 1
       folded = np.zeros(n // 2 + 1)
       for j in range(1, n // 2 + 1):
           if j == n - j:
               folded[j] = sfs[j]         # at the midpoint, don't double-count
           else:
               folded[j] = sfs[j] + sfs[n - j]  # sum the mirror bins
       return folded

   # Example
   n = 10
   theta = 100
   sfs = expected_sfs_neutral(n, theta)
   folded = fold_sfs(sfs)

   print("Unfolded SFS:")
   for j in range(1, n):
       print(f"  SFS[{j}] = {sfs[j]:.2f}")

   print("\nFolded SFS:")
   for j in range(1, n // 2 + 1):
       print(f"  SFS_folded[{j}] = {folded[j]:.2f}")

.. admonition:: When to fold

   Use the **unfolded** SFS when you have a reliable outgroup to determine
   ancestral states. Use the **folded** SFS when you don't. Unfolded contains
   more information (it distinguishes expansions from contractions that look
   similar when folded), but an incorrect polarization is worse than no
   polarization at all.

   In practice, even with an outgroup, some fraction of sites (typically 1-3%)
   have the ancestral state misidentified. ``moments`` can fit an ancestral
   misidentification parameter to account for this.

Folding tells us about the *minor allele count*, but another operation lets us
compare data sets of different sizes: **projection**.


Step 5: Projection -- Downsampling the SFS
============================================

Often we want to compare SFS from samples of different sizes, or we have missing
data at some sites. **Projection** downsamples an SFS from sample size :math:`n` to
a smaller size :math:`n'` without resequencing.

The math: if a site has derived allele count :math:`j` in a sample of :math:`n`,
what's the probability of seeing count :math:`j'` in a subsample of :math:`n'`?
This is a hypergeometric sampling problem:

.. math::

   P(j' \mid j, n, n') = \frac{\binom{j}{j'}\binom{n-j}{n'-j'}}{\binom{n}{n'}}

The projected SFS is:

.. math::

   \text{SFS}'[j'] = \sum_{j=j'}^{n-(n'-j')} \text{SFS}[j] \cdot
   \frac{\binom{j}{j'}\binom{n-j}{n'-j'}}{\binom{n}{n'}}

.. admonition:: Probability Aside -- Hypergeometric sampling explained

   The hypergeometric distribution models drawing balls from an urn *without
   replacement*.  Imagine an urn containing :math:`n` balls, :math:`j` of
   which are red (derived) and :math:`n - j` blue (ancestral).  You draw
   :math:`n'` balls without replacement.  The probability of getting exactly
   :math:`j'` red balls is the hypergeometric formula above.  Projection
   applies this logic to every allele-count class in the SFS, producing the
   expected SFS for a smaller subsample.

**Intuition**: If you have 100 marbles in a bag, 30 red and 70 blue, and you draw
20, the probability of drawing :math:`k` red marbles follows the hypergeometric
distribution. Projection does exactly this for each allele frequency class.

.. code-block:: python

   from scipy.special import comb

   def project_sfs(sfs, n_new):
       """Project an SFS to a smaller sample size.

       Parameters
       ----------
       sfs : ndarray of shape (n+1,)
           Original SFS with sample size n.
       n_new : int
           Target sample size (n_new < n).

       Returns
       -------
       projected : ndarray of shape (n_new+1,)
           Projected SFS.
       """
       n = len(sfs) - 1
       projected = np.zeros(n_new + 1)

       for j in range(0, n + 1):
           if sfs[j] == 0:
               continue
           for j_new in range(max(0, j - (n - n_new)), min(j, n_new) + 1):
               # Hypergeometric probability of drawing j_new derived
               # from j derived and n-j ancestral, sample size n_new
               prob = (comb(j, j_new, exact=True) *
                       comb(n - j, n_new - j_new, exact=True) /
                       comb(n, n_new, exact=True))
               projected[j_new] += sfs[j] * prob  # accumulate weighted counts

       return projected

   # Example: project from n=50 to n=20
   n = 50
   theta = 500
   sfs_50 = expected_sfs_neutral(n, theta)

   sfs_20 = project_sfs(sfs_50, 20)
   sfs_20_direct = expected_sfs_neutral(20, theta)

   print("Projected vs. direct computation (first 10 entries):")
   print(f"{'j':>3} {'Projected':>12} {'Direct':>12} {'Match':>8}")
   for j in range(1, 11):
       match = np.isclose(sfs_20[j], sfs_20_direct[j], rtol=1e-10)
       print(f"{j:3d} {sfs_20[j]:12.4f} {sfs_20_direct[j]:12.4f} {'Y' if match else 'N':>8}")

.. admonition:: Verify: projection preserves the neutral shape

   Under the neutral model, the expected SFS is :math:`\theta/j` regardless of
   sample size. So projecting the neutral SFS from :math:`n = 50` to :math:`n = 20`
   should give exactly :math:`\theta/j` for the projected sample. The code above
   confirms this: the projected and directly-computed SFS match perfectly.

   This is not a coincidence -- it follows from the exchangeability of the
   coalescent. Subsampling a coalescent of :math:`n` lineages gives a coalescent
   of :math:`n'` lineages, so the expected allele frequency spectrum is preserved.

With the SFS itself, folding, and projection in hand, we can now extract
classical summary statistics from the dial face.


Step 6: Nucleotide Diversity and Summary Statistics
=====================================================

The SFS contains enough information to compute all classical summary statistics
of genetic variation. These statistics compress the SFS into single numbers that
highlight specific features.

Watterson's :math:`\theta_W`
-----------------------------

.. math::

   \hat{\theta}_W = \frac{S}{a_n}, \quad \text{where } S = \sum_{j=1}^{n-1} \text{SFS}[j], \quad
   a_n = \sum_{j=1}^{n-1} \frac{1}{j}

:math:`S` is the total number of segregating sites. Dividing by :math:`a_n` (the
:math:`(n-1)`-th harmonic number) gives an estimator of :math:`\theta` that is
unbiased under the neutral model.

**Intuition**: :math:`a_n` is the expected total branch length in coalescent units.
More branches means more opportunity for mutations. Normalizing by :math:`a_n`
cancels this effect.

.. code-block:: python

   def watterson_theta(sfs):
       """Watterson's estimator of theta from an SFS."""
       n = len(sfs) - 1
       S = sfs[1:n].sum()                    # total number of segregating sites
       a_n = sum(1/j for j in range(1, n))    # (n-1)-th harmonic number
       return S / a_n                          # unbiased estimator of theta

Nucleotide diversity :math:`\pi`
---------------------------------

.. math::

   \hat{\pi} = \frac{1}{\binom{n}{2}} \sum_{j=1}^{n-1} j(n-j) \cdot \text{SFS}[j]

This is the average number of pairwise differences: pick two chromosomes at random,
count how many sites differ between them.

**Why** :math:`j(n-j)` **?** If :math:`j` chromosomes carry the derived allele and
:math:`n-j` carry the ancestral, then the number of pairs that differ at this site
is :math:`j \times (n-j)`. We divide by :math:`\binom{n}{2}` (total pairs) to get
a per-pair average.

.. admonition:: Calculus Aside -- Why :math:`\pi = \theta` under neutrality

   Substituting the neutral expectation :math:`\text{SFS}[j] = \theta/j` into
   the formula for :math:`\hat{\pi}`:

   .. math::

      \hat{\pi} = \frac{1}{\binom{n}{2}} \sum_{j=1}^{n-1} j(n-j) \cdot
      \frac{\theta}{j}
      = \frac{\theta}{\binom{n}{2}} \sum_{j=1}^{n-1} (n - j)
      = \frac{\theta}{\binom{n}{2}} \cdot \frac{n(n-1)}{2} = \theta

   The cancellation is exact -- every term telescopes.  This algebraic miracle
   is why both Watterson's estimator (which weights every SFS bin equally)
   and nucleotide diversity (which weights by :math:`j(n-j)`) converge to the
   same value under neutrality.  When they *disagree*, the discrepancy
   signals a departure from the standard neutral model -- the basis for
   Tajima's D below.

.. code-block:: python

   def nucleotide_diversity(sfs):
       """Compute nucleotide diversity (pi) from an SFS."""
       n = len(sfs) - 1
       pi = 0.0
       for j in range(1, n):
           pi += j * (n - j) * sfs[j]  # j*(n-j) = number of differing pairs at this site
       pi /= n * (n - 1) / 2  # divide by total number of pairs = binom(n,2)
       return pi

   # Under the neutral model, both estimators should give theta
   n = 50
   theta = 1000
   sfs = expected_sfs_neutral(n, theta)
   print(f"theta (true):     {theta:.2f}")
   print(f"theta_W:          {watterson_theta(sfs):.2f}")
   print(f"pi:               {nucleotide_diversity(sfs):.2f}")

.. admonition:: Verify: both estimators agree for neutral SFS

   Under the standard neutral model, :math:`E[\hat{\theta}_W] = E[\hat{\pi}] = \theta`.
   Let's verify:

   .. math::

      \hat{\pi} = \frac{1}{\binom{n}{2}} \sum_{j=1}^{n-1} j(n-j) \cdot \frac{\theta}{j}
      = \frac{\theta}{\binom{n}{2}} \sum_{j=1}^{n-1} (n-j)
      = \frac{\theta}{\binom{n}{2}} \cdot \frac{n(n-1)}{2} = \theta \quad \checkmark

Tajima's D
-----------

.. math::

   D = \frac{\hat{\pi} - \hat{\theta}_W}{\sqrt{\text{Var}(\hat{\pi} - \hat{\theta}_W)}}

Tajima's D measures the **difference** between the two estimators. Under the neutral
model, :math:`D \approx 0`. Deviations indicate:

- :math:`D < 0`: excess of rare variants -- population expansion or purifying selection
- :math:`D > 0`: excess of intermediate-frequency variants -- bottleneck or balancing selection

.. code-block:: python

   def tajimas_d(sfs):
       """Compute Tajima's D from an SFS.

       Uses the standard normalization from Tajima (1989).
       """
       n = len(sfs) - 1
       S = sfs[1:n].sum()       # total segregating sites
       if S == 0:
           return 0.0

       pi = nucleotide_diversity(sfs)  # pairwise-difference estimator of theta
       theta_w = watterson_theta(sfs)  # segregating-sites estimator of theta

       # Tajima's normalization constants (derived from coalescent variance)
       a1 = sum(1/i for i in range(1, n))
       a2 = sum(1/i**2 for i in range(1, n))

       b1 = (n + 1) / (3 * (n - 1))
       b2 = 2 * (n**2 + n + 3) / (9 * n * (n - 1))

       c1 = b1 - 1/a1
       c2 = b2 - (n + 2) / (a1 * n) + a2 / a1**2

       e1 = c1 / a1
       e2 = c2 / (a1**2 + a2)

       var = e1 * S + e2 * S * (S - 1)  # variance of pi - theta_W under neutrality
       if var <= 0:
           return 0.0

       return (pi - theta_w) / np.sqrt(var)  # standardized test statistic

Having read the static dial, we now let ``moments`` compute the dial for us.


Step 7: Using moments to Compute the SFS
==========================================

Now that we understand the SFS from first principles, let's see how ``moments``
computes it. The package represents the SFS as a ``moments.Spectrum`` object --
a masked NumPy array with metadata.

.. code-block:: python

   import moments

   # --- Standard neutral model ---
   # moments.Demographics1D.snm returns the SFS for the "standard neutral model"
   # (constant population size, no selection) normalized to theta = 1.
   # This is the equilibrium solution of the moment equations (see moment_equations).
   n = 20
   fs_neutral = moments.Demographics1D.snm([n])

   # This is the expected SFS with theta = 1
   print("Standard neutral model SFS (theta=1):")
   for j in range(1, 6):
       print(f"  SFS[{j}] = {fs_neutral[j]:.6f}  (expected: {1/j:.6f})")

   # --- Two-epoch model: instantaneous expansion ---
   # Population at size 1 (= N_ref) until time T ago, then jumps to size nu.
   # This is like suddenly swapping in a larger mainspring -- the watch
   # runs differently from the moment of the swap onward.
   def two_epoch(params, n):
       nu, T = params
       # Start from equilibrium (standard neutral model)
       fs = moments.Demographics1D.snm([n])
       # Integrate forward through the size change
       # Under the hood, this solves the moment-equation ODEs (see moment_equations)
       fs.integrate([nu], T)
       return fs

   # Example: 10-fold expansion, 0.1 * 2*Ne generations ago
   fs_expansion = two_epoch([10.0, 0.1], n)
   print("\nAfter 10-fold expansion:")
   print(f"  SFS[1] = {fs_expansion[1]:.4f} (neutral: {1.0:.4f})")
   print(f"  SFS[5] = {fs_expansion[5]:.4f} (neutral: {1/5:.4f})")
   print(f"  Tajima's D = {fs_expansion.Tajima_D():.4f} (expect < 0)")

   # --- Bottleneck + recovery ---
   def bottleneck_recovery(params, n):
       nu_B, T_B, nu_R, T_R = params
       fs = moments.Demographics1D.snm([n])
       fs.integrate([nu_B], T_B)   # bottleneck phase: small population, strong drift
       fs.integrate([nu_R], T_R)   # recovery phase: return to reference size
       return fs

   fs_bottle = bottleneck_recovery([0.05, 0.02, 1.0, 0.1], n)
   print(f"\nAfter bottleneck + recovery:")
   print(f"  Tajima's D = {fs_bottle.Tajima_D():.4f} (expect > 0 during bottleneck)")

.. admonition:: What ``integrate`` does under the hood

   When you call ``fs.integrate([nu], T)``, ``moments`` solves the system of
   moment equations (ODEs) that we'll derive in the next chapter. Each SFS entry
   evolves according to how drift, mutation, and selection change it over the
   time interval :math:`T`. The population size :math:`\nu` controls the strength
   of drift: smaller :math:`\nu` means stronger drift, larger :math:`\nu` means
   weaker drift.

   If the SFS is the dial face, then ``integrate`` is the act of winding the
   watch forward by :math:`T` time units with a mainspring of size :math:`\nu`.
   The hands (SFS entries) move according to the ODEs governing the gear train
   (:ref:`moment_equations`).


Step 8: Two-Population SFS with moments
==========================================

For two populations, the SFS is a 2D matrix. ``moments`` handles population splits,
migration, and independent size changes.

.. code-block:: python

   import moments

   def isolation_with_migration(params, ns):
       """Two-population isolation-with-migration model.

       Parameters
       ----------
       params : (nu1, nu2, T, m12, m21)
           nu1, nu2 : relative sizes of pop 1 and pop 2 after split
           T : time since split (in 2*Ne generations)
           m12 : migration rate from pop 2 into pop 1 (2*Ne*m)
           m21 : migration rate from pop 1 into pop 2 (2*Ne*m)
       ns : (n1, n2)
           Sample sizes.

       Returns
       -------
       fs : moments.Spectrum
           Joint SFS.
       """
       nu1, nu2, T, m12, m21 = params

       # Start from equilibrium with combined sample size
       fs = moments.Demographics1D.snm([sum(ns)])

       # Split into two populations (like a single clock being separated
       # into two clocks that now tick independently)
       fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])

       # Integrate with migration
       # Migration matrix: M[i,j] = rate from j into i
       m = np.array([[0, m12],
                      [m21, 0]])
       fs.integrate([nu1, nu2], T, m=m)  # both populations evolve together with gene flow

       return fs

   # Example: symmetric divergence
   ns = (10, 10)
   fs = isolation_with_migration([1.0, 1.0, 0.5, 1.0, 1.0], ns)

   print("Joint SFS (first 5x5 corner):")
   for j1 in range(5):
       row = [f"{fs[j1, j2]:.3f}" for j2 in range(5)]
       print(f"  {' '.join(row)}")

   # F_ST: population differentiation
   print(f"\nF_ST = {fs.Fst():.4f}")

We now have the dial face -- both its theoretical shape and its practical
computation by ``moments``.  In the next chapter we pry open the case and
examine the gear train: the moment equations that make ``fs.integrate()`` work.

Next: :ref:`moment_equations` -- we derive the ODEs that power ``fs.integrate()``.
