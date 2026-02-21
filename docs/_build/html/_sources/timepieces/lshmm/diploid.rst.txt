.. _diploid:

=======================
The Diploid Extension
=======================

   *Two watches, ticking together: when each individual carries two haplotypes.*

In the previous chapters, we built a complete haploid Li & Stephens HMM -- the
template mechanism (:ref:`copying_model`) and the gear train
(:ref:`haploid_algorithms`). That machinery handles one haplotype at a time.
But humans (and most eukaryotes) are **diploid**: each individual carries
**two** copies of each chromosome.

When we observe genotype data (allele dosages: 0, 1, or 2), we see the **sum**
of the two haplotypes but not which allele is on which chromosome. This creates
a harder inference problem: we must simultaneously infer two copying paths, one
for each chromosome -- like two watches ticking together, sharing the same
observed ticks but driven by independent (though correlated) gear trains.

The diploid Li & Stephens model extends the haploid model by running **two
copying processes in parallel**, one for each haplotype of the query individual.

.. note::

   **Prerequisites.** This chapter builds on all preceding chapters:

   - The :ref:`HMM chapter <hmms>` for the general HMM framework.
   - The :ref:`copying model chapter <copying_model>` for the Li-Stephens
     transition and emission probabilities.
   - The :ref:`haploid algorithms chapter <haploid_algorithms>` for the
     forward, backward, and Viterbi algorithms in the haploid case.

   We will extend each of these components to the diploid setting. The key
   new concepts are the :math:`n^2` state space, the four transition types,
   and the genotype emission probabilities.


Step 1: The Diploid State Space
=================================

In the haploid model, the hidden state at site :math:`\ell` is a single index
:math:`Z_\ell \in \{1, \ldots, n\}` -- which reference haplotype is being
copied.

In the diploid model, the hidden state is a **pair** :math:`(Z_\ell^{(1)}, Z_\ell^{(2)})`,
where:

- :math:`Z_\ell^{(1)}` = which reference haplotype is being copied for the
  first chromosome
- :math:`Z_\ell^{(2)}` = which reference haplotype is being copied for the
  second chromosome

The state space is :math:`\{1, \ldots, n\} \times \{1, \ldots, n\}`, giving
:math:`n^2` states. For :math:`n = 100` reference haplotypes, that's 10,000
states.

.. admonition:: Terminology: State Space Explosion

   Going from :math:`n` to :math:`n^2` states is called a **state space
   explosion** -- the cost of modeling two interacting processes. This is a
   recurring theme in computational biology: modeling pairs of things (diploid
   genotypes, pairwise sequence alignment, co-evolution) often squares the
   state space. The :math:`O(n^2)` trick we develop below is the key to
   making this tractable.

.. code-block:: python

   import numpy as np

   # Diploid state space: all pairs (j1, j2) of reference haplotypes
   n = 4  # reference haplotypes
   states = [(j1, j2) for j1 in range(n) for j2 in range(n)]
   print(f"Diploid state space (n={n}): {n**2} states")
   print(f"First few states: {states[:8]}")

The initial distribution is uniform over all pairs:

.. math::

   \pi_{j_1, j_2} = \frac{1}{n^2}

.. admonition:: Probability Aside: Why Uniform Over Pairs?

   Just as the haploid model uses a uniform prior :math:`\pi_j = 1/n` based
   on exchangeability of lineages under the coalescent (see
   :ref:`copying_model`), the diploid model uses :math:`\pi_{j_1,j_2} = 1/n^2`
   because the two chromosomes are independent *a priori*. The uniform pair
   prior is the product of two independent uniform priors:
   :math:`\pi_{j_1,j_2} = \pi_{j_1} \cdot \pi_{j_2} = (1/n)(1/n) = 1/n^2`.

With the state space defined, we now derive how states transition between
adjacent sites.


Step 2: Diploid Transitions
==============================

Between adjacent sites, **each** chromosome independently decides whether to
recombine. This gives four cases:

1. **Neither recombines** (probability :math:`(1-r)^2`): Both chromosomes keep
   their copying sources.

2. **Only chromosome 1 recombines** (probability :math:`r(1-r)`): Chromosome 1
   switches to a random haplotype, chromosome 2 stays.

3. **Only chromosome 2 recombines** (probability :math:`(1-r)r`): Chromosome 1
   stays, chromosome 2 switches.

4. **Both recombine** (probability :math:`r^2`): Both chromosomes switch
   independently.

.. admonition:: Probability Aside: Independence of the Two Chromosomes

   The four cases above arise from the assumption that recombination on the
   two chromosomes is **independent**. This is biologically reasonable: the
   two chromosomes of a diploid individual come from different parents, and
   their recombination events in the next generation are independent meiotic
   processes. Formally, if :math:`R_1` and :math:`R_2` are indicator variables
   for recombination on chromosomes 1 and 2, then:

   .. math::

      P(R_1 = r_1, R_2 = r_2) = P(R_1 = r_1) \cdot P(R_2 = r_2)

   This independence is what allows the diploid transition to **factor** as
   a product of two haploid transitions, which is the key to the
   :math:`O(n^2)` trick.

The transition probability is:

.. math::

   A_{(j_1,j_2)(k_1,k_2)} = P(Z^{(1)}_\ell = k_1, Z^{(2)}_\ell = k_2 \mid Z^{(1)}_{\ell-1} = j_1, Z^{(2)}_{\ell-1} = j_2)

Since the two chromosomes are independent, this factors:

.. math::

   A_{(j_1,j_2)(k_1,k_2)} = A^{(1)}_{j_1 k_1} \cdot A^{(2)}_{j_2 k_2}

where each :math:`A^{(c)}` is the haploid transition:

.. math::

   A^{(c)}_{jk} = (1-r)\delta_{jk} + \frac{r}{n}

Expanding the product:

.. math::

   A_{(j_1,j_2)(k_1,k_2)} = \left[(1-r)\delta_{j_1 k_1} + \frac{r}{n}\right] \cdot \left[(1-r)\delta_{j_2 k_2} + \frac{r}{n}\right]

Let's expand this to see the four cases explicitly:

.. math::

   A_{(j_1,j_2)(k_1,k_2)} = \underbrace{(1-r)^2 \delta_{j_1 k_1} \delta_{j_2 k_2}}_{\text{neither switches}}
   + \underbrace{(1-r)\frac{r}{n} \delta_{j_1 k_1}}_{\text{only chr 2 switches}}
   + \underbrace{\frac{r}{n}(1-r) \delta_{j_2 k_2}}_{\text{only chr 1 switches}}
   + \underbrace{\left(\frac{r}{n}\right)^2}_{\text{both switch}}

**Verification**: summing over all :math:`(k_1, k_2)`:

.. math::

   \sum_{k_1, k_2} A_{(j_1,j_2)(k_1,k_2)} = \left[\sum_{k_1} A^{(1)}_{j_1 k_1}\right] \cdot \left[\sum_{k_2} A^{(2)}_{j_2 k_2}\right] = 1 \cdot 1 = 1 \quad \checkmark

.. code-block:: python

   def diploid_transition_prob(j1, j2, k1, k2, r, n):
       """Compute diploid transition probability.

       Parameters
       ----------
       j1, j2 : int
           Previous state (copying sources for chr 1 and chr 2).
       k1, k2 : int
           Next state.
       r : float
           Per-site recombination probability.
       n : int
           Number of reference haplotypes.

       Returns
       -------
       prob : float
       """
       r_n = r / n
       # Factor as product of two independent haploid transitions.
       # t1: haploid transition for chromosome 1 (from j1 to k1)
       t1 = (1 - r) * (j1 == k1) + r_n
       # t2: haploid transition for chromosome 2 (from j2 to k2)
       t2 = (1 - r) * (j2 == k2) + r_n
       # Independence: joint probability is the product
       return t1 * t2

   # Example: verify the four cases
   n, r = 4, 0.1
   r_n = r / n

   no_switch = diploid_transition_prob(0, 1, 0, 1, r, n)
   chr1_switch = diploid_transition_prob(0, 1, 2, 1, r, n)
   chr2_switch = diploid_transition_prob(0, 1, 0, 3, r, n)
   both_switch = diploid_transition_prob(0, 1, 2, 3, r, n)

   print(f"No switch:        {no_switch:.6f}  "
         f"(expected: {(1-r+r_n)**2:.6f})")
   print(f"Only chr 1:       {chr1_switch:.6f}  "
         f"(expected: {r_n*(1-r+r_n):.6f})")
   print(f"Only chr 2:       {chr2_switch:.6f}  "
         f"(expected: {(1-r+r_n)*r_n:.6f})")
   print(f"Both switch:      {both_switch:.6f}  "
         f"(expected: {r_n**2:.6f})")

   # Verify row sums to 1
   total = sum(diploid_transition_prob(0, 1, k1, k2, r, n)
               for k1 in range(n) for k2 in range(n))
   print(f"Row sum: {total:.6f}")  # Should be 1.0


The three transition magnitudes
---------------------------------

Looking at the probabilities, there are really only **three** distinct values,
depending on how many coordinates match:

.. math::

   A_{(j_1,j_2)(k_1,k_2)} = \begin{cases}
   (1-r)^2 + 2(1-r)\frac{r}{n} + \left(\frac{r}{n}\right)^2 & \text{if } k_1 = j_1 \text{ and } k_2 = j_2 \\[6pt]
   (1-r)\frac{r}{n} + \left(\frac{r}{n}\right)^2 & \text{if } k_1 = j_1 \text{ or } k_2 = j_2 \text{ (but not both)} \\[6pt]
   \left(\frac{r}{n}\right)^2 & \text{if } k_1 \neq j_1 \text{ and } k_2 \neq j_2
   \end{cases}

.. admonition:: Probability Aside: Deriving the Three Magnitudes

   To see why there are only three distinct values, note that each coordinate
   contributes either :math:`(1-r) + r/n` (if it matches) or :math:`r/n`
   (if it doesn't). The product of two such factors gives:

   - Both match: :math:`[(1-r)+r/n]^2 = (1-r)^2 + 2(1-r)r/n + (r/n)^2`
   - One matches, one doesn't: :math:`[(1-r)+r/n] \cdot [r/n] = (1-r)r/n + (r/n)^2`
   - Neither matches: :math:`(r/n)^2`

   These are exactly the ``no_switch``, ``single_switch``, and ``double_switch``
   values in the code below.

The lshmm code uses exactly these three values:

.. code-block:: python

   # The three distinct transition magnitudes.
   # no_switch: both chromosomes stay with their current sources
   no_switch = (1 - r)**2 + 2 * (r_n * (1 - r)) + r_n**2
   # single_switch: exactly one chromosome switches
   single_switch = r_n * (1 - r) + r_n**2
   # double_switch: both chromosomes switch
   double_switch = r_n**2

Note that ``no_switch`` can also be written as :math:`(1 - r + r/n)^2`, which
is just the square of the haploid "stay" probability.

With the diploid transitions fully specified, we now turn to the emission side
of the model, which is substantially different from the haploid case.


Step 3: Diploid Emissions
============================

In the diploid model, we don't observe individual alleles on each chromosome.
Instead, we observe the **genotype** -- the sum of alleles across both
chromosomes:

.. math::

   G = h^{(1)} + h^{(2)} \in \{0, 1, 2\}

where :math:`h^{(1)}, h^{(2)} \in \{0, 1\}` are the alleles on each chromosome.

.. admonition:: Terminology: Genotype Dosage

   The **allele dosage** (or simply **genotype**) at a biallelic site is the
   count of the alternate allele: 0 (homozygous reference), 1 (heterozygous),
   or 2 (homozygous alternate). This is the standard encoding in PLINK and
   VCF files. The key challenge for the diploid model is that genotype 1
   (heterozygous) does not tell us which chromosome carries which allele --
   this is the **phase ambiguity** discussed in Step 8.

Given the diploid state :math:`(j_1, j_2)`, the reference genotype is
:math:`H_{j_1} + H_{j_2}` (the sum of the reference alleles for the two copying
sources). The emission probability depends on how the reference genotype
compares to the query genotype.

Deriving the emission table
------------------------------

Let :math:`p = 1 - \mu` (match probability) and :math:`q = \mu` (mutation probability).
Each chromosome independently matches or mutates. The emission probability for
the diploid state :math:`(j_1, j_2)` with reference genotype :math:`g_r` and
query genotype :math:`g_q` is:

**Case 1: Both homozygous, same genotype** (:math:`g_r = g_q \in \{0, 2\}`)

Both chromosomes must match (no mutation on either):

.. math::

   e = p^2 = (1-\mu)^2

**Case 2: Both homozygous, different genotype** (:math:`g_r = 0, g_q = 2` or
:math:`g_r = 2, g_q = 0`)

Both chromosomes must mutate:

.. math::

   e = q^2 = \mu^2

**Case 3: Both heterozygous** (:math:`g_r = g_q = 1`)

Either both match or both mutate (since a mutation on each chromosome flips it,
giving the same heterozygous genotype):

.. math::

   e = p^2 + q^2 = (1-\mu)^2 + \mu^2

.. admonition:: Probability Aside: Why :math:`p^2 + q^2` for Both Heterozygous?

   When both the reference and the query are heterozygous (genotype 1), there
   are two ways to produce the observed genotype:

   1. **Both chromosomes match** their reference sources: chromosome 1 copies
      the "0" allele faithfully, chromosome 2 copies the "1" allele faithfully
      (or vice versa). Probability: :math:`p \cdot p = p^2`.

   2. **Both chromosomes mutate**: chromosome 1 copies "0" but mutates to "1",
      chromosome 2 copies "1" but mutates to "0". The result is still 0+1 = 1.
      Probability: :math:`q \cdot q = q^2`.

   No other combination works. If only one chromosome mutates, the genotype
   would be 0+0=0 or 1+1=2, not 1. Hence :math:`e = p^2 + q^2`.

   Note that :math:`p^2 + q^2 < 1` (since :math:`2pq > 0`), so heterozygous
   observations are slightly less likely than homozygous ones -- reflecting
   the fact that heterozygosity requires a specific arrangement of alleles.

**Case 4: Reference homozygous, query heterozygous** (:math:`g_r \in \{0, 2\}, g_q = 1`)

Exactly one chromosome must mutate (either could be the one):

.. math::

   e = 2pq = 2\mu(1-\mu)

**Case 5: Reference heterozygous, query homozygous** (:math:`g_r = 1, g_q \in \{0, 2\}`)

Exactly one chromosome must mutate:

.. math::

   e = pq = \mu(1-\mu)

**Wait -- why not** :math:`2pq` **in Case 5?** Because in Case 4, the reference
is homozygous (say :math:`0/0`) and the query is :math:`0/1`. Either the first
or second chromosome could be the one that mutated -- two configurations, hence
the factor of 2. In Case 5, the reference is heterozygous (:math:`0/1`) and the
query is homozygous (say :math:`0/0`). The chromosome carrying the "1" must
mutate to "0" -- that's a specific chromosome, so there's only one configuration.

.. admonition:: Probability Aside: The Factor-of-2 Asymmetry Explained

   This asymmetry is a subtle but important point. Let us spell it out
   carefully.

   **Case 4** (ref = 0/0, query = 0/1): Chromosome 1 copies ref allele 0 and
   chromosome 2 copies ref allele 0. We need the query to be 0/1. This
   requires exactly one mutation:

   - Chromosome 1 mutates (0 -> 1), chromosome 2 matches (0 -> 0):
     probability :math:`q \cdot p`
   - Chromosome 1 matches (0 -> 0), chromosome 2 mutates (0 -> 1):
     probability :math:`p \cdot q`

   Total: :math:`2pq`. The factor of 2 comes from the two **interchangeable**
   chromosomes both starting with the same allele.

   **Case 5** (ref = 0/1, query = 0/0): Chromosome 1 copies allele 0 from
   :math:`j_1`, chromosome 2 copies allele 1 from :math:`j_2`. We need
   query = 0/0:

   - Chromosome 1 matches (0 -> 0), chromosome 2 mutates (1 -> 0):
     probability :math:`p \cdot q`

   That's the only configuration. The chromosomes are **distinguishable**
   (they copy from different sources with different alleles), so there's no
   factor of 2.

Let's verify this with a concrete example. Reference :math:`0/1`, query :math:`0/0`:

- Chromosome 1 (ref=0, query=0): match, probability :math:`p`
- Chromosome 2 (ref=1, query=0): mutation, probability :math:`q`

Total: :math:`pq`. There's no second configuration because the chromosomes are
distinguishable (chromosome 1 copies from :math:`j_1`, chromosome 2 from :math:`j_2`).

.. code-block:: python

   def emission_matrix_diploid(mu, num_sites, num_alleles):
       """Compute emission probability matrix for diploid genotypes.

       Returns matrix of shape (m, 8) indexed by genotype comparison code.

       Indexing scheme (bit-packed, see Step 3 text):
           4 = EQUAL_BOTH_HOM   (ref hom, query hom, same genotype)
           0 = UNEQUAL_BOTH_HOM (ref hom, query hom, different genotype)
           7 = BOTH_HET         (ref het, query het)
           1 = REF_HOM_OBS_HET  (ref hom, query het)
           2 = REF_HET_OBS_HOM  (ref het, query hom)
           3 = MISSING_INDEX    (query is MISSING)
       """
       EQUAL_BOTH_HOM = 4
       UNEQUAL_BOTH_HOM = 0
       BOTH_HET = 7
       REF_HOM_OBS_HET = 1
       REF_HET_OBS_HOM = 2
       MISSING_INDEX = 3

       if isinstance(mu, float):
           mu = np.full(num_sites, mu)

       e = np.full((num_sites, 8), -np.inf)  # -inf flags unused indices

       for i in range(num_sites):
           if num_alleles[i] == 1:
               # Invariant site: no mutation possible
               p_mut = 0.0
               p_no_mut = 1.0
           else:
               # Per-allele mutation probability (symmetric model)
               p_mut = mu[i] / (num_alleles[i] - 1)
               p_no_mut = 1 - mu[i]

           # Case 1: both hom, same genotype -> both match
           e[i, EQUAL_BOTH_HOM]  = p_no_mut ** 2
           # Case 2: both hom, different genotype -> both mutate
           e[i, UNEQUAL_BOTH_HOM] = p_mut ** 2
           # Case 3: both het -> both match OR both mutate
           e[i, BOTH_HET]         = p_no_mut**2 + p_mut**2
           # Case 4: ref hom, query het -> exactly one mutates (factor of 2)
           e[i, REF_HOM_OBS_HET]  = 2 * p_mut * p_no_mut
           # Case 5: ref het, query hom -> exactly one mutates (no factor of 2)
           e[i, REF_HET_OBS_HOM]  = p_mut * p_no_mut
           # Missing data: no information
           e[i, MISSING_INDEX]    = 1.0

       return e

   # Example: verify emission probabilities
   mu = 0.01
   e_dip = emission_matrix_diploid(mu, 1, np.array([2]))
   print("Diploid emission probabilities (mu=0.01):")
   print(f"  Both hom, match:     {e_dip[0, 4]:.6f}  (p^2)")
   print(f"  Both hom, mismatch:  {e_dip[0, 0]:.6f}  (q^2)")
   print(f"  Both het:            {e_dip[0, 7]:.6f}  (p^2 + q^2)")
   print(f"  Ref hom, query het:  {e_dip[0, 1]:.6f}  (2pq)")
   print(f"  Ref het, query hom:  {e_dip[0, 2]:.6f}  (pq)")

The genotype comparison index
--------------------------------

To look up the correct emission, we need a function that compares the reference
genotype to the query genotype and returns the right index. The lshmm library
uses a clever bit-packing scheme:

.. math::

   \text{index} = 4 \cdot \text{is\_match} + 2 \cdot \text{is\_ref\_het} + \text{is\_query\_het}

.. admonition:: Terminology: Bit-Packing

   **Bit-packing** encodes multiple boolean flags into a single integer by
   treating each flag as a binary digit. Here, ``is_match`` contributes
   :math:`4 = 2^2`, ``is_ref_het`` contributes :math:`2 = 2^1`, and
   ``is_query_het`` contributes :math:`1 = 2^0`. The resulting index ranges
   from 0 to 7, mapping all 9 possible genotype comparisons to 5 distinct
   emission categories (with 3 indices unused). This is faster than a chain
   of if-else statements and is a common optimization in HMM implementations.

This maps all 9 possible (ref, query) genotype pairs to 5 distinct categories:

.. code-block:: python

   def genotype_comparison_index(ref_gt, query_gt):
       """Map (ref_genotype, query_genotype) to emission matrix index.

       Genotypes are allele dosages: 0, 1, or 2.
       Uses bit-packing: index = 4*is_match + 2*is_ref_het + is_query_het
       """
       MISSING = -1
       if query_gt == MISSING:
           return 3  # MISSING_INDEX: no information from this site
       # Pack three boolean flags into one integer index
       is_match = int(ref_gt == query_gt)     # bit 2 (weight 4)
       is_ref_het = int(ref_gt == 1)          # bit 1 (weight 2)
       is_query_het = int(query_gt == 1)      # bit 0 (weight 1)
       return 4 * is_match + 2 * is_ref_het + is_query_het

   # Verify: all (ref, query) combinations
   print(f"{'Ref GT':>8} {'Query GT':>10} {'Index':>7} {'Category'}")
   print("-" * 45)
   for ref_gt in [0, 1, 2]:
       for query_gt in [0, 1, 2]:
           idx = genotype_comparison_index(ref_gt, query_gt)
           categories = {
               0: "UNEQUAL_BOTH_HOM",
               1: "REF_HOM_OBS_HET",
               2: "REF_HET_OBS_HOM",
               4: "EQUAL_BOTH_HOM",
               7: "BOTH_HET",
           }
           print(f"{ref_gt:>8} {query_gt:>10} {idx:>7} "
                 f"{categories.get(idx, '???')}")

With the emission probabilities and comparison index defined, we can now build
the diploid forward algorithm.


Step 4: Diploid Forward Algorithm
====================================

The diploid forward algorithm is structurally similar to the haploid version,
but operates on the :math:`n \times n` state space. The forward variable is now
a matrix:

.. math::

   F_{j_1, j_2}(\ell) = P(X_1, \ldots, X_\ell, Z^{(1)}_\ell = j_1, Z^{(2)}_\ell = j_2)

The recursion mirrors the haploid case but accounts for the four transition
types:

.. math::

   F_{j_1, j_2}(\ell) = e_{j_1, j_2}(X_\ell) \cdot \sum_{k_1, k_2} F_{k_1, k_2}(\ell-1) \cdot A_{(k_1,k_2)(j_1,j_2)}

**Naive implementation**: The sum has :math:`n^2` terms, and we compute it for
each of :math:`n^2` states, giving :math:`O(n^4)` per site. For :math:`n = 100`,
that's :math:`10^8` per site -- too slow.

**Exploiting structure**: We can decompose the sum using the same trick as the
haploid case -- the template mechanism's special structure comes to the rescue
again. The transition factors as a product of haploid transitions, so:

.. math::

   \sum_{k_1, k_2} F_{k_1, k_2} A_{(k_1,k_2)(j_1,j_2)}
   = \underbrace{(1-r)^2 F_{j_1, j_2}}_{\text{no switch}}
   + \underbrace{(1-r) \frac{r}{n} \left[\sum_{k_2} F_{j_1, k_2} + \sum_{k_1} F_{k_1, j_2}\right]}_{\text{single switch}}
   + \underbrace{\left(\frac{r}{n}\right)^2 \sum_{k_1, k_2} F_{k_1, k_2}}_{\text{both switch}}

.. admonition:: Probability Aside: The :math:`O(n^2)` Trick for Diploids

   The :math:`O(n^2)` trick is the diploid analog of the haploid
   :math:`O(n)` trick from the :ref:`copying model chapter <copying_model>`.
   The key insight is identical: the sums that appear in the recursion can be
   precomputed and reused.

   The three sums needed are:

   1. **Row sums**: :math:`\sum_{k_2} F_{j_1, k_2}` for each :math:`j_1` --
      sum each row of the :math:`n \times n` forward matrix. Cost: :math:`O(n^2)`.
   2. **Column sums**: :math:`\sum_{k_1} F_{k_1, j_2}` for each :math:`j_2` --
      sum each column. Cost: :math:`O(n^2)`.
   3. **Total sum**: :math:`\sum_{k_1, k_2} F_{k_1, k_2}` -- sum all entries.
      Cost: :math:`O(n^2)`.

   Once these are computed, each of the :math:`n^2` forward variables can be
   updated in :math:`O(1)`, giving :math:`O(n^2)` per site total. This is a
   factor of :math:`n^2` faster than the naive :math:`O(n^4)`.

The three sums can be precomputed:

- :math:`\sum_{k_2} F_{j_1, k_2}` = row sum of :math:`F` for each :math:`j_1`:
  :math:`O(n^2)` total
- :math:`\sum_{k_1} F_{k_1, j_2}` = column sum of :math:`F` for each :math:`j_2`:
  :math:`O(n^2)` total
- :math:`\sum_{k_1, k_2} F_{k_1, k_2}` = total sum: :math:`O(n^2)` total

This reduces the cost to :math:`O(n^2)` per site (from :math:`O(n^4)` naive).

.. code-block:: python

   def forward_diploid(n, m, G, s, emission_matrix, r, norm=True):
       """Forward algorithm for the diploid Li-Stephens model.

       Parameters
       ----------
       n : int
           Number of reference haplotypes.
       m : int
           Number of sites.
       G : ndarray of shape (m, n, n)
           Reference genotype matrix. G[l, j1, j2] = allele dosage
           when copying from (j1, j2).
       s : ndarray of shape (1, m)
           Query genotype (allele dosages: 0, 1, or 2).
       emission_matrix : ndarray of shape (m, 8)
           Diploid emission probabilities.
       r : ndarray of shape (m,)
           Per-site recombination probability.
       norm : bool
           Whether to normalize.

       Returns
       -------
       F : ndarray of shape (m, n, n)
           Forward probabilities (n x n matrix at each site).
       c : ndarray of shape (m,)
           Scaling factors.
       ll : float
           Log-likelihood (base 10).
       """
       F = np.zeros((m, n, n))
       c = np.ones(m)
       r_n = r / n  # Pre-compute r/n for the O(n^2) trick

       # Initialization: uniform over all n^2 pairs, times emission
       for j1 in range(n):
           for j2 in range(n):
               F[0, j1, j2] = 1 / (n**2)  # Uniform prior over pairs
               ref_gt = G[0, j1, j2]       # Reference genotype for this pair
               idx = genotype_comparison_index(ref_gt, s[0, 0])
               F[0, j1, j2] *= emission_matrix[0, idx]  # Emission

       if norm:
           c[0] = np.sum(F[0, :, :])
           F[0, :, :] /= c[0]  # Normalize

           for l in range(1, m):
               # Precompute sums for the O(n^2) trick
               F_no_change = np.zeros((n, n))
               F_j_change = np.zeros(n)

               for j1 in range(n):
                   for j2 in range(n):
                       # No-switch contribution: both stay
                       F_no_change[j1, j2] = (1 - r[l])**2 * F[l-1, j1, j2]
                       # Single-switch contribution: accumulate row/col sums
                       F_j_change[j1] += (1 - r[l]) * r_n[l] * F[l-1, j2, j1]

               # Build forward probabilities using the three magnitudes
               # Start with double-switch (constant across all states)
               F[l, :, :] = r_n[l]**2  # Both switch (normalized: total sum = 1)

               for j1 in range(n):
                   # Add single-switch contributions (one coordinate matches)
                   F[l, j1, :] += F_j_change  # One switches
                   F[l, :, j1] += F_j_change  # The other switches
                   for j2 in range(n):
                       # Add no-switch contribution (both coordinates match)
                       F[l, j1, j2] += F_no_change[j1, j2]

               # Apply emission probabilities
               for j1 in range(n):
                   for j2 in range(n):
                       ref_gt = G[l, j1, j2]
                       idx = genotype_comparison_index(ref_gt, s[0, l])
                       F[l, j1, j2] *= emission_matrix[l, idx]

               # Normalize
               c[l] = np.sum(F[l, :, :])
               F[l, :, :] /= c[l]

           ll = np.sum(np.log10(c))

       else:
           for l in range(1, m):
               F_no_change = np.zeros((n, n))
               F_j1_change = np.zeros(n)
               F_j2_change = np.zeros(n)
               F_both_change = 0.0

               for j1 in range(n):
                   for j2 in range(n):
                       F_no_change[j1, j2] = (1-r[l])**2 * F[l-1, j1, j2]
                       F_j1_change[j1] += (1-r[l]) * r_n[l] * F[l-1, j2, j1]
                       F_j2_change[j1] += (1-r[l]) * r_n[l] * F[l-1, j1, j2]
                       F_both_change += r_n[l]**2 * F[l-1, j1, j2]

               # Build forward probabilities (unnormalized)
               F[l, :, :] = F_both_change  # Both switch
               for j1 in range(n):
                   F[l, j1, :] += F_j2_change     # Chr 2 switches
                   F[l, :, j1] += F_j1_change     # Chr 1 switches
                   for j2 in range(n):
                       F[l, j1, j2] += F_no_change[j1, j2]  # Neither switches

               # Apply emission
               for j1 in range(n):
                   for j2 in range(n):
                       ref_gt = G[l, j1, j2]
                       idx = genotype_comparison_index(ref_gt, s[0, l])
                       F[l, j1, j2] *= emission_matrix[l, idx]

           ll = np.log10(np.sum(F[m-1, :, :]))

       return F, c, ll

With the forward algorithm generalized to the diploid setting, we now extend
the Viterbi algorithm -- finding the most likely gear sequence for two watches
ticking together.


Step 5: Diploid Viterbi Algorithm
====================================

The diploid Viterbi algorithm finds the most likely pair of copying paths. The
:math:`O(n^2)` trick works the same way as for the forward algorithm, but with
:math:`\max` replacing :math:`\sum`.

At each state :math:`(j_1, j_2)`, the best predecessor is one of three types:

1. **No switch**: same state :math:`(j_1, j_2)`, transition weight ``no_switch``
2. **Single switch**: state :math:`(j_1, k_2)` or :math:`(k_1, j_2)` for the
   best :math:`k_2` or :math:`k_1`, transition weight ``single_switch``
3. **Double switch**: the globally best state :math:`(k_1^*, k_2^*)`,
   transition weight ``double_switch``

To evaluate the "single switch" option efficiently, we precompute the row and
column maxima of :math:`V`:

.. math::

   V^{\text{row}}_j = \max_{k} V_{j, k} \quad \text{(best column for each row)}

.. math::

   V^{\text{col}}_j = \max_{k} V_{k, j} \quad \text{(best row for each column)}

These are computed in :math:`O(n^2)` total.

For state :math:`(j_1, j_2)`, the best single-switch predecessor has value
:math:`\max(V^{\text{row}}_{j_1}, V^{\text{col}}_{j_2}) \cdot \texttt{single\_switch}`.

.. admonition:: Probability Aside: Viterbi Decisions in the Diploid Model

   The diploid Viterbi algorithm makes a three-way decision at each state
   (vs. the two-way stay/switch decision in the haploid case):

   1. **No switch**: :math:`V_{j_1,j_2}(\ell-1) \cdot \texttt{no\_switch}`
   2. **Single switch**: :math:`\max(V^{\text{row}}_{j_1}, V^{\text{col}}_{j_2}) \cdot \texttt{single\_switch}`
   3. **Double switch**: :math:`V^*(\ell-1) \cdot \texttt{double\_switch}`

   The winner determines both the Viterbi value *and* the pointer for
   traceback. Just as in the haploid case, a switch requires the alternative
   to be sufficiently better than the current state to overcome the
   recombination penalty. In the diploid case, a double switch requires
   overcoming :math:`(r/n)^2`, which is quadratically smaller than the
   single-switch penalty -- making double switches extremely rare.

.. code-block:: python

   def viterbi_diploid(n, m, G, s, emission_matrix, r):
       """Viterbi algorithm for the diploid Li-Stephens model.

       Returns
       -------
       V : ndarray of shape (n, n)
           Viterbi probabilities at the last site.
       P : ndarray of shape (m, n, n), dtype int
           Pointer array (flattened index into n*n state space).
       ll : float
       """
       V = np.zeros((n, n))
       V_prev = np.zeros((n, n))
       P = np.zeros((m, n, n), dtype=np.int64)
       c = np.ones(m)  # Rescaling factors for numerical stability
       r_n = r / n

       # Initialization: uniform prior over all n^2 pairs, times emission
       for j1 in range(n):
           for j2 in range(n):
               V_prev[j1, j2] = 1 / (n**2)
               ref_gt = G[0, j1, j2]
               idx = genotype_comparison_index(ref_gt, s[0, 0])
               V_prev[j1, j2] *= emission_matrix[0, idx]

       # Forward pass: find best path to each diploid state at each site
       for l in range(1, m):
           # Rescale to prevent underflow
           c[l] = np.amax(V_prev)
           argmax = np.argmax(V_prev)  # Flattened index of global max
           V_prev /= c[l]

           # Precompute row maxima for single-switch decisions.
           # V_rowcol_max[j] = max over all partners k of V_prev[j, k]
           V_rowcol_max = np.amax(V_prev, axis=1)
           arg_rowcol_max = np.argmax(V_prev, axis=1)

           # The three distinct transition magnitudes
           no_switch = (1 - r[l])**2 + 2*(r_n[l]*(1-r[l])) + r_n[l]**2
           single_switch = r_n[l] * (1 - r[l]) + r_n[l]**2
           double_switch = r_n[l]**2

           j1_j2 = 0  # Flattened index for pointer array
           for j1 in range(n):
               for j2 in range(n):
                   # Find best single-switch predecessor:
                   # one coordinate stays, the other switches to the
                   # best available partner
                   V_single = max(V_rowcol_max[j1], V_rowcol_max[j2])
                   P_single = np.argmax(
                       np.array([V_rowcol_max[j1], V_rowcol_max[j2]])
                   )
                   # Reconstruct flattened index of the best single-switch source
                   if P_single == 0:
                       template_single = j1 * n + arg_rowcol_max[j1]
                   else:
                       template_single = arg_rowcol_max[j2] * n + j2

                   # Compare the three options
                   # Option 1: No switch (both stay)
                   V[j1, j2] = V_prev[j1, j2] * no_switch
                   P[l, j1, j2] = j1_j2  # Default: no switch

                   # Option 2: Single switch (one chromosome recombines)
                   single_val = single_switch * V_single
                   if single_val > double_switch:
                       if V[j1, j2] < single_val:
                           V[j1, j2] = single_val
                           P[l, j1, j2] = template_single

                   # Option 3: Double switch (both chromosomes recombine)
                   else:
                       if V[j1, j2] < double_switch:
                           V[j1, j2] = double_switch
                           P[l, j1, j2] = argmax

                   # Emission: multiply by the appropriate genotype emission
                   ref_gt = G[l, j1, j2]
                   idx = genotype_comparison_index(ref_gt, s[0, l])
                   V[j1, j2] *= emission_matrix[l, idx]

                   j1_j2 += 1

           V_prev = np.copy(V)  # Save for next iteration

       # Log-likelihood: sum of log rescaling factors + log of final max
       ll = np.sum(np.log10(c)) + np.log10(np.amax(V))
       return V, P, ll


Diploid traceback and phasing
---------------------------------

The traceback recovers the flattened index at each site, which is then
**unraveled** into the two haplotype indices. This step implicitly performs
**phasing** -- assigning alleles to chromosomes:

.. code-block:: python

   def backwards_viterbi_diploid(m, V_last, P):
       """Traceback for diploid Viterbi.

       Follows pointers backward from the best final state,
       recovering the flattened index at each site.
       """
       path = np.zeros(m, dtype=np.int64)
       # Start at the last site: pick the best diploid state
       path[m - 1] = np.argmax(V_last)

       # Trace backward through the pointer array
       for j in range(m - 2, -1, -1):
           path[j] = P[j + 1].ravel()[path[j + 1]]

       return path

   def get_phased_path(n, flat_path):
       """Convert flattened diploid path to two haploid paths.

       Parameters
       ----------
       n : int
           Number of reference haplotypes.
       flat_path : ndarray of shape (m,)
           Flattened indices into n*n state space.

       Returns
       -------
       path1, path2 : tuple of ndarray of shape (m,)
           Copying paths for each chromosome.
       """
       # np.unravel_index converts flat index back to (row, col) = (chr1, chr2)
       return np.unravel_index(flat_path, (n, n))

With the algorithms in place, we need one more utility: constructing the
reference genotype matrix from the haplotype panel.


Step 6: Building the Reference Genotype Matrix
=================================================

The diploid model needs the reference **genotype** at each site for each pair
of copying sources. Given the reference haplotype panel :math:`H` of shape
:math:`(m, n)`, the reference genotype matrix :math:`G` of shape :math:`(m, n, n)`
is:

.. math::

   G_{\ell, j_1, j_2} = H_{\ell, j_1} + H_{\ell, j_2}

This is just the outer sum of the haplotype alleles at each site.

.. admonition:: Terminology: Outer Sum

   The **outer sum** of two vectors :math:`a` and :math:`b` is the matrix
   :math:`C_{ij} = a_i + b_j`. It is the additive analog of the outer
   product :math:`C_{ij} = a_i \cdot b_j`. In NumPy, ``np.add.outer(a, b)``
   computes this efficiently. For the diploid LS model, the outer sum at
   each site gives the genotype (allele dosage) for every pair of copying
   sources.

.. code-block:: python

   def build_genotype_matrix(H):
       """Build reference genotype matrix from haplotype panel.

       Parameters
       ----------
       H : ndarray of shape (m, n)
           Reference haplotype panel.

       Returns
       -------
       G : ndarray of shape (m, n, n)
           Reference genotype matrix. G[l, j1, j2] = H[l, j1] + H[l, j2].
       """
       m, n = H.shape
       G = np.zeros((m, n, n), dtype=np.int8)
       for l in range(m):
           # Outer sum: genotype for every pair (j1, j2) at site l
           G[l, :, :] = np.add.outer(H[l, :], H[l, :])
       return G

   # Example
   H = np.array([
       [0, 1, 0, 1],
       [1, 0, 1, 0],
   ])
   G = build_genotype_matrix(H)
   print("Reference genotype matrix at site 0:")
   print(G[0])
   print("\nDiagonal (self-pairs):", np.diag(G[0]))

Now let us put everything together in a complete diploid example.


Step 7: Complete Diploid Example
===================================

.. code-block:: python

   # Full diploid pipeline
   np.random.seed(42)
   n = 6   # reference haplotypes
   m = 50  # sites

   # Simulate reference panel (biallelic)
   H = np.random.binomial(1, 0.3, size=(m, n))
   G = build_genotype_matrix(H)  # Reference genotype matrix (m x n x n)

   # Create a diploid query: two haplotypes copied from different sources
   true_path1 = np.zeros(m, dtype=int)
   true_path2 = np.zeros(m, dtype=int)
   true_path1[:25] = 1; true_path1[25:] = 4  # Chr 1: switch at site 25
   true_path2[:] = 2  # Chromosome 2 copies from h_2 the whole time

   # Build query genotype (allele dosage = sum of the two haplotypes)
   h1 = np.array([H[l, true_path1[l]] for l in range(m)])
   h2 = np.array([H[l, true_path2[l]] for l in range(m)])
   query_gt = (h1 + h2).reshape(1, -1)  # Genotype: 0, 1, or 2

   print(f"True copying path, chr 1: h_{true_path1[0]} -> h_{true_path1[25]}")
   print(f"True copying path, chr 2: h_{true_path2[0]} (constant)")
   print(f"Query genotypes: {query_gt[0, :10]}...")

   # Set up model parameters
   mu = 0.01
   e_dip = emission_matrix_diploid(mu, m, np.full(m, 2))
   r = np.full(m, 0.05)
   r[0] = 0.0  # No recombination before the first site

   # Run diploid Viterbi: find the most likely gear sequence for both watches
   V, P, ll = viterbi_diploid(n, m, G, query_gt, e_dip, r)
   flat_path = backwards_viterbi_diploid(m, V, P)
   path1, path2 = get_phased_path(n, flat_path)

   print(f"\nViterbi log-likelihood: {ll:.2f}")
   print(f"Decoded chr 1 path: {path1}")
   print(f"Decoded chr 2 path: {path2}")

   # Note: the phasing might be swapped (chr1 and chr2 are interchangeable)
   # Check both orientations
   acc_direct = (np.mean(path1 == true_path1) + np.mean(path2 == true_path2)) / 2
   acc_swapped = (np.mean(path1 == true_path2) + np.mean(path2 == true_path1)) / 2
   print(f"Accuracy (direct):  {acc_direct:.1%}")
   print(f"Accuracy (swapped): {acc_swapped:.1%}")
   print(f"Best accuracy:      {max(acc_direct, acc_swapped):.1%}")


Step 8: The Phase Ambiguity
==============================

An important subtlety: the diploid model treats the state :math:`(j_1, j_2)`
and :math:`(j_2, j_1)` as **different** states, even though the observed genotype
is the same. This means the model implicitly performs **phasing** -- it infers
which allele is on which chromosome.

However, the genotype :math:`g = h_1 + h_2` is symmetric: swapping the two
chromosomes gives the same observation. So the posterior has a symmetry:

.. math::

   P((j_1, j_2) \mid \text{data}) = P((j_2, j_1) \mid \text{data})

when the data is unphased genotypes. The Viterbi algorithm breaks this
symmetry by choosing one orientation arbitrarily.

.. admonition:: Probability Aside: Phase Symmetry and Identifiability

   The phase symmetry means that the diploid LS model with unphased genotype
   data has a **non-identifiability**: states :math:`(j_1, j_2)` and
   :math:`(j_2, j_1)` are observationally equivalent. The Viterbi algorithm
   picks one arbitrarily, but the forward-backward algorithm reveals the
   symmetry: the posteriors for :math:`(j_1, j_2)` and :math:`(j_2, j_1)`
   will be equal.

   This non-identifiability is not a bug -- it reflects the genuine biological
   ambiguity of unphased genotype data. Resolving it requires additional
   information, such as:

   - **Read-backed phasing**: long sequencing reads that span multiple
     heterozygous sites can link alleles on the same chromosome.
   - **Family data**: parent-offspring trios allow Mendelian transmission to
     resolve phase.
   - **Population-based phasing** (e.g., SHAPEIT, Eagle): iterative
     algorithms that use the LS model on many individuals simultaneously,
     leveraging the fact that most haplotypes in the population are shared.

   In these more advanced settings, the LS model is applied iteratively:
   the current phase estimate is used to update the reference panel, which
   is then used to re-phase the query, and so on until convergence.

In practice, if you care about phasing (resolving which allele is on which
chromosome), you need additional information -- such as read-backed phasing,
family data, or population-based phasing algorithms that use the LS model
iteratively.


Summary
========

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Component
     - Haploid
     - Diploid
   * - State space
     - :math:`n` states
     - :math:`n^2` states
   * - Transition
     - Stay :math:`(1-r) + r/n`
     - No switch / single / double
   * - Emission
     - Match :math:`1-\mu` / mismatch :math:`\mu`
     - 5 genotype categories
   * - Forward complexity
     - :math:`O(mn)`
     - :math:`O(mn^2)`
   * - Viterbi complexity
     - :math:`O(mn)`
     - :math:`O(mn^2)`

The diploid model is more expensive (:math:`n^2` vs :math:`n` states), but the
:math:`O(n^2)` trick (precomputing row/column sums and maxima) keeps the cost
manageable. For :math:`n = 100` and :math:`m = 10^6`, the diploid forward
algorithm requires about :math:`10^{10}` operations -- feasible on modern
hardware.

You've now built a complete Li & Stephens HMM from scratch -- haploid and
diploid, forward-backward and Viterbi. Every gear is exposed, every equation
derived, every algorithm implemented and tested. Like a watchmaker who has
assembled a grand complication from individual components, you own this
mechanism completely -- from the template mechanism of the copying model, through
the gear train of the haploid algorithms, to the paired movements of the
diploid extension.

This versatile gear appears in many movements across computational genetics.
The next time you encounter a tool built on the Li & Stephens model -- whether
it is an imputation engine, a phasing algorithm, an ancestry painter, or an
ARG inference method -- you will recognize the familiar ticking of the
mechanism you have built here, and you will know exactly how it works.
