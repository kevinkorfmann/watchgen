.. _copying_model:

==================
The Copying Model
==================

   *The biggest gear in the mechanism: how haplotypes copy from each other.*

In the :ref:`overview <lshmm_overview>`, we described the Li & Stephens HMM as
a versatile gear that appears in many movements. Now we open the case and
examine the largest gear inside: the **copying model** itself. This is the
template mechanism that defines what it means for one haplotype to be an
"imperfect mosaic" of others.

The Li & Stephens copying model answers two questions at every site along the
genome:

1. **Did a recombination happen?** If so, the copying source switches.
2. **Did a mutation happen?** If so, the query allele differs from the source.

These two questions define the **transition** and **emission** probabilities of
the HMM. We derive both from first principles.

.. note::

   **Prerequisites.** This chapter assumes you have read the
   :ref:`HMM chapter <hmms>`, where transition and emission probabilities
   were introduced in their general form, and the :ref:`overview <lshmm_overview>`,
   which motivates the copying model biologically. We also reference concepts
   from coalescent theory (genealogical trees, recombination); see the
   coalescent theory chapter or Wakeley's *Introduction to Coalescent Theory*
   for background.


Step 1: The Biological Picture
================================

Imagine you have :math:`n` haplotypes that share a common ancestor. When you add
a new haplotype (the query), its ancestry traces back through a genealogical
tree at each genomic position. Due to recombination, the tree changes along the
genome, and the query's closest relative in the reference panel changes too.

Li and Stephens modeled this as follows:

- At each site, the query **copies** its allele from one of the :math:`n`
  reference haplotypes
- Between adjacent sites, there is a small probability :math:`r` that a
  **recombination** occurs, switching the copying source
- At each site, there is a small probability :math:`\mu` that a **mutation**
  occurs, so the query allele differs from the copied allele

The key insight is that this process is **Markov**: the copying source at site
:math:`\ell` depends only on the source at site :math:`\ell - 1` (and the
recombination probability), not on any earlier sites. This makes it an HMM.

.. admonition:: Probability Aside: The Markov Property

   The Markov property -- that the future depends on the past only through the
   present -- is what makes HMMs computationally tractable. For the LS model,
   this means that the copying source at site :math:`\ell` depends only on the
   source at site :math:`\ell - 1`. Biologically, this is an approximation:
   the true genealogical process has long-range correlations (e.g., the tree
   at site :math:`\ell` is correlated with the tree at site :math:`\ell - 10`
   even after conditioning on site :math:`\ell - 1`). The Markov approximation
   works well because nearby sites have highly correlated trees, and the
   errors from ignoring long-range correlations are small in practice.

This biological picture gives us the template mechanism. Now let us formalize
each component, starting with the simplest: the initial distribution.


Step 2: The Initial Distribution
==================================

At the first site, we have no prior information about which reference haplotype
the query is copying. So we assign a **uniform** initial distribution:

.. math::

   \pi_j = P(Z_1 = j) = \frac{1}{n}, \quad j = 1, 2, \ldots, n

This says: before seeing any data, every reference haplotype is equally likely
to be the copying source.

**Why uniform?** Under the coalescent, the query is equally likely to be most
closely related to any of the :math:`n` reference haplotypes (by exchangeability
of lineages). A more sophisticated model could use the coalescent prior, but
the uniform prior works well in practice and keeps things simple.

.. admonition:: Terminology: Exchangeability

   **Exchangeability** means that the joint distribution of a collection of
   random variables is invariant to permutations. In the coalescent,
   exchangeability of lineages means that before observing any data, any
   labeling of the :math:`n` haplotypes is equally likely. This justifies the
   uniform prior :math:`\pi_j = 1/n`. Exchangeability breaks down when there
   is population structure (e.g., the query is known to come from a specific
   subpopulation), but the uniform prior remains the standard choice in the
   LS model.

.. code-block:: python

   import numpy as np

   def initial_distribution(n):
       """Uniform initial distribution over n reference haplotypes.

       Parameters
       ----------
       n : int
           Number of reference haplotypes.

       Returns
       -------
       pi : ndarray of shape (n,)
           Initial state probabilities.
       """
       # Each of the n reference haplotypes is equally likely
       # to be the copying source at the first site.
       return np.ones(n) / n

   # Example
   n = 5
   pi = initial_distribution(n)
   print(f"Initial distribution (n={n}): {pi}")
   print(f"Sum: {pi.sum():.1f}")  # Should be 1.0

With the initial distribution in hand, we move to the most important component
of the model: the transition probabilities.


Step 3: Transition Probabilities
==================================

The transition probability :math:`A_{ij}` answers: given that we were copying
from haplotype :math:`i` at site :math:`\ell - 1`, what is the probability of
copying from haplotype :math:`j` at site :math:`\ell`?

Li and Stephens proposed:

.. math::

   A_{ij} = P(Z_\ell = j \mid Z_{\ell-1} = i) =
   \begin{cases}
   (1 - r) + r/n & \text{if } i = j \\
   r/n & \text{if } i \neq j
   \end{cases}

Or more compactly:

.. math::

   A_{ij} = (1 - r)\delta_{ij} + \frac{r}{n}

where :math:`\delta_{ij}` is the Kronecker delta (1 if :math:`i = j`, 0
otherwise) and :math:`r` is the recombination probability.

.. admonition:: Terminology: Kronecker Delta

   The **Kronecker delta** :math:`\delta_{ij}` is a notational shorthand
   that equals 1 when :math:`i = j` and 0 when :math:`i \neq j`. It appears
   throughout HMM derivations as a compact way to write "same state" vs.
   "different state" in a single formula. In Python, it is simply
   ``int(i == j)``.

Deriving the transition
-------------------------

Let's unpack this formula piece by piece.

**The two things that can happen between adjacent sites:**

1. **No recombination** (probability :math:`1 - r`): The copying source stays
   the same. If we were copying from haplotype :math:`i`, we continue copying
   from haplotype :math:`i`.

2. **Recombination** (probability :math:`r`): The copying source switches to a
   randomly chosen haplotype. Since we have no information about which haplotype
   is chosen, we pick uniformly among all :math:`n` haplotypes (including the
   current one).

Combining these two cases:

.. math::

   A_{ij} &= P(\text{no recomb}) \cdot P(\text{copy } j \mid \text{no recomb})
           + P(\text{recomb}) \cdot P(\text{copy } j \mid \text{recomb}) \\
   &= (1 - r) \cdot \delta_{ij} + r \cdot \frac{1}{n}

When :math:`i = j` (staying):

.. math::

   A_{ii} = (1 - r) + \frac{r}{n}

When :math:`i \neq j` (switching):

.. math::

   A_{ij} = \frac{r}{n}

.. admonition:: Probability Aside: Law of Total Probability

   The derivation above uses the **law of total probability**: the probability
   of an event (copying from :math:`j`) is the sum over all ways it can
   happen, weighted by the probability of each way. Here the two "ways" are
   "no recombination" and "recombination." This is the same decomposition
   used in the general HMM forward recursion (see the :ref:`HMM chapter <hmms>`),
   and it appears repeatedly throughout this book.

**Verification: rows sum to 1.** For any row :math:`i`:

.. math::

   \sum_{j=1}^n A_{ij} &= A_{ii} + \sum_{j \neq i} A_{ij}
   = \left(1 - r + \frac{r}{n}\right) + (n - 1) \cdot \frac{r}{n} \\
   &= 1 - r + \frac{r}{n} + \frac{(n-1)r}{n}
   = 1 - r + \frac{r}{n} \cdot n = 1 - r + r = 1 \quad \checkmark

**Why** :math:`1/n` **and not** :math:`1/(n-1)` **?** After a recombination, the
new source is drawn uniformly from all :math:`n` haplotypes -- including the one
we were already copying. This means a recombination can "switch" us back to the
same haplotype. This is the correct behavior under the coalescent: the detached
lineage can re-coalesce with any lineage, including the one it was previously
closest to.

If we used :math:`1/(n-1)`, we would be forcing a switch away from the current
source, which would overestimate the recombination signal.

.. code-block:: python

   def transition_matrix(n, r):
       """Build the Li-Stephens transition matrix.

       Parameters
       ----------
       n : int
           Number of reference haplotypes.
       r : float
           Recombination probability between adjacent sites.

       Returns
       -------
       A : ndarray of shape (n, n)
           Transition matrix.
       """
       # Start with all entries set to the "switch" probability r/n.
       A = np.full((n, n), r / n)
       # The diagonal gets an extra (1 - r) for the "stay" probability.
       # So diagonal entries are (1 - r) + r/n.
       np.fill_diagonal(A, (1 - r) + r / n)
       return A

   # Example
   n = 4
   r = 0.1
   A = transition_matrix(n, r)
   print(f"Transition matrix (n={n}, r={r}):")
   print(np.round(A, 4))
   print(f"\nRow sums: {A.sum(axis=1)}")  # Should all be 1.0
   print(f"Diagonal: {np.diag(A)}")       # (1-r) + r/n = 0.925
   print(f"Off-diagonal: {A[0, 1]}")      # r/n = 0.025

Having established the transition probabilities, let us now connect them to
the underlying population genetics.


The connection to population genetics
----------------------------------------

Where does :math:`r` come from? In the coalescent, a recombination occurs
between two adjacent sites with probability proportional to the genetic distance
between them. The per-site recombination probability is:

.. math::

   r_\ell = 1 - \exp\left(-\frac{\rho_\ell}{n}\right) \approx \frac{\rho_\ell}{n}

where :math:`\rho_\ell = 4N_e r_{\text{phys}} d_\ell` is the population-scaled
recombination rate between sites :math:`\ell - 1` and :math:`\ell`, with
:math:`d_\ell` being the physical distance in base pairs.

.. admonition:: Terminology: Population-Scaled Parameters

   In population genetics, rates are often expressed in units of the
   effective population size :math:`N_e`:

   - :math:`\rho = 4 N_e r_{\text{phys}}` is the population-scaled
     recombination rate (per base pair per generation, scaled by :math:`4N_e`).
   - :math:`\theta = 4 N_e \mu_{\text{phys}}` is the population-scaled
     mutation rate.

   These scaled parameters absorb the population size, which simplifies the
   math because coalescent theory naturally works in units of :math:`N_e`
   generations. The factor of 4 arises from diploid organisms (2 copies per
   individual, and the coalescent rate is :math:`1/(2N_e)` per pair per
   generation).

**Why divide by** :math:`n` **?** The effective recombination rate in the
Li-Stephens model is scaled by the number of haplotypes. Intuitively: with more
reference haplotypes, each one represents a smaller fraction of the total
genealogical diversity, so recombination between adjacent segments is more likely
to land on a different haplotype. The factor :math:`1/n` normalizes for this.

In practice, :math:`r_\ell` varies along the genome (recombination hotspots and
coldspots). The lshmm library accepts an array of per-site recombination
probabilities.

.. code-block:: python

   def compute_recombination_probs(rho, n):
       """Compute per-site recombination probabilities.

       Parameters
       ----------
       rho : ndarray of shape (m,)
           Population-scaled recombination rate at each site.
       n : int
           Number of reference haplotypes.

       Returns
       -------
       r : ndarray of shape (m,)
           Per-site recombination probability (r[0] = 0 by convention).
       """
       # The exact formula from the coalescent:
       # probability of at least one recombination event in the
       # interval, given rate rho/n.
       r = 1 - np.exp(-rho / n)
       r[0] = 0.0  # No recombination before the first site
       return r

   # Example: uniform recombination rate
   m = 10
   n = 100
   rho = np.full(m, 0.04)  # typical per-site value
   r = compute_recombination_probs(rho, n)
   print(f"Recombination probabilities (first 5 sites): {np.round(r[:5], 6)}")


Site-specific recombination
-----------------------------

In the lshmm implementation, the transition probability at site :math:`\ell` is:

.. math::

   A_{ij}^{(\ell)} = (1 - r_\ell)\delta_{ij} + \frac{r_\ell}{n_\ell}

where :math:`n_\ell` is the number of **copiable** reference entries at site
:math:`\ell`. Why might :math:`n_\ell` differ from :math:`n`?

When the reference panel includes **ancestral haplotypes** (as in tree
sequence-based panels), some entries at some sites are marked as ``NONCOPY``
(:math:`-2`). These represent ancestors that don't exist at every site. The
model correctly excludes them from the switching probability at those sites.

With the transition probabilities fully specified, we turn to the other side of
the template mechanism: the emission probabilities.


Step 4: Emission Probabilities
================================

The emission probability answers: given that we are copying from haplotype
:math:`j` at site :math:`\ell`, how likely is the observed query allele
:math:`s_\ell`?

This is the "imperfection" in the template mechanism. Most of the time, the
stamp produces a faithful copy; occasionally, a mutation introduces a
discrepancy.

The alleles match or don't match
-----------------------------------

The simplest version considers biallelic sites (alleles 0 and 1). If the
reference haplotype :math:`j` carries allele :math:`h_j` at site :math:`\ell`:

.. math::

   e_j(s_\ell) = P(X_\ell = s_\ell \mid Z_\ell = j) =
   \begin{cases}
   1 - \mu & \text{if } s_\ell = h_j \text{ (match: no mutation)} \\
   \mu & \text{if } s_\ell \neq h_j \text{ (mismatch: mutation)}
   \end{cases}

where :math:`\mu` is the per-site mutation probability.

**Intuition**: Most of the time (:math:`1 - \mu \approx 0.999`), the query
carries the same allele as the haplotype it's copying. Occasionally
(:math:`\mu \approx 0.001`), a mutation occurred, and the alleles differ.

.. admonition:: Probability Aside: Emission Probabilities as Likelihoods

   The emission probability :math:`e_j(s_\ell)` can be read two ways:

   1. **Generatively**: if the query copies from haplotype :math:`j`, how
      likely is it to produce allele :math:`s_\ell`?
   2. **As a likelihood**: given the observed allele :math:`s_\ell`, how much
      evidence does it provide for or against copying from haplotype :math:`j`?

   When :math:`s_\ell = h_j` (match), the likelihood ratio is
   :math:`(1-\mu)/\mu \approx 999` for typical :math:`\mu` -- strong evidence
   for copying from :math:`j`. When :math:`s_\ell \neq h_j` (mismatch),
   the likelihood ratio is :math:`\mu/(1-\mu) \approx 0.001` -- strong evidence
   against. This is how the HMM "learns" which haplotype the query is copying
   from: mismatches penalize the wrong source, and matches reward the right one.

.. code-block:: python

   def emission_probability(query_allele, ref_allele, mu):
       """Compute emission probability for one site.

       Parameters
       ----------
       query_allele : int
           Allele in the query haplotype (0 or 1).
       ref_allele : int
           Allele in the reference haplotype (0 or 1).
       mu : float
           Mutation probability.

       Returns
       -------
       prob : float
       """
       if query_allele == ref_allele:
           return 1 - mu  # Match: no mutation needed
       else:
           return mu       # Mismatch: mutation needed

   # Example: show emission probabilities for all allele combinations
   mu = 0.01
   for q, h in [(0, 0), (0, 1), (1, 0), (1, 1)]:
       p = emission_probability(q, h, mu)
       print(f"query={q}, ref={h}: P = {p:.4f}  "
             f"({'match' if q == h else 'mismatch'})")


Handling multiallelic sites
------------------------------

When a site has :math:`a` distinct alleles (e.g., A, C, G, T), a mutation could
change the allele to any of the :math:`a - 1` alternatives. The emission
probabilities become:

.. math::

   e_j(s_\ell) =
   \begin{cases}
   1 - \mu & \text{if } s_\ell = h_j \text{ (match)} \\
   \frac{\mu}{a - 1} & \text{if } s_\ell \neq h_j \text{ (mismatch)}
   \end{cases}

**Why divide by** :math:`a - 1` **?** The total mutation probability is
:math:`\mu`. If there are :math:`a` alleles, a mutation switches to one of the
:math:`a - 1` other alleles. Assuming equal probability for each target allele,
the probability of mutating to any specific different allele is
:math:`\mu / (a - 1)`.

.. admonition:: Probability Aside: The Symmetric Mutation Model

   The assumption that mutations are equally likely to produce any of the
   :math:`a - 1` alternative alleles is called the **symmetric** or
   **Jukes-Cantor** mutation model (for DNA, :math:`a = 4`). More realistic
   models allow different rates for transitions (purine-to-purine or
   pyrimidine-to-pyrimidine) vs. transversions, but the symmetric model keeps
   the LS-HMM simple and is adequate for most applications. The key
   requirement is that the emission probabilities sum to 1 over all possible
   query alleles, which we verify below.

**Verification**: The emission probabilities should sum to 1 over all possible
query alleles:

.. math::

   P(\text{match}) + (a - 1) \cdot P(\text{mismatch per allele})
   = (1 - \mu) + (a - 1) \cdot \frac{\mu}{a - 1} = 1 - \mu + \mu = 1 \quad \checkmark

The lshmm implementation stores these in an **emission matrix** of shape
:math:`(m, 2)` for the haploid case, where column 0 is the mismatch probability
and column 1 is the match probability:

.. code-block:: python

   def emission_matrix_haploid(mu, num_sites, num_alleles):
       """Compute the emission probability matrix for the haploid case.

       Parameters
       ----------
       mu : float or ndarray of shape (m,)
           Per-site mutation probability.
       num_sites : int
           Number of sites.
       num_alleles : ndarray of shape (m,)
           Number of distinct alleles at each site.

       Returns
       -------
       e : ndarray of shape (m, 2)
           Column 0 = mismatch probability, column 1 = match probability.
       """
       if isinstance(mu, float):
           mu = np.full(num_sites, mu)

       e = np.zeros((num_sites, 2))
       for i in range(num_sites):
           if num_alleles[i] == 1:
               # Invariant site: only one allele exists, so no mutation
               # is possible. The query must carry that allele.
               e[i, 0] = 0.0       # mismatch impossible
               e[i, 1] = 1.0       # match certain
           else:
               # Multiallelic site: spread the mutation probability
               # equally across the (a-1) alternative alleles.
               e[i, 0] = mu[i] / (num_alleles[i] - 1)  # per-allele mismatch
               e[i, 1] = 1 - mu[i]                      # match

       return e

   # Example: 6 sites, mix of biallelic and invariant
   num_alleles = np.array([2, 2, 1, 2, 3, 2])
   mu = 0.01
   e = emission_matrix_haploid(mu, 6, num_alleles)
   print("Emission matrix (mismatch | match):")
   for i in range(6):
       print(f"  Site {i} ({num_alleles[i]} alleles): "
             f"mismatch={e[i,0]:.6f}, match={e[i,1]:.6f}")


Scaled mutation rate
-----------------------

There's a subtle but important distinction in how :math:`\mu` is interpreted.
The lshmm library supports two conventions, controlled by the
``scale_mutation_rate`` flag:

**Convention 1** (``scale_mutation_rate=False``, default): :math:`\mu` is the
total probability of mutation to **any** different allele. The per-allele
mismatch probability is :math:`\mu / (a - 1)`.

**Convention 2** (``scale_mutation_rate=True``): :math:`\mu` is the probability
of mutation to **one specific** allele. The total mutation probability is
:math:`(a - 1) \cdot \mu`, and the match probability is
:math:`1 - (a - 1) \cdot \mu`.

These are just two ways of parameterizing the same model. Convention 1 is more
natural for the HMM (the total mutation rate doesn't depend on the number of
alleles), while Convention 2 is more natural for molecular evolution (the
per-allele rate is constant).

Now that we have specified what happens when the query matches and when it does
not, we need to handle two special cases that arise in real data.


Handling NONCOPY and MISSING
-------------------------------

The lshmm library handles two special allele values:

**NONCOPY** (:math:`-2`): A reference entry marked as NONCOPY at a site means
"this haplotype cannot be copied at this site." The emission probability is
:math:`0`, which effectively removes this state from consideration.

**Why NONCOPY?** In tree sequence-based reference panels, ancestral haplotypes
may not span all sites. A haplotype that doesn't exist at a site shouldn't be
a valid copying source there.

**MISSING** (:math:`-1`): A query allele marked as MISSING means "we don't know
what allele the query carries here." The emission probability is :math:`1` for
all states -- the observation provides no information.

**Why MISSING?** Missing data is common in real genotype datasets. By setting the
emission to 1, we effectively skip the site in the HMM computation: the forward
probabilities pass through unchanged (up to normalization).

.. admonition:: Probability Aside: Missing Data and Marginalization

   Setting the emission probability to 1 for missing data is equivalent to
   **marginalizing** over the unknown allele. If we don't know what allele the
   query carries, we sum over all possibilities:

   .. math::

      P(X_\ell = \text{?} \mid Z_\ell = j) = \sum_{a} P(X_\ell = a \mid Z_\ell = j) = 1

   This follows because the emission probabilities sum to 1 over all alleles
   (as verified above). So "missing data" is not a special case -- it is the
   natural consequence of marginalizing over an unobserved variable. This is a
   general technique in probabilistic modeling that appears throughout the
   :ref:`HMM chapter <hmms>`.

.. code-block:: python

   def emission_prob_with_specials(ref_allele, query_allele, site, emission_matrix):
       """Compute emission probability handling NONCOPY and MISSING.

       Parameters
       ----------
       ref_allele : int
           Allele in the reference (-2 = NONCOPY).
       query_allele : int
           Allele in the query (-1 = MISSING).
       site : int
           Site index.
       emission_matrix : ndarray of shape (m, 2)
           Emission probabilities.

       Returns
       -------
       prob : float
       """
       NONCOPY = -2  # Sentinel value: reference haplotype doesn't exist here
       MISSING = -1  # Sentinel value: query allele is unknown

       if ref_allele == NONCOPY:
           return 0.0  # Can't copy from a non-existent haplotype
       elif query_allele == MISSING:
           return 1.0  # Missing data: no information (marginalized)
       else:
           if ref_allele == query_allele:
               return emission_matrix[site, 1]  # Match
           else:
               return emission_matrix[site, 0]  # Mismatch

   # Example: demonstrate NONCOPY and MISSING
   e = emission_matrix_haploid(0.01, 3, np.array([2, 2, 2]))
   print(f"Normal match:    {emission_prob_with_specials(0, 0, 0, e):.4f}")
   print(f"Normal mismatch: {emission_prob_with_specials(0, 1, 0, e):.4f}")
   print(f"NONCOPY ref:     {emission_prob_with_specials(-2, 0, 0, e):.4f}")
   print(f"MISSING query:   {emission_prob_with_specials(0, -1, 0, e):.4f}")

With the transition and emission probabilities fully defined, we now address a
practical question: what value should we use for the mutation probability
:math:`\mu`?


Step 5: Estimating the Mutation Probability
=============================================

If you don't know the mutation probability :math:`\mu`, Li and Stephens (2003)
provide a self-consistent estimator based on the Watterson estimator of
:math:`\theta`.

The Watterson estimator
-------------------------

Given :math:`n` haplotypes, the population-scaled mutation rate :math:`\theta`
can be estimated as:

.. math::

   \hat{\theta} = \frac{S}{\sum_{k=1}^{n-1} 1/k}

where :math:`S` is the number of segregating sites. But we don't need to count
:math:`S` -- we just need the denominator to set the mutation probability.

.. admonition:: Terminology: Segregating Sites and the Harmonic Number

   A **segregating site** is a genomic position where at least two haplotypes
   in the sample carry different alleles. The number of segregating sites
   :math:`S` is a summary statistic of genetic diversity.

   The sum :math:`\sum_{k=1}^{n-1} 1/k` is the :math:`(n-1)`-th
   **harmonic number**, often written :math:`H_{n-1}`. It arises naturally in
   coalescent theory as the expected total branch length of a genealogical
   tree with :math:`n` leaves (in units of :math:`2N_e` generations). The
   Watterson estimator divides :math:`S` by this expected branch length to
   estimate :math:`\theta`.

Li and Stephens (2003, equations A2 and A3) define:

.. math::

   \tilde{\theta} = \frac{1}{\sum_{k=1}^{n-2} 1/k}

Note the upper limit is :math:`n - 2`, not :math:`n - 1`. This is because in
the LS model, we're conditioning on :math:`n - 1` haplotypes and adding a new
one, so the harmonic number uses :math:`n - 2`.

The mutation probability per site is then:

.. math::

   \mu = \frac{\tilde{\theta}}{2(n + \tilde{\theta})} = \frac{1}{2} \cdot \frac{\tilde{\theta}}{n + \tilde{\theta}}

**Derivation**: Under the infinite-sites model with :math:`n` haplotypes, the
expected proportion of sites where the query differs from the closest reference
haplotype is approximately :math:`\tilde{\theta} / (n + \tilde{\theta})`. The
factor of :math:`1/2` accounts for the fact that :math:`\mu` is the per-branch
mutation probability (mutation can happen on either the query branch or the
reference branch).

.. admonition:: Probability Aside: Why the Factor of 1/2?

   The mutation probability :math:`\mu` in the LS model represents the chance
   of a mismatch between the query and its closest reference at a single site.
   A mismatch can arise from a mutation on *either* the query branch or the
   reference branch of the genealogical tree. If the total expected number of
   mutations between the query and its closest reference is proportional to
   :math:`\tilde{\theta}/(n + \tilde{\theta})`, then attributing mutations
   equally to the two branches gives each a rate of half that. This is a
   simplification -- in reality, the two branches may have different lengths --
   but it works well as a default estimate.

**Intuition**: More reference haplotypes (:math:`n \uparrow`) means the query is
more likely to find a close match, so :math:`\mu` decreases. Higher mutation rate
(:math:`\tilde{\theta} \uparrow`) means more mismatches are expected, so
:math:`\mu` increases.

.. code-block:: python

   def estimate_mutation_probability(n):
       """Estimate mutation probability from the number of haplotypes.

       Based on Li & Stephens (2003), equations A2 and A3.

       Parameters
       ----------
       n : int
           Number of reference haplotypes (must be >= 3).

       Returns
       -------
       mu : float
           Estimated per-site mutation probability.
       """
       if n < 3:
           raise ValueError("Need at least 3 haplotypes.")

       # Watterson-style estimator with n-2 upper limit.
       # The harmonic sum runs from 1 to n-2 because we condition on
       # n-1 existing haplotypes and add one more.
       theta_tilde = 1.0 / sum(1.0 / k for k in range(1, n - 1))

       # Per-branch mutation probability: half the expected mismatch rate.
       mu = 0.5 * theta_tilde / (n + theta_tilde)
       return mu

   # How does mu change with panel size?
   print(f"{'n':>5} {'mu':>12} {'1/mu':>12}")
   print("-" * 32)
   for n in [5, 10, 50, 100, 500, 1000, 5000]:
       mu = estimate_mutation_probability(n)
       print(f"{n:>5} {mu:>12.6f} {1/mu:>12.1f}")

The output shows that :math:`\mu` decreases as :math:`n` grows -- with more
reference haplotypes, the model expects fewer mismatches because there's always a
close relative in the panel.

This completes the specification of the model. But before we move to the
algorithms, there is one crucial computational insight that makes the whole
machinery practical.


Step 6: The O(K) Trick -- Making It Fast
==========================================

This is the most important computational insight of the Li-Stephens model, and
it's worth understanding deeply because it appears in every algorithm that uses
this model (SINGER included). Think of it as the equivalent of a jeweled
bearing in watchmaking -- a small design choice that dramatically reduces
friction and makes the entire mechanism viable.

The naive forward step
-------------------------

Recall the HMM forward recursion from the :ref:`HMM chapter <hmms>`:

.. math::

   \alpha_j(\ell) = e_j(X_\ell) \sum_{i=1}^n \alpha_i(\ell-1) \cdot A_{ij}

For a general transition matrix, this sum costs :math:`O(n)` per state, giving
:math:`O(n^2)` per site and :math:`O(mn^2)` total. With :math:`n = 1000`
reference haplotypes and :math:`m = 10^6` sites, that's :math:`10^{12}`
operations -- far too slow.

Exploiting the structure
--------------------------

The Li-Stephens transition matrix has a very special structure:

.. math::

   A_{ij} = (1 - r)\delta_{ij} + \frac{r}{n}

Substituting into the forward recursion:

.. math::

   \alpha_j(\ell) &= e_j(X_\ell) \sum_{i=1}^n \alpha_i(\ell-1) \left[(1 - r)\delta_{ij} + \frac{r}{n}\right] \\
   &= e_j(X_\ell) \left[(1 - r)\alpha_j(\ell-1) + \frac{r}{n} \sum_{i=1}^n \alpha_i(\ell-1)\right]

In the first term, the Kronecker delta :math:`\delta_{ij}` kills all terms
except :math:`i = j`, giving :math:`\alpha_j(\ell-1)`.

In the second term, the factor :math:`r/n` doesn't depend on :math:`i`, so it
factors out. The remaining sum :math:`\sum_i \alpha_i(\ell-1)` is just the total
forward probability at the previous site.

**The key observation**: the sum :math:`S = \sum_i \alpha_i(\ell-1)` is computed
**once** in :math:`O(n)` time and reused for all :math:`n` states. Each
individual :math:`\alpha_j` then costs :math:`O(1)`. Total: :math:`O(n)` per
site instead of :math:`O(n^2)`.

.. admonition:: Probability Aside: Why This Structure is Special

   Not all HMMs have this :math:`O(n)` trick. The LS transition matrix is a
   **rank-1 update** of the identity matrix:
   :math:`A = (1-r)I + (r/n)\mathbf{1}\mathbf{1}^T`. The rank-1 part
   :math:`(r/n)\mathbf{1}\mathbf{1}^T` has all entries equal, so its
   contribution to the sum factorizes. Any HMM whose transition matrix can be
   decomposed as "diagonal + low-rank" enjoys a similar speedup. This structure
   arises naturally in the LS model because recombination is "memoryless" --
   after a recombination event, the new source is drawn uniformly, regardless
   of the previous source.

.. code-block:: python

   def forward_step_naive(alpha_prev, A, emission, n):
       """Naive O(n^2) forward step -- for comparison only.

       For each state j, we sum over all n previous states i,
       multiplying by the transition probability A[i, j].
       """
       alpha = np.zeros(n)
       for j in range(n):
           # Inner loop over all n previous states: this is the O(n^2) part
           alpha[j] = emission[j] * np.sum(alpha_prev * A[:, j])
       return alpha

   def forward_step_fast(alpha_prev, r, r_n, emission, n):
       """O(n) forward step using Li-Stephens structure.

       Parameters
       ----------
       alpha_prev : ndarray of shape (n,)
           Forward probabilities at previous site.
       r : float
           Recombination probability at this site.
       r_n : float
           r / n (or r / n_copiable for NONCOPY support).
       emission : ndarray of shape (n,)
           Emission probabilities at this site.

       Returns
       -------
       alpha : ndarray of shape (n,)
       """
       alpha = np.zeros(n)
       for j in range(n):
           # Stay term: probability of staying in state j (no recombination)
           # plus the probability of recombining back to j.
           # Switch term: probability of recombining to j from any other state,
           # which uses the precomputed sum S (implicit when normalized).
           alpha[j] = alpha_prev[j] * (1 - r) + r_n
           alpha[j] *= emission[j]  # Multiply by emission probability
       return alpha

   # Verify they give the same answer
   n = 5
   r = 0.1
   A = transition_matrix(n, r)
   alpha_prev = np.random.dirichlet(np.ones(n))  # Random normalized probs
   emission = np.random.uniform(0.5, 1.0, n)

   alpha_naive = forward_step_naive(alpha_prev, A, emission, n)
   alpha_fast = forward_step_fast(alpha_prev, r, r / n, emission, n)

   print(f"Naive:  {np.round(alpha_naive, 8)}")
   print(f"Fast:   {np.round(alpha_fast, 8)}")
   print(f"Match:  {np.allclose(alpha_naive, alpha_fast)}")

Wait -- the fast version doesn't look right. Let's be more careful.

In the fast version, the switch term should be :math:`r_n \cdot S` where
:math:`S = \sum_i \alpha_i(\ell-1)`. But when we use **scaled** forward
probabilities (which sum to 1 after normalization), :math:`S = 1`, so the switch
term simplifies to just :math:`r_n`. Let's verify this.

With scaling (normalized forward probabilities)
---------------------------------------------------

In the lshmm implementation, forward probabilities are normalized at each step.
After normalization, :math:`\sum_j \hat{\alpha}_j(\ell-1) = 1`. Therefore:

.. math::

   \hat{\alpha}_j(\ell) \propto e_j(X_\ell) \left[(1 - r)\hat{\alpha}_j(\ell-1) + \frac{r}{n} \cdot \underbrace{\sum_i \hat{\alpha}_i(\ell-1)}_{= 1}\right]

.. math::

   = e_j(X_\ell) \left[(1 - r)\hat{\alpha}_j(\ell-1) + \frac{r}{n}\right]

This is exactly what the lshmm code computes:

.. code-block:: python

   # From fb_haploid.py, the core forward step (with normalization):
   # F[l, i] = F[l-1, i] * (1 - r[l]) + r_n[l]   # transition
   # F[l, i] *= emission_prob                       # emission

That's it. One multiplication, one addition, and one emission multiplication
per state. :math:`O(n)` per site, :math:`O(mn)` total. For :math:`n = 1000`
and :math:`m = 10^6`, that's :math:`10^9` operations -- a thousand times faster
than naive.

Without scaling (unnormalized forward probabilities)
-------------------------------------------------------

If you don't normalize (e.g., for debugging or small problems), you must
explicitly compute the sum:

.. code-block:: python

   # From fb_haploid.py, the unnormalized version:
   # S = np.sum(F[l-1, :])                          # total forward prob (O(n))
   # F[l, i] = F[l-1, i] * (1 - r[l]) + S * r_n[l] # transition
   # F[l, i] *= emission_prob                        # emission

Here ``np.sum(F[l-1, :])`` computes :math:`S` once per site, and the total cost
is still :math:`O(n)` per site.

.. code-block:: python

   def forward_ls_haploid(H, s, mu, r, normalize=True):
       """Complete forward algorithm for the haploid Li-Stephens model.

       Parameters
       ----------
       H : ndarray of shape (m, n)
           Reference panel (m sites, n haplotypes).
       s : ndarray of shape (m,)
           Query haplotype.
       mu : float
           Mutation probability.
       r : ndarray of shape (m,)
           Per-site recombination probability (r[0] should be 0).

       Returns
       -------
       F : ndarray of shape (m, n)
           Forward probabilities.
       c : ndarray of shape (m,)
           Scaling factors (c[l] = sum of unscaled F[l, :]).
       ll : float
           Log-likelihood (base 10).
       """
       m, n = H.shape
       F = np.zeros((m, n))
       c = np.zeros(m) if normalize else np.ones(m)
       r_n = r / n  # Pre-compute r/n for each site

       # Initialization: uniform prior times emission at site 0
       for j in range(n):
           if s[0] == H[0, j]:
               F[0, j] = (1 / n) * (1 - mu)  # Match: pi * (1 - mu)
           else:
               F[0, j] = (1 / n) * mu          # Mismatch: pi * mu

       if normalize:
           c[0] = F[0, :].sum()       # Scaling factor = total probability
           F[0, :] /= c[0]            # Normalize so probabilities sum to 1

       # Forward pass: iterate over sites 1, ..., m-1
       for l in range(1, m):
           if normalize:
               # Scaled: sum of previous F is 1, so switch term = r_n[l]
               for j in range(n):
                   F[l, j] = F[l - 1, j] * (1 - r[l]) + r_n[l]
                   # Emission
                   if s[l] == H[l, j]:
                       F[l, j] *= (1 - mu)  # Match
                   else:
                       F[l, j] *= mu        # Mismatch
               c[l] = F[l, :].sum()
               F[l, :] /= c[l]
           else:
               # Unscaled: must compute sum explicitly
               S = F[l - 1, :].sum()  # Total forward prob at previous site
               for j in range(n):
                   F[l, j] = F[l - 1, j] * (1 - r[l]) + S * r_n[l]
                   if s[l] == H[l, j]:
                       F[l, j] *= (1 - mu)
                   else:
                       F[l, j] *= mu

       # Recover log-likelihood from scaling factors (or final sum)
       if normalize:
           ll = np.sum(np.log10(c))
       else:
           ll = np.log10(F[m - 1, :].sum())

       return F, c, ll

   # Test: small example
   H = np.array([
       [0, 0, 1, 0],  # site 0
       [0, 1, 0, 0],  # site 1
       [1, 0, 0, 1],  # site 2
       [0, 0, 1, 0],  # site 3
       [0, 1, 1, 0],  # site 4
   ])
   s = np.array([0, 0, 1, 0, 1])
   mu = 0.01
   r = np.array([0.0, 0.05, 0.05, 0.05, 0.05])

   F, c, ll = forward_ls_haploid(H, s, mu, r, normalize=True)
   print(f"Log-likelihood: {ll:.4f}")
   print(f"\nForward probs at last site (posterior):")
   for j in range(4):
       print(f"  h_{j}: {F[-1, j]:.4f}")

This shows the posterior probability of copying from each reference haplotype at
the last site. The haplotype with the highest probability is the one whose allele
pattern best matches the query in that region.


Step 7: Putting It All Together
=================================

Now that every component of the template mechanism is in place -- initial
distribution, transition probabilities, emission probabilities, mutation
estimation, and the :math:`O(n)` trick -- let's see the complete model in
action on a larger example.

.. code-block:: python

   # Simulate a reference panel and a mosaic query
   np.random.seed(42)
   n = 10   # reference haplotypes
   m = 100  # sites

   # Random reference panel (biallelic)
   H = np.random.binomial(1, 0.3, size=(m, n))

   # Create a mosaic query: copy from different haplotypes in blocks
   true_path = np.zeros(m, dtype=int)
   true_path[0:30] = 2    # Copy from haplotype 2 for sites 0-29
   true_path[30:70] = 5   # Copy from haplotype 5 for sites 30-69
   true_path[70:100] = 8  # Copy from haplotype 8 for sites 70-99

   # Start with perfect copy, then add mutations
   s = np.array([H[l, true_path[l]] for l in range(m)])
   # Add 3 mutations at random positions
   mutation_sites = np.random.choice(m, 3, replace=False)
   s[mutation_sites] = 1 - s[mutation_sites]  # Flip the allele

   print(f"True path: copies from h_{true_path[0]} (sites 0-29), "
         f"h_{true_path[30]} (30-69), h_{true_path[70]} (70-99)")
   print(f"Mutations at sites: {sorted(mutation_sites)}")

   # Run forward algorithm
   mu = estimate_mutation_probability(n)
   r = np.full(m, 0.05)
   r[0] = 0.0

   F, c, ll = forward_ls_haploid(H, s, mu, r, normalize=True)

   # Find the most likely state at each site (crude decoding)
   decoded_path = np.argmax(F, axis=1)

   # Compare to truth
   accuracy = np.mean(decoded_path == true_path)
   print(f"\nDecoded accuracy (argmax of forward): {accuracy:.1%}")
   print(f"Log-likelihood: {ll:.2f}")

   # Show the decoded path vs true path around breakpoints
   print(f"\nAround breakpoint at site 30:")
   for l in range(27, 33):
       print(f"  Site {l}: true=h_{true_path[l]}, "
             f"decoded=h_{decoded_path[l]}, "
             f"P(true)={F[l, true_path[l]]:.3f}")

Summary
========

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Component
     - Formula
   * - Initial distribution
     - :math:`\pi_j = 1/n`
   * - Transition (same)
     - :math:`A_{ii} = (1 - r) + r/n`
   * - Transition (switch)
     - :math:`A_{ij} = r/n \quad (i \neq j)`
   * - Emission (match)
     - :math:`e_j(s_\ell) = 1 - \mu`
   * - Emission (mismatch)
     - :math:`e_j(s_\ell) = \mu / (a-1)`
   * - Mutation estimate
     - :math:`\mu = \tilde{\theta} / [2(n + \tilde{\theta})]`
   * - O(K) trick
     - Factor out :math:`\sum_i \alpha_i` for :math:`O(n)` per site

These are the fundamental gears of the Li & Stephens template mechanism. We
have built each gear from first principles, verified it works, and shown how
the special structure of the transition matrix -- a rank-1 update of the
identity -- yields the :math:`O(n)` trick that makes the model practical.

Next, we'll assemble these gears into the complete algorithms: forward-backward
and Viterbi -- the gear train that turns the model into answers.

Next: :ref:`haploid_algorithms` -- The complete algorithms for haploid data.
