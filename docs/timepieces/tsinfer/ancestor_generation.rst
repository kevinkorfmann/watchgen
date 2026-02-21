.. _tsinfer_ancestor_generation:

==============================
Gear 1: Ancestor Generation
==============================

   *To know the ancestors, listen to the frequencies. The louder the signal,
   the older the voice.*

Ancestor generation is the first phase of tsinfer -- **extracting the
template gears** from which the rest of the movement will be assembled.
Given the variant matrix :math:`\mathbf{D}`, we construct a set of putative
ancestral haplotypes that will serve as the "reference panel" for later
phases. The key insight is that **allele frequency is a proxy for allele
age**: high-frequency derived alleles are (on average) older than
low-frequency ones.

Before proceeding, make sure you are comfortable with the
:ref:`tsinfer overview <tsinfer_overview>`, particularly the terminology
table and the three-phase pipeline diagram. You should also have a working
understanding of the :ref:`Li & Stephens HMM <lshmm_timepiece>`, since the
ancestors we build here will be fed directly into that model during
matching.

This chapter covers:

1. Which sites qualify for inference
2. How frequency encodes time
3. How ancestors are constructed by consensus voting
4. How ancestors extend left and right from their focal sites


Step 1: Site Selection
=======================

Not every site in the variant matrix is suitable for tree inference. tsinfer
filters sites into two categories: **inference sites** that participate in
tree building, and **non-inference sites** whose mutations are placed later
by parsimony.

A site qualifies as an inference site if and only if:

1. **Biallelic**: exactly two alleles observed (ancestral and derived)
2. **Ancestral allele known**: we know which allele is ancestral
3. **At least 2 derived copies**: :math:`\sum_{i=1}^n D_{ij} \geq 2`
4. **At least 1 ancestral copy**: :math:`\sum_{i=1}^n (1 - D_{ij}) \geq 1`

**Why these criteria?**

- **Biallelic**: Multiallelic sites require more complex handling. By
  restricting to biallelic sites, we get a clean binary signal.

- **Ancestral known**: Without knowing the ancestral state, we can't tell
  which allele is "new" (derived) and which is "old" (ancestral). The
  frequency-as-time mapping requires this polarity.

- **At least 2 derived**: Singletons (exactly one derived copy) don't help
  with tree topology -- they create leaf-specific mutations that don't
  group any samples together. Including them would add ancestors that
  represent a single sample, which is redundant.

- **At least 1 ancestral**: If *all* samples carry the derived allele, the
  site is fixed and provides no information about relationships.

.. admonition:: Confusion Buster -- Derived vs. Ancestral Alleles

   A quick reminder: the **ancestral allele** is the original nucleotide at a
   genomic position -- the one present before any mutation occurred in the
   history of the sample. The **derived allele** is the result of a mutation.
   In the variant matrix :math:`\mathbf{D}`, ancestral is encoded as ``0`` and
   derived as ``1``. The distinction matters because tsinfer uses the *frequency
   of the derived allele* as a clock: common derived alleles are assumed to be
   old, and rare ones are assumed to be young. If we got the polarity wrong
   (called ancestral what is actually derived), the time ordering of ancestors
   would be inverted, and the inferred tree would be distorted.

.. code-block:: python

   import numpy as np

   def select_inference_sites(D, ancestral_known):
       """Select sites suitable for tree inference.

       Parameters
       ----------
       D : ndarray of shape (n, m)
           Variant matrix (0 = ancestral, 1 = derived).
       ancestral_known : ndarray of shape (m,), dtype=bool
           Whether the ancestral allele is known at each site.

       Returns
       -------
       inference_sites : ndarray of int
           Indices of sites that qualify for inference.
       non_inference_sites : ndarray of int
           Indices of sites excluded from inference.
       """
       n, m = D.shape
       is_inference = np.zeros(m, dtype=bool)

       for j in range(m):
           # Skip sites where we don't know which allele is ancestral
           if not ancestral_known[j]:
               continue

           # Count how many samples carry the derived allele (1)
           derived_count = D[:, j].sum()
           # The rest carry the ancestral allele (0)
           ancestral_count = n - derived_count

           # Check all four criteria:
           # - biallelic (exactly 2 distinct values observed)
           # - at least 2 derived copies (no singletons)
           # - at least 1 ancestral copy (not fixed for derived)
           num_alleles = len(np.unique(D[:, j]))
           if (num_alleles == 2 and
               derived_count >= 2 and
               ancestral_count >= 1):
               is_inference[j] = True

       inference_sites = np.where(is_inference)[0]
       non_inference_sites = np.where(~is_inference)[0]
       return inference_sites, non_inference_sites

   # Example
   np.random.seed(42)
   n, m = 20, 15
   D = np.random.binomial(1, 0.3, size=(n, m))
   # Force some edge cases
   D[:, 0] = 1       # Fixed derived -- should be excluded
   D[:, 1] = 0       # Fixed ancestral -- should be excluded
   D[0, 2] = 1; D[1:, 2] = 0  # Singleton -- should be excluded
   ancestral_known = np.ones(m, dtype=bool)
   ancestral_known[3] = False  # Unknown ancestral -- should be excluded

   inf_sites, non_inf_sites = select_inference_sites(D, ancestral_known)
   print(f"Total sites: {m}")
   print(f"Inference sites: {inf_sites}")
   print(f"Non-inference sites: {non_inf_sites}")
   for j in inf_sites:
       freq = D[:, j].sum() / n
       print(f"  Site {j}: derived freq = {freq:.2f}")

With the inference sites identified, we can now assign each one a time. This
is where the frequency-as-age insight comes in.


Step 2: Frequency as a Time Proxy
===================================

The crucial mapping that makes tsinfer work: **derived allele frequency
encodes coalescent time**.

For each inference site :math:`j`, the time proxy is:

.. math::

   t_j = \frac{\text{count of derived alleles at site } j}
              {\text{count of non-missing alleles at site } j}

or equivalently, for complete data:

.. math::

   t_j = \frac{\sum_{i=1}^{n} D_{ij}}{n}

.. admonition:: Confusion Buster -- What is Allele Frequency?

   The **allele frequency** of the derived allele at a site is simply the
   fraction of sampled chromosomes that carry it. If you have :math:`n = 100`
   haplotypes and 73 of them carry the derived allele at a particular site,
   the derived allele frequency is :math:`73/100 = 0.73`. This is a *sample*
   frequency (based on your data), not to be confused with the true population
   frequency (which you don't observe). The **site frequency spectrum (SFS)**
   is the distribution of allele frequencies across all sites in the genome --
   a fundamental summary statistic in population genetics.

**Why does frequency approximate time?**

Under neutral evolution in a constant-size population, the expected frequency
of a derived allele is related to its age. Consider a mutation that arose
:math:`\tau` generations ago in a population of effective size :math:`N_e`.
Its expected frequency under the diffusion approximation is:

.. math::

   \mathbb{E}[f \mid \tau] \approx \frac{\tau}{4N_e}

for :math:`\tau \ll 4N_e`. The intuition is clear: older mutations have had
more time to drift upward in frequency. A mutation at frequency 0.8 has been
around much longer (on average) than one at frequency 0.05.

This is an approximation -- individual allele trajectories are stochastic,
and selection can distort the relationship. But across many sites, it's
remarkably effective.

.. admonition:: Probability Aside -- The Diffusion Approximation

   The relationship :math:`\mathbb{E}[f \mid \tau] \approx \tau / 4N_e`
   comes from the **diffusion approximation** to the Wright-Fisher model.
   In a diploid population of effective size :math:`N_e`, the allele frequency
   :math:`f(t)` evolves as a diffusion process with variance
   :math:`f(1-f)/(2N_e)` per generation. A new mutation starts at frequency
   :math:`1/(2N_e)` and drifts. Conditional on surviving to the present, its
   expected frequency after :math:`\tau` generations is approximately
   :math:`\tau / (4N_e)` when :math:`\tau` is small relative to
   :math:`4N_e`. This is a *mean* relationship -- the variance around it is
   large. A mutation at frequency 0.5 might be 1,000 generations old or
   100,000 generations old. But tsinfer only needs the *ordering* to be
   approximately right, not the exact ages, so this crude proxy works
   surprisingly well in practice. Exact times can be refined later with
   ``tsdate``.

.. admonition:: Why not use a more sophisticated time estimate?

   One could use the full site frequency spectrum or coalescent-based
   estimators. tsinfer deliberately uses the simple frequency proxy because:
   (a) it's fast to compute, (b) it only needs to get the *ordering* of
   ancestors approximately right (not exact times), and (c) exact times
   can be refined later with ``tsdate``.

.. code-block:: python

   def compute_ancestor_times(D, inference_sites):
       """Compute time proxy for each inference site.

       Parameters
       ----------
       D : ndarray of shape (n, m)
           Variant matrix.
       inference_sites : ndarray of int
           Indices of inference sites.

       Returns
       -------
       times : ndarray of float
           Time proxy for each inference site (= derived allele frequency).
       """
       n = D.shape[0]
       times = np.zeros(len(inference_sites))
       for k, j in enumerate(inference_sites):
           # Count non-missing entries (in case of missing data)
           non_missing = np.sum(D[:, j] >= 0)
           # Count derived alleles
           derived = np.sum(D[:, j] == 1)
           # Time proxy = derived allele frequency
           times[k] = derived / non_missing
       return times

   # Example
   times = compute_ancestor_times(D, inf_sites)
   print("Inference sites with time proxies:")
   order = np.argsort(-times)  # Oldest (highest frequency) first
   for idx in order:
       j = inf_sites[idx]
       print(f"  Site {j}: freq = {times[idx]:.2f} (time proxy)")

**Verification**: Times should be in :math:`(0, 1)` for inference sites,
since we excluded fixed sites (frequency 0 or 1) and singletons
(frequency :math:`1/n`):

.. math::

   \frac{2}{n} \leq t_j \leq \frac{n-1}{n} \quad \checkmark

Now that each inference site has a time, we can identify which samples
"belong" to each ancestor. These are the focal samples.


Step 3: The Focal Samples
===========================

For each inference site :math:`j`, the **focal samples** are the samples
that carry the derived allele:

.. math::

   S_j = \{i : D_{ij} = 1\}

These are the samples that "vote" on what the ancestor at time :math:`t_j`
looks like. The idea: if a group of samples all share a derived allele at
site :math:`j`, they likely share a common ancestor near time :math:`t_j`.
The allelic states of those samples at *nearby* sites tell us what that
ancestor's haplotype looked like.

.. admonition:: Probability Aside -- Why Shared Derived Alleles Imply Shared Ancestry

   Under the infinite-sites mutation model, each derived allele arose exactly
   once. Therefore, all samples carrying the derived allele at site :math:`j`
   descend from the single individual in whom the mutation occurred. The
   focal samples :math:`S_j` are precisely the descendants of this mutational
   ancestor. Their consensus haplotype in the genomic neighborhood of site
   :math:`j` is an estimate of what that ancestor's haplotype looked like --
   imperfect (because recombination and further mutations have reshuffled
   things since then), but informative.

.. code-block:: python

   def get_focal_samples(D, site_index):
       """Get the samples carrying the derived allele at a site.

       Parameters
       ----------
       D : ndarray of shape (n, m)
           Variant matrix.
       site_index : int
           The site index.

       Returns
       -------
       focal : ndarray of int
           Indices of samples carrying allele 1 (the derived allele).
       """
       return np.where(D[:, site_index] == 1)[0]

   # Example
   for j in inf_sites[:3]:
       focal = get_focal_samples(D, j)
       print(f"Site {j}: focal samples = {focal}, count = {len(focal)}")

With focal samples in hand, the next step is the heart of ancestor
generation: building each ancestor's haplotype by extending outward from
the focal site.


Step 4: Ancestor Construction by Consensus
============================================

An ancestor is built by extending outward -- left and right -- from a focal
site, using **majority voting** among the focal samples. Think of this as
extracting a template gear from the mechanism: we examine the samples that
share a particular tooth (the derived allele at the focal site) and infer
what the rest of the gear looked like by consensus.

The algorithm at a high level:

1. Start at the focal site :math:`j`. The ancestor carries the derived
   allele (``1``) here by definition.

2. Move one site to the left (site :math:`j-1`). Among the focal samples
   :math:`S_j`, count how many carry allele ``0`` and how many carry
   allele ``1``. The **consensus allele** is whichever has the majority.

3. Continue extending left. At each site, some focal samples may
   **disagree** with the consensus. Track the "agreement count" -- the
   number of focal samples that still match the ancestor.

4. **Stop** when one of these conditions is met:

   - We reach the first site
   - We encounter a site with a *higher* time (older ancestor). This means
     we've hit a boundary where a different, older ancestor takes over.
   - The agreement drops below a threshold

5. Repeat for rightward extension.

Encountering an older site
----------------------------

**Why stop at older sites?** Consider two inference sites: site :math:`j`
with frequency 0.3 (time = 0.3) and site :math:`k > j` with frequency 0.7
(time = 0.7). The ancestor at site :math:`k` is *older* than the one at
site :math:`j`. When extending the ancestor for site :math:`j` rightward,
we should stop at site :math:`k` because the genealogy may change at a site
where a more ancient ancestor is defined.

More precisely, at an older intervening site, the ancestor for :math:`j`
should carry the **ancestral allele** (``0``), because the derived allele
at that older site arose on a branch *above* the ancestor for :math:`j`.

In our watch metaphor, this is like recognizing that one gear was
manufactured before another: the older gear's teeth define the boundary
conditions for the younger gear.

.. code-block:: python

   def build_ancestor(D, inference_sites, times, focal_site_idx):
       """Build an ancestral haplotype by extending from a focal site.

       Parameters
       ----------
       D : ndarray of shape (n, m)
           Variant matrix.
       inference_sites : ndarray of int
           Sorted array of inference site positions.
       times : ndarray of float
           Time proxy for each inference site.
       focal_site_idx : int
           Index into inference_sites (not the site position!).

       Returns
       -------
       ancestor : dict
           'haplotype': allelic states at each inference site in [start, end]
           'start': leftmost inference site index (inclusive)
           'end': rightmost inference site index (exclusive)
           'focal': the focal inference site index
           'time': time proxy
       """
       n_inf = len(inference_sites)
       focal_j = inference_sites[focal_site_idx]
       focal_time = times[focal_site_idx]
       # The focal samples: everyone carrying derived allele at the focal site
       focal_samples = get_focal_samples(D, focal_j)

       # The ancestor's haplotype (over inference sites)
       # -1 = not yet defined; will be filled in as we extend
       haplotype = np.full(n_inf, -1, dtype=int)
       # At the focal site itself, the ancestor carries the derived allele
       haplotype[focal_site_idx] = 1

       # --- Extend leftward ---
       start = focal_site_idx
       for k in range(focal_site_idx - 1, -1, -1):
           site_k = inference_sites[k]

           # Stop if we hit an older site (higher frequency = older)
           if times[k] > focal_time:
               # At this older site, our ancestor carries the ancestral allele
               haplotype[k] = 0
               start = k
               break

           # Consensus vote among focal samples at this site
           alleles = D[focal_samples, site_k]
           ones = np.sum(alleles == 1)
           zeros = np.sum(alleles == 0)

           # Majority wins: if tied, prefer derived (1)
           if ones >= zeros:
               haplotype[k] = 1
           else:
               haplotype[k] = 0

           start = k

       # --- Extend rightward ---
       end = focal_site_idx + 1
       for k in range(focal_site_idx + 1, n_inf):
           site_k = inference_sites[k]

           # Stop if we hit an older site
           if times[k] > focal_time:
               haplotype[k] = 0
               end = k + 1
               break

           # Consensus vote
           alleles = D[focal_samples, site_k]
           ones = np.sum(alleles == 1)
           zeros = np.sum(alleles == 0)

           if ones >= zeros:
               haplotype[k] = 1
           else:
               haplotype[k] = 0

           end = k + 1

       return {
           'haplotype': haplotype[start:end],
           'start': start,
           'end': end,
           'focal': focal_site_idx,
           'time': focal_time,
       }

   # Example: build ancestors for the first few inference sites
   for idx in range(min(3, len(inf_sites))):
       anc = build_ancestor(D, inf_sites, times, idx)
       print(f"Ancestor for site {inf_sites[idx]}:")
       print(f"  Time: {anc['time']:.2f}")
       print(f"  Span: sites {anc['start']} to {anc['end']}")
       print(f"  Haplotype: {anc['haplotype']}")
       print()

With individual ancestors constructed, the next step is to assemble the
complete set and sort them for the matching phases that follow.


Step 5: The Full Ancestor Generation Algorithm
================================================

Now we put it all together. tsinfer generates one ancestor per inference
site, then sorts them by time (oldest first):

.. code-block:: python

   def generate_ancestors(D, ancestral_known):
       """Generate all putative ancestors from variant data.

       Parameters
       ----------
       D : ndarray of shape (n, m)
           Variant matrix.
       ancestral_known : ndarray of shape (m,), dtype=bool
           Whether the ancestral allele is known at each site.

       Returns
       -------
       ancestors : list of dict
           Each ancestor has 'haplotype', 'start', 'end', 'focal', 'time'.
       inference_sites : ndarray of int
           The inference site indices.
       """
       # First, select which sites will participate in inference
       inference_sites, _ = select_inference_sites(D, ancestral_known)
       # Assign a time (= derived allele frequency) to each inference site
       times = compute_ancestor_times(D, inference_sites)

       ancestors = []
       for idx in range(len(inference_sites)):
           # Build one ancestor per inference site
           anc = build_ancestor(D, inference_sites, times, idx)
           ancestors.append(anc)

       # Sort by time (oldest = highest frequency first)
       # This ordering is critical: during matching, older ancestors
       # must be placed before younger ones.
       ancestors.sort(key=lambda a: -a['time'])

       return ancestors, inference_sites

   # Example
   ancestors, inf_sites = generate_ancestors(D, ancestral_known)
   print(f"Generated {len(ancestors)} ancestors")
   print(f"\nAncestors (oldest first):")
   for i, anc in enumerate(ancestors):
       site = inf_sites[anc['focal']]
       print(f"  {i}: site={site}, time={anc['time']:.2f}, "
             f"span=[{anc['start']},{anc['end']}), "
             f"len={len(anc['haplotype'])}")


Step 6: Grouping by Time
==========================

Ancestors at the same frequency (time) form a **time group**. Within a
time group, ancestors are processed in a specific order during matching.
Between groups, the ordering is strict: older groups are processed first.

**Why group?** Ancestors at the same time are contemporaneous -- they
cannot be related by ancestor-descendant relationships. They must all be
matched against strictly older ancestors. This natural partitioning
simplifies the matching phase and ensures we never try to express an
ancestor as a mosaic of its own contemporaries.

This grouping structure will become important in the
:ref:`ancestor matching chapter <tsinfer_ancestor_matching>`, where each
time group is matched as a batch against all previously placed ancestors.

.. code-block:: python

   from collections import defaultdict

   def group_ancestors_by_time(ancestors):
       """Group ancestors by their time proxy.

       Parameters
       ----------
       ancestors : list of dict
           Ancestors sorted by time (oldest first).

       Returns
       -------
       groups : list of (time, list_of_ancestors)
           Groups sorted by time (oldest first).
       """
       groups = defaultdict(list)
       for anc in ancestors:
           # Group by exact frequency value
           groups[anc['time']].append(anc)

       # Sort by time (descending = oldest first)
       sorted_groups = sorted(groups.items(), key=lambda x: -x[0])
       return sorted_groups

   # Example
   groups = group_ancestors_by_time(ancestors)
   print(f"Number of time groups: {len(groups)}")
   for time_val, group in groups:
       print(f"  Time {time_val:.2f}: {len(group)} ancestors")


The Ultimate Ancestor
======================

tsinfer adds one special ancestor: the **ultimate ancestor**, which spans
the entire genome and carries the ancestral allele (``0``) at every
inference site. This ancestor sits at time :math:`t = 1.0` (the oldest
possible) and serves as the root of the ancestor tree.

.. math::

   a_{\text{root}} = (0, 0, 0, \ldots, 0), \quad t_{\text{root}} = 1.0

Every other ancestor descends from this root. Without it, the oldest
"real" ancestors would have no parent to copy from during matching.
In our watch metaphor, the ultimate ancestor is the **mainplate** --
the foundation on which every other component is mounted.

.. code-block:: python

   def add_ultimate_ancestor(ancestors, num_inference_sites):
       """Add the ultimate (root) ancestor.

       Parameters
       ----------
       ancestors : list of dict
           Existing ancestors.
       num_inference_sites : int
           Total number of inference sites.

       Returns
       -------
       ancestors : list of dict
           Updated list with the ultimate ancestor prepended.
       """
       ultimate = {
           # All-zero haplotype: ancestral allele at every site
           'haplotype': np.zeros(num_inference_sites, dtype=int),
           'start': 0,
           'end': num_inference_sites,
           'focal': -1,  # No focal site -- this is a virtual ancestor
           'time': 1.0,  # Oldest possible time
       }
       # Prepend so it appears first (oldest) in the sorted list
       return [ultimate] + ancestors

   # Example
   ancestors_with_root = add_ultimate_ancestor(ancestors, len(inf_sites))
   print(f"Ultimate ancestor: time={ancestors_with_root[0]['time']}, "
         f"haplotype={ancestors_with_root[0]['haplotype'][:5]}...")


Verification
=============

Let's verify the key properties of our ancestor generation:

.. code-block:: python

   def verify_ancestors(ancestors, D, inference_sites):
       """Verify correctness of generated ancestors."""
       n, m = D.shape
       n_inf = len(inference_sites)

       print("Verification checks:")

       # 1. Each ancestor's time is in (0, 1]
       times = [a['time'] for a in ancestors]
       assert all(0 < t <= 1.0 for t in times), "Times out of range!"
       print(f"  [ok] All times in (0, 1]: min={min(times):.3f}, "
             f"max={max(times):.3f}")

       # 2. Ancestors are sorted by time (oldest first)
       for i in range(len(ancestors) - 1):
           assert ancestors[i]['time'] >= ancestors[i+1]['time'], \
               "Ancestors not sorted!"
       print(f"  [ok] Ancestors sorted by time (oldest first)")

       # 3. Each ancestor carries the derived allele at its focal site
       for anc in ancestors:
           if anc['focal'] >= 0:  # Skip ultimate ancestor
               focal_in_haplotype = anc['focal'] - anc['start']
               assert anc['haplotype'][focal_in_haplotype] == 1, \
                   "Focal site should carry derived allele!"
       print(f"  [ok] All ancestors carry derived allele at focal site")

       # 4. Haplotypes contain only 0s and 1s
       for anc in ancestors:
           assert set(anc['haplotype']).issubset({0, 1}), \
               "Invalid allele!"
       print(f"  [ok] All haplotypes contain only 0s and 1s")

       print(f"\nAll checks passed for {len(ancestors)} ancestors.")

   verify_ancestors(ancestors_with_root, D, inf_sites)

With ancestor generation complete, we have extracted all the template gears.
The next chapter introduces the engine that will mesh them together: the
Li & Stephens copying model.


Exercises
==========

.. admonition:: Exercise 1: Frequency vs. age under the coalescent

   Using ``msprime``, simulate 100 independent genealogies for :math:`n = 50`
   samples. For each mutation, record its true age (time of the mutation
   event) and its frequency in the sample. Plot frequency vs. age. How
   well does the linear approximation :math:`\mathbb{E}[f] \propto \tau`
   hold? Where does it break down?

.. admonition:: Exercise 2: Ancestor accuracy

   Simulate a tree sequence with ``msprime`` and then generate ancestors
   using the algorithm above. Compare the inferred ancestor haplotypes to
   the *true* ancestral haplotypes (available from the simulated tree
   sequence). What fraction of alleles are correctly inferred? Does
   accuracy depend on the ancestor's time?

.. admonition:: Exercise 3: Extension with sample dropout

   The current implementation uses *all* focal samples for consensus voting
   at every site. Implement a variant where samples that **disagree** with
   the consensus are dropped from future votes (sample dropout). Compare
   the ancestor haplotypes with and without dropout. Does dropout improve
   accuracy for long ancestors?

Next: :ref:`tsinfer_copying_model` -- the Li & Stephens engine that powers the matching phases.
