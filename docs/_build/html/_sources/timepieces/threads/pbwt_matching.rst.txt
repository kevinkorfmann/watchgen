.. _pbwt_matching:

=========================================
Haplotype Matching with the PBWT
=========================================

   *Reduce the haystack before searching for the needle.*

The first step in the Threads pipeline is to identify, for each sample, a
small set of candidate haplotype matches from the full reference panel. This
is done using the **positional Burrows-Wheeler transform (PBWT)**, a data
structure that sorts haplotypes by their shared prefixes at each site.


Why Pre-Filtering is Necessary
================================

A direct application of the Li-Stephens model to ARG inference requires each
sample :math:`n` to search through :math:`n - 1` previously threaded sequences,
resulting in :math:`O(MN^2)` time for the whole inference. For biobank data
with :math:`N > 10^5` samples and :math:`M > 10^6` sites, this is
computationally infeasible.

Threads reduces the per-sample search space from :math:`n - 1` to :math:`L`
candidates (:math:`L \ll N`) by exploiting the structure of the PBWT: sequences
that are neighbours in the prefix array tend to share long identical-by-state
(IBS) tracts. This heuristic, also employed by modern phasing and imputation
algorithms like IMPUTE5, allows Threads to identify close matches without
exhaustive comparison.


The Chunking Strategy
=======================

Threads partitions the input genomic region into small chunks of default size
0.5 cM. Haplotype matching is performed independently within each chunk. This
chunking serves two purposes:

1. It allows the algorithm to maintain a **locally optimized** set of
   candidates -- matches that are relevant to the local genealogy rather than
   the genome-wide average.

2. It bounds the memory used per sample: within each chunk, the maximum
   number of matches is :math:`L \cdot M_{\text{query}} / M_{\text{chunks}}`.

We denote the number of chunks by :math:`M_{\text{chunk}}`.


PBWT Prefix Array Sorting
============================

At each site :math:`i`, the PBWT maintains a prefix array :math:`a` that sorts
haplotypes by their allelic history up to that site. The update rule is simple:
haplotypes carrying allele 0 at site :math:`i` are placed before those carrying
allele 1, preserving the relative order within each group from the previous
site.

.. code-block:: text

   Site i:   sort₀ = [h3, h1, h4, h2, h5]

   Alleles:  h3→0, h1→1, h4→0, h2→1, h5→0

   Site i+1: sort₁ = [h3, h4, h5,  |  h1, h2]
                       ^^^^^^^^^^^^     ^^^^^^
                       allele = 0       allele = 1
                      (order preserved) (order preserved)

This sorting is performed in :math:`O(N)` time per site using a single pass
through the current array. The PBWT is never stored in full -- genotypes are
streamed through the algorithm one site at a time.


L-Neighbourhood Querying
===========================

At regular intervals (default: every 0.01 cM), the prefix array is queried for
the :math:`L` nearest neighbours around each sequence. For a sequence :math:`n`
at prefix index :math:`i` (i.e., :math:`a_{i,j} = n` at query site :math:`j`),
the query finds the first :math:`L/2` sequences immediately above and below
index :math:`i` in the prefix array that satisfy :math:`a_{k,j} < n` (i.e.,
they were threaded before sample :math:`n`).

If fewer than :math:`L/2` qualifying neighbours exist on one side, the search
window expands in the opposite direction until :math:`L` total matches are found
or all sequences have been considered. The default neighbourhood size is
:math:`L = 4`.

The querying is implemented by sequentially inserting each sequence's prefix
array index into a **red-black tree** (C++ ``std::set``) and querying the
:math:`L`-sized neighbourhood around the insertion point. This gives
:math:`O(\log N)` time per insertion and query.

We denote the total number of query sites by :math:`M_{\text{query}}`.


Candidate Filtering
=====================

Once all sites within a chunk have been processed, each sequence :math:`n` has
a multi-set of candidate matches -- sequences that appeared in the
:math:`L`-neighbourhood at one or more query sites. These candidates are
filtered by match count:

- By default, any match observed in fewer than 4 queries is discarded.
- If no matches survive this filter, the threshold is decreased by 1 until at
  least one sequence remains.
- For :math:`n < 100`, all observed matches are retained (the panel is too
  small for aggressive filtering).
- For :math:`n \geq 10{,}000`, the minimum match count is doubled to
  heuristically limit the number of candidates.

Additionally, the top 4 matches from adjacent chunks are included to account
for recombination events near chunk boundaries.

.. note::

   Singleton variants are excluded from the matching step, as they tend to
   have higher phasing and genotyping errors and may interfere with
   identification of long IBS regions.


Complexity Analysis
=====================

The complete haplotype matching step has the following complexity:

**Time:** :math:`O(MN + M_{\text{query}} \cdot N \cdot \log N)`

- :math:`O(MN)` for the PBWT prefix array updates across all sites
- :math:`O(M_{\text{query}} \cdot N \cdot \log N)` for the red-black tree
  insertions and queries at each query site

**Memory:** :math:`O(N \cdot L \cdot M_{\text{query}})`

- Only the match candidates for each sequence are kept in memory
- The full PBWT and genotype matrix are never stored

The matching requires only a **single pass** through the genotype data,
streaming from disk.
