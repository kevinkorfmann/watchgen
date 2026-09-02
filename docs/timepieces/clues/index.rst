.. _clues_timepiece:

===================
Timepiece XV: CLUES
===================

   *Selection inference conditional on allele-frequency histories and gene trees*

CLUES approximates the likelihood of a selection coefficient by integrating
over an allele-frequency trajectory and, when genealogy samples are supplied,
over uncertain local trees.  It is an **approximate full likelihood** method,
not an exact likelihood for the original sequence data.

.. important:: Two methods and two parameterizations

   The original CLUES release accompanied Stern, Wilton, and Nielsen (2019)
   :cite:`clues`.  It used ARGweaver samples and assumed positive additive
   selection in the published software.  **CLUES2** added ancient genotypes and
   haplotypes, Relate and SINGER conversion, negative selection, dominance, and
   multiple selection epochs :cite:`clues2`.  This chapter targets CLUES2 source
   commit ``b20dc5d``.  Results from original CLUES and CLUES2 need not agree:
   their documented selection parameterizations differ.

The public CLUES2 parameterization uses genotype fitnesses
``1, 1 + h*s, 1 + s`` and defines ``N`` as the **haploid** effective size.  The
CLI converts that value before calling internal kernels.  This boundary is
essential for avoiding factor-of-two errors.

The accompanying Python code is a teaching model, not a replacement for
``inference.py``.  It covers source-checked transition and emission kernels,
log-space HMM recursion, importance-ratio aggregation, and the likelihood-ratio
statistic.  File conversion, epoch bookkeeping, optimization retries, and
trajectory Monte Carlo remain the original software's responsibility.

.. toctree::
   :maxdepth: 2

   overview
   wright_fisher_hmm
   emission_probabilities
   inference
   demo
