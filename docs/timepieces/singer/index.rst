.. _singer_timepiece:

======================
Timepiece VII: SINGER
======================

*Sampling and INference of GEnealogies with Recombination*

SINGER is a Bayesian sampler for ancestral recombination graphs (ARGs) under
an SMC approximation.  It initializes an ARG by adding phased haplotypes one
at a time, using one HMM for joining branches and a second HMM for joining
times.  It then explores ARG space with sub-graph pruning and regrafting
(SGPR), and periodically recalibrates node times against mapped mutations
:cite:`singer`.

This chapter follows the published Methods and Supplementary Sections B.1-B.4
and checks the control flow against the authors' C++ implementation.  The
Python module illustrates individual equations and a single-tree SPR; it is
not a replacement for SINGER's ARG data structures or sampler.

.. admonition:: Scope and parity

   The paper benchmarked SINGER v0.1.8.  The source audit for this chapter used
   ``popgenmethods/SINGER`` commit
   ``eb8e39b1a15be4a9a4df4fdaab61847bf73515d7``.  The official program, not
   ``watchgen.mini_singer``, is the ground truth for production behavior.

.. toctree::
   :maxdepth: 2

   overview
   branch_sampling
   time_sampling
   arg_rescaling
   sgpr
   demo
