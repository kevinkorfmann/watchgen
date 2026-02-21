.. _lshmm_timepiece:

====================================
Timepiece II: The Li & Stephens HMM
====================================

   *Your genome is a mosaic. This model finds the seams.*

.. epigraph::

   "Modeling linkage disequilibrium and identifying recombination hotspots using
   single-nucleotide polymorphism data"

   -- Li and Stephens (2003)

The Mechanism at a Glance
==========================

The **Li & Stephens Hidden Markov Model** (Li and Stephens, 2003) is one of the
most influential models in population genetics. It answers a deceptively simple
question: given a panel of reference haplotypes, how was a new haplotype
assembled from pieces of them?

The answer turns out to be a Hidden Markov Model where the hidden states are
"which reference haplotype am I copying right now?" and the transitions are
recombination events that switch the copying source. This simple idea underpins
modern haplotype imputation, phasing, ancestry inference, and -- as we saw in
the prerequisite on :ref:`HMMs <hmms>` -- even full ARG inference.

Think of it this way: every genome is a mosaic, assembled from fragments of
ancestral genomes by generations of recombination. The Li & Stephens model is the
mechanism that detects where the seams are -- where one ancestral fragment ends and
another begins. Like a jeweler's loupe that reveals the individual facets of a
gemstone, this model reveals the hidden structure of how genomes were assembled.

The Li & Stephens HMM is also a **versatile gear** -- a mechanism that appears
inside many of the other Timepieces in this collection. SINGER uses it for branch
transitions. tsinfer uses it for ancestor and sample matching. Understanding this
Timepiece will pay dividends throughout the rest of the book.

.. admonition:: Prerequisites for this Timepiece

   - :ref:`Hidden Markov Models <hmms>` -- the forward algorithm, stochastic
     traceback, and especially the Li-Stephens :math:`O(K)` trick
   - :ref:`Coalescent Theory <coalescent_theory>` -- for understanding why
     haplotypes are mosaics of ancestral sequences

Chapters
========

.. toctree::
   :maxdepth: 2

   overview
   copying_model
   haploid_algorithms
   diploid

Each chapter derives the math, explains the intuition, implements the code,
and verifies it works. By the end, you'll have built a complete LS HMM library
from scratch -- and you'll understand every gear that makes it tick.
