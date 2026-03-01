.. _tsdate_timepiece:

=====================================
Timepiece IX: tsdate
=====================================

   *Dating Nodes in a Tree Sequence*

The Mechanism at a Glance
==========================

**tsdate** is a Bayesian method for estimating the **age of every ancestral
node** in a tree sequence. Given a genealogy (typically the topology from
tsinfer, which has no meaningful branch lengths), tsdate uses the **molecular
clock** -- the principle that mutations accumulate proportionally to time -- to
infer *when* each ancestor lived.

If tsinfer gives you the skeleton of a watch movement -- the shapes and connections
of the gears -- tsdate adds the calibration marks: real time estimates that tell you
not just *how* the parts connect, but *when* each part was made. Without tsdate, you
know the topology (which ancestors are connected to which descendants); with tsdate,
you know the chronology (how many generations separate them).

The core equation is Bayes' rule (introduced in the :ref:`HMMs prerequisite <hmms>`):

.. math::

   P(\mathbf{t} \mid \mathbf{D}, \mathcal{T}) \propto
   P(\mathbf{D} \mid \mathbf{t}, \mathcal{T}) \cdot P(\mathbf{t} \mid \mathcal{T})

where :math:`\mathbf{t}` is the vector of node ages, :math:`\mathbf{D}` is the
mutational data (how many mutations sit on each edge), and :math:`\mathcal{T}` is
the tree topology. In words: the probability of the node ages given the data is
proportional to the probability of the data given the ages (the **likelihood**) times
the prior probability of those ages (the **prior**).

.. admonition:: Primary Reference

   :cite:`tsdate`

The four gears of tsdate:

1. **The Coalescent Prior** (the escapement) -- A prior on node ages derived from
   coalescent theory: nodes with more descendant samples are expected to be younger,
   because large subtrees coalesce quickly. This is where the theory from
   :ref:`Coalescent Theory <coalescent_theory>` provides the baseline expectation.

2. **The Mutation Likelihood** (the gear train) -- A Poisson model for the number of
   mutations on each edge, connecting observed data to branch lengths. More mutations
   on an edge = longer branch = greater time separation between parent and child.

3. **Belief Propagation** (the mainspring) -- Message-passing algorithms that combine
   prior and likelihood across the interconnected nodes of the tree sequence. Two
   flavors:

   - *Inside-Outside* (discrete time grid) -- exact on a grid
   - *Variational Gamma* (continuous time, default) -- approximate but efficient

4. **Rescaling** (the regulator) -- A post-processing step that adjusts node times so
   the inferred mutation rate matches the empirical rate across time windows. Like
   a watchmaker adjusting the beat rate against a reference frequency.

These gears mesh together into a dating pipeline:

.. code-block:: text

   Tree sequence from tsinfer (topology only, no meaningful times)
                      |
                      v
        +---> Construct coalescent prior for each node
        |              |
        |              v
        |     Compute mutation likelihood for each edge
        |              |
        |              v
        |     Run belief propagation (EP or inside-outside)
        |              |
        |              v
        |     Rescale times to match mutation clock
        |              |
        +--------------+
               (iterate)
                      |
                      v
       Dated tree sequence (node ages = posterior means)

Where tsinfer Ends and tsdate Begins
======================================

The standard pipeline for scalable genealogical inference is:

.. code-block:: text

   Genetic variation data
           |
           v
       tsinfer  -->  Tree topology (no meaningful branch lengths)
           |
           v
       tsdate   -->  Dated tree sequence (calibrated node ages)

**tsinfer** (Timepiece VI) gives you the *skeleton of the movement* -- the frame on
which everything hangs: which samples share which ancestors, and the tree structure
at each position. But the "times" it assigns are just a frequency-based proxy
(older ancestors have higher-frequency derived alleles). These are ordinal, not
quantitative: they tell you ancestor A is older than B, but not by how much.

**tsdate** adds the *calibration* -- real time estimates based on the molecular
clock. Every edge in the tree can carry mutations, and the number of mutations is
informative about the edge's length. By combining this mutation signal with a prior
from coalescent theory, tsdate computes posterior estimates for every node's age.

Together, tsinfer + tsdate form a complete inference pipeline: one provides the
structure, the other provides the timing. Like a watch case and a calibrated dial --
both essential, each useless without the other.

.. admonition:: Prerequisites for this Timepiece

   - :ref:`Coalescent Theory <coalescent_theory>` -- coalescence times and the
     Poisson mutation model
   - :ref:`Ancestral Recombination Graphs <args>` -- tree sequences
   - :ref:`Hidden Markov Models <hmms>` -- for understanding belief propagation
   - Familiarity with :ref:`tsinfer <tsinfer_timepiece>` (Timepiece VI) is helpful
     but not strictly required

Chapters
========

.. toctree::
   :maxdepth: 2

   overview
   coalescent_prior
   mutation_likelihood
   inside_outside
   variational_gamma
   rescaling
   demo

Each chapter derives the math, explains the intuition, implements the code,
and verifies it works. By the end, you'll have built a complete node-dating
engine from scratch -- and you'll understand every gear that drives it.
