Sample matching and post-processing
===================================

Threading observed haplotypes
-----------------------------

After the ancestor tree sequence exists, each phased sample is matched through it
using the same family of copying updates. Sample nodes are at time zero. Copying
segments define edges to ancestral nodes, while discrepancies at inference sites
define mutations. The tree-compressed state space is essential to scalability; the
dense Viterbi function in this book is an equation check, not the production sample
matcher.

Non-inference variants
----------------------

Variants excluded from inference still carry information. For each such site,
tsinfer examines the local inferred tree and places mutations parsimoniously so the
sample genotypes are represented. With a known ancestral state, the root state is
constrained accordingly. Recurrent or back mutations can be necessary; a binary
single-mutation story is not guaranteed for arbitrary data.

Post-processing in stable 0.4.1
-------------------------------

The stable implementation performed several separate operations:

#. identify and detach a virtual-root-like oldest edge when the expected format is
   present, moving affected mutations appropriately;
#. optionally split a genome-wide ultimate ancestor when its children change;
#. erase topology before the first site and after ``last_site + 1`` (capped by
   sequence length); and
#. call tskit's simplifier with sites, populations, and individuals retained and
   unary nodes kept.

These steps are not interchangeable. Merely deleting every edge whose parent is a
chosen root can strand mutations. Likewise, walking upward from samples and dropping
unvisited nodes does not reproduce tskit's coordinate-aware simplify operation.

Current development versions can differ in their post-processing and input model.
Always use production tsinfer with the matching tskit version and preserve
provenance. The teaching helpers below only expose small invariants.

.. literalinclude:: ../../../watchgen/mini_tsinfer.py
   :language: python
   :pyobject: fitch_parsimony

.. literalinclude:: ../../../watchgen/mini_tsinfer.py
   :language: python
   :pyobject: erase_flanks
