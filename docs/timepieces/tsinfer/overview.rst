.. _tsinfer_overview:

Pipeline and evidence boundary
==============================

What tsinfer infers
-------------------

At nearby sites, a haplotype often resembles long pieces of other haplotypes.
tsinfer uses this copying structure to build a tree sequence. The core pipeline is
deterministic for fixed input, parameters, software version, and implementation
ordering, but that does not make the inferred genealogy uniquely true. It is a
heuristic point estimate produced by an approximate model.

The 2019 method has three inference stages. Ancestor generation turns carrier
patterns at selected variants into partially defined ancestral haplotypes.
Ancestor matching processes these in decreasing proxy age, expressing each as a
mosaic of older ancestors. Sample matching uses the same matching machinery to
thread observed haplotypes through the ancestor tree sequence. Non-inference sites
are then mapped parsimoniously.

Input assumptions
-----------------

The primary-paper pipeline assumed phased haplotypes and ancestral states for the
variants used in inference. In the paper-era defaults, an inference site was a
known-ancestral, biallelic site with at least two derived copies and at least one
ancestral copy. Singletons, fixed sites, unknown ancestral states, and unsuitable
multiallelic sites were excluded from topology inference. Excluded sites were not
discarded from the final data set: their mutations could be positioned on the
inferred trees by parsimony.

Current releases have changed their input storage and pipeline interfaces. The
latest official documentation should therefore be treated as the operational
authority, while tag 0.1.4 is the authority for the historical algorithm described
in the paper.

What the miniature does not do
------------------------------

``watchgen.mini_tsinfer`` checks ancestor grouping and extension, HMM probability
transforms, a dense Viterbi recurrence, path-coordinate conversion, a repeated-path
detector, and small parsimony/flank examples. It deliberately does not claim to
construct a valid production tree sequence. In particular, an upward graph walk is
not tskit's simplify algorithm, and a rectangular reference panel is not the
tree-compressed matcher used by production tsinfer.

This boundary matters methodologically: passing tests against a self-contained toy
pipeline would not establish parity with the software. The tests instead target
equations and fixtures taken from official implementations, and real analyses must
run the released tsinfer/tskit stack.
