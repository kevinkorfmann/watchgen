Ancestor matching and path compression
======================================

Oldest usable ancestors first
-----------------------------

Ancestors are processed in proxy-time groups. Members of one group are matched
against ancestors from strictly older groups; peers are not allowed to become
references for one another merely because of iteration order. Each copying path
becomes tree-sequence edges, and allele differences become mutations.

Paper release 0.1.4 used a highly optimized tree-based matching criterion that
prioritized matches, recombinations, and mismatches. Later releases exposed
probabilistic Li--Stephens-like matching. These are related designs, but it is
incorrect to project every modern probability and default backward onto the 2019
code. This chapter uses 0.1.4 for ancestor-generation parity and labels the 0.4.1
probability equations separately.

From site indexes to coordinates
--------------------------------

Internally, a matched path is represented on inference-site indexes. In the
paper/stable coordinate map, path index zero maps to genomic coordinate zero, an
internal segment beginning at site :math:`j` maps to that site's position, and the
final right endpoint maps to sequence length. Thus a source path ``[A, A, B]`` at
positions ``[10, 20, 40]`` on a sequence of length 100 yields edges
``[0, 40)`` from ``A`` and ``[40, 100)`` from ``B``. Ending the final edge at
``last_position + 1`` is not generally equivalent.

What path compression means
---------------------------

When two children copy the same contiguous *multi-edge path*, tsinfer can insert a
synthetic ancestor for the shared path. Reusing a single equal edge is insufficient:
the source builder searches for contiguous runs containing more than one matching
edge. It then rewires both the existing and new child through the synthetic node and
assigns a time that preserves parent-before-child ordering.

The miniature only detects eligible repeated runs. It does not manufacture a node
or claim that a fixed epsilon below the parent is always valid. Production tsinfer's
indexed builder is the ground truth for topology mutation and node-time placement.

.. literalinclude:: ../../../watchgen/mini_tsinfer.py
   :language: python
   :pyobject: path_to_edges

.. literalinclude:: ../../../watchgen/mini_tsinfer.py
   :language: python
   :pyobject: shared_path_segments
