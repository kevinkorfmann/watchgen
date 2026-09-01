.. _sgpr:
.. _singer_sgpr:

========================================
Sub-graph pruning and regrafting (SGPR)
========================================

The proposal
============

SGPR is an ARG operation, not an independent SPR on every marginal tree.  A
proposal starts at the rightmost coordinate reached by the previous cut (or
wraps to the chromosome start), samples a time uniformly between zero and the
local tree height, and chooses uniformly among branches crossing that time.
The equivalent point is propagated left and right through neighboring trees.
Propagation stops where a recombination decouples the relevant ancestral
material.  In every affected tree, SINGER removes the portion from the cut to
the upper endpoint of its branch.

The remaining graph :math:`H` is regrafted by branch and time sampling above
the cut.  This is the key difference from the Kuhner move: the Kuhner regraft
is simulated from the prior, whereas SGPR's threading proposal uses the data.

Acceptance probability
======================

The exact Metropolis-Hastings rule is

.. math::

   A_H(G\rightarrow G')=
   \min\!\left\{1,
   \frac{P(G'\mid D)q_H(G'\rightarrow G)}
        {P(G\mid D)q_H(G\rightarrow G')}\right\}.

The supplement then assumes that the approximate threading algorithm samples
from the conditional posterior.  Under that assumption, posterior terms
cancel.  For SINGER's cut scheme, the approximation reduces to

.. math::

   A_H(G\rightarrow G')
   \approx \min\!\left\{1,
   \frac{h(\Psi_x)}{h(\Psi'_x)}\right\},

where :math:`x` is the rightmost coordinate of the previous cut and
:math:`h(\Psi_x)` is the marginal-tree height there.  The height ratio is not
a universal SPR identity; it depends on the conditional-posterior proposal
approximation and the specified cut distribution.

The paper reports SGPR acceptance close to one for its 50-sequence benchmark,
compared with about 22% for ARGweaver's subtree-rethreading proposals.  That is
an empirical result for the reported experiment, not a guaranteed acceptance
rate for every dataset.

Single-tree illustration
========================

The miniature's :func:`~watchgen.mini_singer.spr_move` validates and performs
one marginal-tree SPR so the topology change can be inspected.  It does not
propagate cuts, preserve ancestral material across a tree sequence, compute a
proposal density, or run an MCMC chain.  Calling it “SGPR” without this
qualification would overstate parity with the original software.

.. literalinclude:: ../../../watchgen/mini_singer.py
   :language: python
   :pyobject: sgpr_acceptance_ratio
