.. _arg_rescaling:

=============
ARG rescaling
=============

Purpose and timing
==================

The threading HMMs assume a constant population size.  SINGER reduces the
resulting time-scale bias by applying a monotone, piecewise-linear
transformation learned from mapped mutations.  This does not infer a
population-size history, change topology, or constitute an MCMC proposal.
The implementation runs it after initialization and after every thinning
interval, immediately before a sample is written.

Equal-ARG-length windows
========================

Let :math:`L(G)` be total ARG branch length, with every marginal-tree branch
weighted by its genomic span.  SINGER chooses :math:`J=100` time windows by
default so that each contains :math:`L(G)/J` branch length.  A mutation mapped
to a branch crossing several windows is divided among them in proportion to
the branch's overlap with each window.

If :math:`m_i` is the fractional mutation count in window :math:`i`, its
expected count under scaled mutation rate :math:`\theta` is
:math:`\theta L(G)/(2J)`.  The window scale factor is therefore

.. math::

   c_i=\frac{2Jm_i}{\theta L(G)}.

For old boundaries :math:`t_i` and new boundaries :math:`\widetilde t_i`,

.. math::

   \widetilde t_0=0,\qquad
   \widetilde t_i=\widetilde t_{i-1}+c_i(t_i-t_{i-1}),

and a node time :math:`t\in[t_{i-1},t_i)` maps to

.. math::

   \widetilde t=\widetilde t_{i-1}+c_i(t-t_{i-1}).

Non-negative factors preserve node order.  A zero-mutation window can collapse
an interval; in real analyses, sparse data are therefore a substantive and
numerical warning, not evidence of instantaneous ancestry.

Mutation-rate maps
==================

With heterogeneous mutation rate :math:`\mu(x)`, expected counts must integrate
the rate across each branch's genomic span and multiply by its overlap with
the time window.  Evaluating :math:`\mu` only at a span midpoint is not the
published calculation and can be biased across sharp rate changes.

.. literalinclude:: ../../../watchgen/mini_singer.py
   :language: python
   :pyobject: compute_scaling_factors
