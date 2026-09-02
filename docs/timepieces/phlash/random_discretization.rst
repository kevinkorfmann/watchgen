.. _phlash_random_discretization:

==============================
Random Endpoints, Fixed Shape
==============================

PHLASH samples histories whose time grids differ, but the production model does
not independently jitter every interior breakpoint.  With :math:`M=16`, it puts
priors on two endpoints,

.. math::

   \log t_1 \sim N(\log 10^{-4},1), \qquad
   \log t_M \sim N(\log 15,1),

and geometrically spaces the remaining positive points between them.  In the
source, transformed variables ensure :math:`t_M=t_1+\exp(\delta)>t_1`.
The default tied pattern ``14*1+1*2`` expands 15 inferred rate values across 16
HMM intervals.

What averaging does
===================

Each SVGD particle represents a low-dimensional piecewise-constant history.
Posterior summaries evaluate many such histories on a shared plotting grid and
aggregate them, commonly with pointwise quantiles.  Different sampled endpoints
soften the visual dependence on one fixed grid and let an ensemble approximate a
smoother curve.

That mechanism does **not** prove that discretization errors are independent,
unbiased, or canceled exactly.  The paper reports empirical accuracy and notes
that a fully arbitrary grid was tried but discarded because the particle method
had difficulty converging.  The mini implementation therefore constructs the
actual geometric grid and does not fabricate noisy gradients to demonstrate a
``1/sqrt(K)`` convergence claim.

Coalescent mass on the grid
===========================

For rate :math:`c_i` on :math:`[t_i,t_{i+1})`, survival through a finite
interval is multiplied by
:math:`\exp[-c_i(t_{i+1}-t_i)]`.  Interval masses are successive survival
differences, with the residual survival assigned to the open-ended final
interval.  ``coalescence_probabilities`` returns them in chronological order,
matching ``SizeHistory.p_coal`` in the official source.
