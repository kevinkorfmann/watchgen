.. _discoal_allele_trajectory:

===============================
Allele-frequency trajectories
===============================

The earlier version of this chapter used an integer birth--death chain and called
it discoal's conditioned jump process. That chain does not occur in the original
software. Production discoal uses a two-point approximation to a conditional
Wright--Fisher diffusion, following Coop and Griffiths :cite:`coop_griffiths_2004`.
The distinction matters because the time step, drift, variance, and direction of
simulation determine the duration of the sweep and therefore every downstream
coalescence and recombination probability.

Scaling and direction
=====================

Let :math:`x` denote the selected-allele frequency and
:math:`\alpha=2N_{\mathrm{sweep}}s`. The program constructs the trajectory
backward, beginning close to the sweep endpoint and moving toward a single-copy
frequency. If :math:`\delta t` is in :math:`2N_{\mathrm{sweep}}` units, the
default grid is

.. math::

   \delta t = \frac{1}{40N_{\mathrm{sweep}}}.

This corresponds to :math:`2/40=0.05` generations per entry when the same
:math:`N_{\mathrm{sweep}}` converts the clock. The command-line option ``-i``
changes the scalar 40, while ``-N`` changes the sweep effective size. These
parameters control numerical resolution; they are not substitutes for the
demographic population-size history.

The stochastic selected phase
=============================

Write :math:`q=1-x`. Production discoal updates :math:`q` using the forward
conditional jump rule and then transforms back to :math:`x`. Equivalently, the
backward selected-frequency step is

.. math::

   x' = x
        - \frac{\alpha x(1-x)}{\tanh[\alpha(1-x)]}\,\delta t
        \;\pm\;\sqrt{x(1-x)\,\delta t},

with equal probability for the two signs. The limiting ratio is evaluated as one
when :math:`\alpha(1-x)` approaches zero. This is a discrete approximation to the
conditional diffusion, not a Wright--Fisher generation and not a chain that changes
the allele count by exactly one.

``mini_discoal.stochastic_trajectory`` implements this law. It exposes the grid
size and endpoint frequencies explicitly and returns a
``mini_discoal.SweepTrajectory`` whose frequencies are ordered backward in time.
Production discoal rejects invalid proposed trajectories in its surrounding event
logic. The mini reflects a local boundary overshoot, so exact random-number-stream
parity is not claimed; the drift, two-point variance, time scaling, and phase
change are the parity targets.

Standing variation
==================

For ``-f f0``, selection acts only above :math:`f_0`. Once the backward path falls
below :math:`f_0`, discoal continues with a neutral conditional process,

.. math::

   x' = x - x\,\delta t \;\pm\;\sqrt{x(1-x)\,\delta t},

until the mutational origin is reached. A soft sweep from standing variation is
therefore not represented by merely starting a logistic curve at :math:`f_0` and
then switching immediately to an unstructured neutral coalescent. The neutral
standing phase is part of the linked genealogy. The
``selection_start_frequency`` argument of
``mini_discoal.stochastic_trajectory`` preserves this distinction.

The deterministic production curve
==================================

The deterministic ``-wd`` mode is also not the textbook logistic curve from
:math:`1/(2N)` to :math:`1-1/(2N)` that appeared in the previous chapter. The C
function ``detSweepFreq`` defines

.. math::

   \epsilon = \frac{0.05}{\alpha}, \qquad
   t_s = -\frac{2\log\epsilon}{\alpha},

and evaluates

.. math::

   x(t) = \frac{\epsilon}
   {\epsilon + (1-\epsilon)\exp[\alpha(t-t_s)]}.

Time :math:`t` again increases backward from the sweep endpoint. The function
``mini_discoal.discoal_deterministic_frequency`` is a direct translation. For
:math:`\alpha=200`, the values at internal times 0, 0.02, and 0.04 are
``0.999750``, ``0.986538``, and ``0.573048``. These fixed values are regression
tests against the source formula.

``mini_discoal.deterministic_trajectory`` evaluates this curve on the production
grid until the requested origin frequency. For teaching examples, a small
``sweep_N`` keeps the array manageable. The production default is one million and
is intentionally much finer.

Fixation probability
====================

The diffusion fixation probability used for interpretation is

.. math::

   h(x) = \frac{1-e^{-\alpha x}}{1-e^{-\alpha}}.

Its neutral limit is :math:`h(x)=x`. The implementation
``mini_discoal.fixation_probability`` uses ``expm1`` for numerical stability. This
formula helps explain conditioning, but discoal does not generate its stochastic
trajectory by inserting :math:`h(k)` into an integer nearest-neighbour chain.
