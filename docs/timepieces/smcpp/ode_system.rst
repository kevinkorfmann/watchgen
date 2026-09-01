.. _smcpp_ode:

=====================
CSFS Matrix Machinery
=====================

   *Compute a labeled frequency spectrum without enumerating genealogies.*

What the production matrices do
===============================

The original SMC++ implementation does not evolve a probability
:math:`p_j(t)` for a number of background lineages and does not derive an
effective hazard :math:`h(t)`. That construction would lose the descendant
labels needed for CSFS emissions and would incorrectly make the distinguished
pair's transition process depend on sample size.

Instead, the production calculation separates expected branch lengths below
and above the conditioned pair TMRCA:

.. math::

   \operatorname{CSFS}(\tau)
   =\operatorname{CSFS}(\tau\downarrow)
   +\operatorname{CSFS}(\tau\uparrow).

Below :math:`\tau`, the two distinguished ancestors are prohibited from
coalescing with each other. Above :math:`\tau`, they have become a single
ancestral block and the genealogy follows an ordinary coalescent. The paper
turns expected times with :math:`k` ancestors into descendant-count categories
using combinatorial matrices.

Below the distinguished TMRCA
=============================

Suppose there are :math:`k` ancestral lineages. In ordinary coalescent time,
the total rate before :math:`\tau` is

.. math::

   \left[\binom{k}{2}-1\right]\alpha(t),

where :math:`\alpha(t)=1/\lambda(t)` is the pairwise coalescence rate. The
subtracted event is the merger of the two blocks containing the distinguished
leaves. The production source integrates expected waiting times under those
rates, then applies matrices that give the probability that a branch subtends
zero or one distinguished leaves and a specified number of undistinguished
leaves.

Only rows :math:`a=0` and :math:`a=1` can receive branch length below
:math:`\tau`: a branch cannot subtend both distinguished leaves before their
ancestors have merged.

Above the distinguished TMRCA
=============================

At :math:`\tau`, the distinguished ancestors merge. The process above that
time begins with one block containing both distinguished leaves plus the
remaining ancestral blocks. SMC++ computes ordinary expected branch lengths
and propagates descendant counts toward the present with a forward-time Moran
model dual to the coalescent. Its exact spectral decomposition is what makes
large-sample evaluation practical.

The relevant generator is tridiagonal because the number of derived copies in
a fixed-size Moran population changes by at most one at an event. This Moran
generator is unrelated to the upper-bidiagonal pure-death matrix previously
shown in this chapter.

The small implementation
========================

For teaching, explicit partition states make both halves visible. The helper
below constructs all reachable partitions, assigns one coalescent-rate unit to
each allowed merger, and records a branch-count reward for each CSFS cell.

.. literalinclude:: ../../../watchgen/mini_smcpp.py
   :language: python
   :pyobject: _conditioned_state_system

For a row-vector distribution :math:`p_0`, generator :math:`Q`, and reward
matrix :math:`B`, the finite-horizon expected reward is

.. math::

   \int_0^t p_0e^{sQ/\lambda}B\,ds.

The block-matrix identity

.. math::

   \exp\!\left[t
   \begin{pmatrix}Q/\lambda&B\\0&0\end{pmatrix}\right]
   =
   \begin{pmatrix}
   e^{tQ/\lambda}&\int_0^t e^{sQ/\lambda}B\,ds\\
   0&I
   \end{pmatrix}

evaluates evolution and occupation in one matrix exponential.

.. literalinclude:: ../../../watchgen/mini_smcpp.py
   :language: python
   :pyobject: _finite_occupation

After the forced merger at :math:`\tau`, the ordinary coalescent is an
absorbing chain. If :math:`Q_T` is its transient generator, the expected state
occupation vector is :math:`p_T(-Q_T)^{-1}`. Multiplication by the reward matrix
gives the above-:math:`\tau` branch lengths.

Why the full sample is informative
==================================

The CSFS changes with :math:`\tau` because conditioning reshapes where branch
length lies among descendant categories. A recent distinguished-pair TMRCA
produces little branch length in categories containing one distinguished
allele; an older TMRCA produces more. Meanwhile, the undistinguished count
:math:`b` reports how mutations are distributed through the rest of the
genealogy. These two pieces let one emission carry both linked pairwise
information and large-sample frequency-spectrum information.

Computational boundary
======================

The set-partition version is exact only within its stated constant-population
model and practical only for small samples. It is a transparent oracle for
unit tests, not a scalable implementation. Original SMC++ uses analytic
integrals, cached combinatorial matrices, extended precision, and automatic
differentiation; those are essential engineering components rather than
cosmetic optimizations.
