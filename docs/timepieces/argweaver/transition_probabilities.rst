.. _argweaver_transitions:

========================
Transition Probabilities
========================

ARGweaver's hidden state at a genomic position is a pair :math:`s=(v,a)`.
The branch :math:`v` says where the threaded lineage joins the partial local
tree, and :math:`a` indexes its coalescence time :math:`t_a`. A transition to
:math:`s'=(w,b)` sums over recombination and re-coalescence events that can turn
the first attachment into the second under the discrete SMC.

The transition is not a source-independent rank-one update. The current
coalescence time changes the length of the augmented tree, the available
recombination material, and several lineage-count corrections. Whether the
source and destination use the same branch also changes which latent
recombination events are compatible. These dependencies appear directly in
both the reference Python formula and ``TransMatrix::get_time`` in the original
C++ source.

No recombination
================

If no recombination occurs between adjacent sites, the attachment state is
unchanged. For source state :math:`s`, the contribution is

.. math::

   \mathbf{1}[s=s']\exp\{-\rho\max(L_s,1)\},

where :math:`L_s` is the length of the local tree after adding the threaded
branch in state :math:`s`. The lower bound of one generation follows the
reference implementation's guard for a per-base transition. Because
:math:`L_s` depends on the source coalescence time, even the diagonal
no-recombination probability is state dependent.

Recombination time
==================

Let :math:`k` denote a discrete recombination interval. The reference
implementation augments the partial-tree lineage counts by the threaded branch.
In the literal testing formula, the probability contribution for interval
:math:`k` is

.. math::

   P(R_k\mid s) =
   \frac{\{n_B(k)+\mathbf{1}[k<a]\}\Delta t_k}
        {\{n_R(k)+\mathbf{1}[k\le a]
           +\mathbf{1}[k=a<w]\}L_s^{\mathrm{basal}}}
   \left(1-e^{-\rho\max(L_s,1)}\right).

Here :math:`n_B(k)` counts branches, :math:`n_R(k)` counts valid discrete
recombination points, :math:`w` is the greater of the source time and root time,
and :math:`L_s^{\mathrm{basal}}` includes the required basal interval. The exact
bookkeeping is important: using only the partial tree's branch material erases
the source-state dependence and produces the wrong kernel.

Re-coalescence
==============

After recombination, the detached lineage survives through successive time
intervals and then coalesces at destination time :math:`b`. For interval
:math:`m`, its hazard is

.. math::

   \lambda_m = \frac{n_B^{*}(m)}{2N_m},

so survival from :math:`k` to :math:`b` contributes

.. math::

   \exp\left\{-\sum_{m=k}^{b-1}
       \frac{n_B^{*}(m)\Delta t_m}{2N_m}\right\}.

At a nonterminal destination interval, the event then contributes

.. math::

   \frac{1-exp\{-n_B^{*}(b)\Delta t_b/(2N_b)\}}
        {n_C(b)},

where :math:`n_C(b)` is the number of valid coalescence points. The terminal
grid interval absorbs the remaining tail probability. The factor :math:`2N`
is required because ARGweaver's population-size parameter is a diploid
effective size.

Summing latent events
=====================

The dense transition probability can be written as

.. math::

   T_{s,s'} =
   \mathbf{1}[s=s']P(\text{no recombination}\mid s)
   + \sum_{r\in\mathcal{R}(s,s')}
     P(r\mid s)P(s'\mid r,s).

The compatible set :math:`\mathcal{R}(s,s')` includes recombinations on the
threaded lineage. When the source and destination use the same branch, it also
includes events on that branch below the two attachment times. This is why a
single destination vector cannot describe every source row.

Compressed transition representation
=====================================

ARGweaver does not materialize and multiply an arbitrary dense matrix at every
site. ``src/argweaver/trans.cpp`` precomputes time-indexed terms named ``D``,
``E``, ``lnB``, ``lnE2``, ``lnNegG1``, ``G2``, ``G3``, ``lnG4``, and
``norecombs``. ``TransMatrix::get_time`` selects different combinations
according to whether :math:`a<b`, :math:`a=b`, or :math:`a>b`, and according to
whether the branch is unchanged.

The forward routine first groups source probability by time, multiplies the
result by an :math:`n_t\times n_t` time matrix, and then adds same-branch
corrections. Thus the optimization exploits time grouping and branch-local
structure. It is not a rank-one identity update, and describing it as simply
:math:`O(S)` hides the explicit time-grid work in the source.

Switch transitions
==================

A recombination breakpoint already present in the partial ARG changes the
local tree itself. ARGweaver therefore uses a separate ``TransMatrixSwitch``
between blocks. Many source states map deterministically across the SPR because
their attachment branch is unaffected. States on the pruned branch or at the
SPR's coalescence point require probabilistic treatment, and states that violate
the new branch-age constraints are unavailable.

This switch matrix is not interchangeable with the within-block transition.
The former conditions on a known change in the partial ARG; the latter sums
over possible recombination events on the lineage currently being threaded.

A bounded teaching approximation
================================

The miniature module retains a deliberately state-independent matrix only as a
probability-bookkeeping exercise. It uses the correct :math:`2N` hazard and
normalizes the represented re-coalescence mass, but it omits branch identity,
the source state's added tree length, same-branch latent events, the absorbing
tail construction, and switch transitions.

.. literalinclude:: ../../../watchgen/mini_argweaver.py
   :language: python
   :pyobject: build_simple_transition_matrix

It must therefore not be presented as a parity implementation of ARGweaver's
transition kernel. Source-level validation belongs to the production formulas
and compressed matrix tests; the toy's appropriate invariants are nonnegative
entries, exact row sums, and increased diagonal mass as :math:`\rho` decreases.

The practical conclusion is that ARGweaver gains efficiency from a structured,
compressed, source-dependent transition kernel. The state pair :math:`(v,a)`
matters on both sides of every transition.
