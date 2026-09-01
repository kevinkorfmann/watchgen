.. _branch_sampling:

===============
Branch sampling
===============

State space
===========

For bin :math:`\ell`, the hidden state :math:`B_\ell` is a joining branch in
the current marginal tree.  If the partial ARG already recombines between two
bins, a branch can split into carried *partial-branch* states.  SINGER retains
all full branches but prunes a partial state when its forward probability is
below :math:`\epsilon` (normally 1%).  This bounds the state space by roughly
:math:`2n-1+1/\epsilon`; empirically it stays close to :math:`2n-1`.

The partial states are not optional decoration.  Without them, a branch-only
Markov path can combine locally valid transitions into a globally impossible
threading that would require a second recombination between adjacent bins.

Joining mass
============

Let :math:`\lambda_\Psi(t)` be the number of extant branches at time
:math:`t` in marginal tree :math:`\Psi`.  In coalescent units with pairwise
rate one, the survival function and joining-time density are

.. math::

   \bar F_\Psi(t)
   = \exp\!\left[-\int_0^t\lambda_\Psi(u)\,du\right],
   \qquad
   f_\Psi(t)=\lambda_\Psi(t)\bar F_\Psi(t).

At time :math:`t`, each of the :math:`\lambda_\Psi(t)` branches is equally
eligible.  The probability of joining one particular branch spanning
:math:`[x,y)` is therefore

.. math::

   p_i=\int_x^y \frac{f_\Psi(t)}{\lambda_\Psi(t)}\,dt
      =\int_x^y\bar F_\Psi(t)\,dt.

The implementation accelerates this calculation by replacing the stochastic
lineage count with

.. math::

   \lambda(t)\approx
   \frac{n}{n+(1-n)e^{-t/2}},
   \qquad
   \bar F(t)\approx
   \frac{e^{-t}}{[n+(1-n)e^{-t/2}]^2}.

A branch is represented by one time :math:`\tau_i`, chosen so that
:math:`\lambda(\tau_i)=\sqrt{\lambda(x)\lambda(y)}`.  This is a heuristic
representative time, not an expectation conditional on joining the branch.

Emissions
=========

Threading a lineage bisects the joining branch into lower and upper pieces.
SINGER imputes the binary state at the joining point by parsimony and
multiplies Poisson probabilities for the three incident edges.  With
:math:`m=\theta l/2`, an edge requiring no state change contributes
:math:`e^{-m}`; an edge requiring one change contributes
:math:`m e^{-m}`.  The latter is the probability of *exactly one* mutation,
not :math:`1-e^{-m}`.

For a root branch, the root state is weighted by ``-polar``: 0 has probability
:math:`p_{root}` and 1 has probability :math:`1-p_{root}`.  The default 0.5
represents unpolarized data; the authors suggest a value such as 0.99 when 0
is reliably ancestral.

Transitions
===========

If the partial ARG has no recombination at the boundary, the probability that
the new lineage recombines before its representative joining time is

.. math::

   r_i=1-e^{-\rho\tau_i/2}.

For a full target branch :math:`j`, define :math:`q_j=r_jp_j`; partial targets
have :math:`q_j=0`.  Then

.. math::

   \Pr(B_\ell=j\mid B_{\ell-1}=i)
   =(1-r_i)\delta_{ij}+r_i\frac{q_j}{\sum_k q_k}.

This drops the SMC rejoining distribution's dependence on the source branch,
which gives the transition its linear Li-Stephens form.  If the partial ARG
already recombines at the boundary, the threaded lineage may only hitchhike
through the corresponding full or partial state: SINGER forbids introducing
a second recombination between the same adjacent bins.

.. literalinclude:: ../../../watchgen/mini_singer.py
   :language: python
   :pyobject: emission_probability
