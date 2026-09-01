.. _time_sampling:

=============
Time sampling
=============

Conditional HMM
===============

Time sampling conditions on the full joining-branch path returned by branch
sampling.  For a branch :math:`[x_\ell,y_\ell)`, SINGER partitions the interval
into 5% quantiles of an exponential distribution (20 states by default).  If
the boundaries are :math:`t_{\ell,0},\ldots,t_{\ell,d}`, state :math:`i` uses
a representative time satisfying

.. math::

   e^{-\tau_{\ell,i}}
   =\frac{e^{-t_{\ell,i}}+e^{-t_{\ell,i+1}}}{2}.

The PSMC-like kernel
====================

Let :math:`s` be the previous joining time and :math:`\rho` the scaled
recombination rate for one bin.  Conditioned on a recombination, the new time
has density

.. math::

   q_0(t\mid s)=
   \begin{cases}
      [1-e^{-t}]/s, & t<s,\\
      [e^{-(t-s)}-e^{-t}]/s, & t\ge s.
   \end{cases}

Without conditioning, multiply the continuous part by
:math:`1-e^{-\rho s}` and add an atom of mass :math:`e^{-\rho s}` at
:math:`t=s`.  The atom is a probability mass, not a density spike that should
be integrated numerically.

Writing :math:`Q_\rho(t\mid s)` for the resulting CDF, the interval transition
is

.. math::

   q^{\ell-1,\ell}_{ij}=
   \frac{Q_\rho(t_{\ell,j+1}\mid\tau_{\ell-1,i})
        -Q_\rho(t_{\ell,j}\mid\tau_{\ell-1,i})}
        {Q_\rho(y_\ell\mid\tau_{\ell-1,i})
        -Q_\rho(x_\ell\mid\tau_{\ell-1,i})}.

The denominator conditions the new coalescence time to lie on the sampled
joining branch.  Each row therefore sums to one.

Three boundary cases
====================

**Type A**
   Neither the partial ARG nor joining branch changes.  Symmetries in the
   transition matrix permit the forward update in :math:`O(d)` rather than
   :math:`O(d^2)`.  For :math:`i>j`, :math:`q_{ij}=q_{j+1,j}`; for :math:`i<j`,
   adjacent columns have a source-independent ratio.

**Type B**
   The partial ARG recombines.  Joining-time intervals that remain on the
   sampled branch hitchhike to the next tree; incompatible intervals vanish,
   and SINGER creates additional intervals to cover any remainder.

**Type C**
   The partial ARG is unchanged but the joining branch changes, so the newly
   threaded lineage recombined.  The transition is conditioned on that event,
   equivalently using :math:`\rho\rightarrow\infty` in the kernel.

Recombination time
==================

The two HMMs locate a recombination boundary but do not infer its exact time.
If the recombining branch begins at :math:`l`, the two neighboring joining
times imply an upper bound :math:`u`, and re-coalescence occurs at :math:`v`,
the conditional density is proportional to :math:`e^{-(v-x)}` for
:math:`l<x<u`.  SINGER uses its median,

.. math::

   x_{1/2}=\log\!\left(\frac{e^l+e^u}{2}\right).

.. literalinclude:: ../../../watchgen/mini_singer.py
   :language: python
   :pyobject: time_transition_matrix
