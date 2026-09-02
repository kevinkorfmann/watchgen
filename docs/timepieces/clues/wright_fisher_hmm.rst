.. _clues_wright_fisher:

====================================
Frequency Grid and Backward Dynamics
====================================

CLUES2 uses :math:`K` quantiles of a Beta(1/2, 1/2) distribution, including 0
and 1.  This places more grid points near the absorbing boundaries, where drift
is small.  The default CLI value is 450, but convergence should be checked by
increasing ``--df`` for the analysis at hand.

Population-size boundary
========================

The command-line ``--N`` and the CLUES2 paper define :math:`N_{haploid}` as the
number of gene copies, equivalently the inverse coalescence rate.  In
``inference.py``, CLUES2 **divides it by two** before passing
:math:`N_{diploid}=N_{haploid}/2` to its internal kernels.  Thus the internal
source expression

.. math::

   \sigma^2 = \frac{x(1-x)}{2N_{diploid}}

is exactly the paper-level expression ``x(1 - x) / N_haploid``.  Passing the
public haploid value directly to the internal kernel would halve the variance.

Backward mean
=============

With genotype fitnesses :math:`1`, :math:`1+hs`, and :math:`1+s`, CLUES2 uses
the following inverse deterministic selection map as its backward Gaussian
mean:

.. math::

   \mu(x;s,h)=x+
   \frac{s(x-1)x[-x+h(-1+2x)]}
        {-1+s[2h(x-1)-x]x}.

When :math:`s=0`, :math:`\mu=x`.  Positive selection generally shifts the
frequency downward going into the past.

Discretized transition
======================

For interior source bin :math:`x_i`, a normal distribution with mean
:math:`\mu_i` and variance :math:`x_i(1-x_i)/N_{haploid}` is integrated between
adjacent-bin midpoints.  Probability below the first midpoint goes to loss and
probability above the final midpoint goes to fixation.  Those boundary states
are absorbing.

The source computes only a band around :math:`\mu_i\pm3.3\sigma_i`, evaluates a
2,000-point interpolated normal CDF, and renormalizes the retained row.  This is
a numerical approximation, not an exact Wright--Fisher binomial transition.

.. code-block:: python

   from watchgen.mini_clues import build_frequency_bins, build_transition_matrix

   frequencies, _, _ = build_frequency_bins(9)
   log_transition = build_transition_matrix(
       frequencies, n_haploid=200, s=0.02
   )

The middle row of this example is pinned in the tests to official source commit
``b20dc5d`` at approximately :math:`2\times10^{-15}` absolute tolerance.
