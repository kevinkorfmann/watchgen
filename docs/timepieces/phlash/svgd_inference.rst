.. _phlash_svgd:

.. _phlash_svgd_inference:

========================
Particle-Based Inference
========================

PHLASH 1.0.6 uses BlackJAX's Stein variational gradient descent with AMSGrad.
SVGD evolves an ensemble :math:`x_1,\ldots,x_J`.  For a positive-definite kernel
:math:`k`, its idealized direction at particle :math:`x_i` is

.. math::

   \varphi(x_i)=\frac{1}{J}\sum_{j=1}^{J}
   \left[k(x_j,x_i)\nabla\log p(x_j)
   +\nabla_{x_j}k(x_j,x_i)\right].

The first term moves particles toward high target density; the second prevents
collapse.  ``svgd_direction`` implements this equation for an RBF kernel and the
tests verify that two particles repel symmetrically when the target score is
zero.

What the particles mean
=======================

The paper describes the output as a posterior ensemble and used 500 particles
in its experiments.  SVGD is nevertheless a deterministic, optimization-based
approximation once its initial particles and stochastic minibatches are fixed;
the particles are not independent MCMC draws.  Composite-likelihood dependence
and variational approximation both matter when interpreting credible bands.

In the released ``fit`` routine, initial transformed parameters are Gaussian
perturbations around a default model, gradients come from JAX around
``log_density``, and chunks are resampled during optimization.  Optional held-out
data control early stopping.  The old mini loop used an invented force toward
zero plus random noise; it has been removed because it could not validate PHLASH.
