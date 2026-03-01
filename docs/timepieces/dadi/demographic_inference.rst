.. _dadi_demographic_inference:

========================
Demographic Inference
========================

   *The mainspring -- where the model meets reality and parameters are adjusted until the predicted dial matches observation.*

With the diffusion equation solved and the expected SFS in hand, the final
step is inference: finding the demographic parameters that best explain the
observed data. This chapter covers the likelihood framework, optimization
algorithms, uncertainty quantification, and the extensions that make ``dadi``
a practical tool for population genomic analysis.

The Poisson Composite Likelihood
==================================

``dadi`` uses the same **Poisson Random Field (PRF)** model as ``moments``
(see :ref:`demographic_inference` in the moments Timepiece for a detailed
derivation). Under the infinite-sites model, each segregating site is an
independent Poisson draw, and the SFS entries are independent Poisson random
variables.

The log-likelihood is:

.. math::

   \ell(\boldsymbol{\Theta}) = \sum_j \left[ -M_j(\boldsymbol{\Theta}) + D_j \ln M_j(\boldsymbol{\Theta}) - \ln D_j! \right]

where :math:`D_j` is the observed count in SFS bin :math:`j` and
:math:`M_j(\boldsymbol{\Theta})` is the expected count under demographic
parameters :math:`\boldsymbol{\Theta}`.

In ``dadi``'s code, this is ``Inference.ll(model, data)``:

.. code-block:: python

   import dadi

   # model: expected SFS from the diffusion solver
   # data: observed SFS from real data
   ll = dadi.Inference.ll(model, data)

Multinomial vs. Poisson Likelihood
====================================

The Poisson likelihood requires knowing :math:`\theta` (the population-scaled
mutation rate), which sets the overall scale of the expected SFS. In many
analyses, :math:`\theta` is a nuisance parameter -- you want to infer relative
sizes and times, not the absolute mutation rate.

``dadi`` offers a **multinomial likelihood** (``Inference.ll_multinom``) that
analytically maximizes over :math:`\theta`:

.. math::

   \hat{\theta} = \frac{\sum_j D_j}{\sum_j M_j^{(0)}}

where :math:`M_j^{(0)}` is the expected SFS computed with :math:`\theta = 1`.
The model SFS is then rescaled by :math:`\hat{\theta}` before computing the
Poisson likelihood.

.. code-block:: python

   # Multinomial likelihood (automatically optimizes theta)
   ll = dadi.Inference.ll_multinom(model, data)

   # Optimal theta scaling factor
   theta_opt = dadi.Inference.optimal_sfs_scaling(model, data)

The multinomial approach is the default for most analyses, since it eliminates
one parameter from the optimization.

Optimization
==============

``dadi`` provides several optimization functions, all following the same
interface: minimize the negative log-likelihood over the demographic parameters.

**BFGS in log-space (recommended):**

The primary optimizer is ``Inference.optimize_log``, which transforms
parameters to log-space before applying the BFGS quasi-Newton method:

.. code-block:: python

   import dadi

   # p0: initial parameter guess
   # data: observed SFS
   # model_func: demographic model function
   # pts: grid sizes for extrapolation
   popt = dadi.Inference.optimize_log(
       p0=[1.0, 0.5],           # initial guess (nu, T)
       data=data,                # observed SFS
       model_func=func_ex,      # extrapolated model function
       pts=[40, 50, 60],        # grid sizes
       lower_bound=[1e-2, 0],   # parameter lower bounds
       upper_bound=[100, 3],    # parameter upper bounds
       verbose=1,               # print progress
       maxiter=100               # maximum iterations
   )

The log-space transformation ensures that parameters remain positive and that
the optimizer takes proportionally sized steps (a step of 0.1 in log-space
corresponds to a ~10% change in the parameter).

**Other optimizers:**

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Function
     - Method
     - When to use
   * - ``optimize_log``
     - BFGS (log-space)
     - Default; fast, uses gradients
   * - ``optimize``
     - BFGS (linear)
     - When log transform is inappropriate
   * - ``optimize_log_lbfgsb``
     - L-BFGS-B
     - When bounds are important
   * - ``optimize_log_fmin``
     - Nelder-Mead
     - Robust to noisy likelihoods
   * - ``optimize_log_powell``
     - Powell
     - No gradient needed
   * - ``optimize_grid``
     - Grid search
     - Exploring the landscape
   * - ``optimize_cons``
     - SLSQP
     - With equality/inequality constraints

**Fixed parameters:**

To fix certain parameters while optimizing others, pass a ``fixed_params``
list where ``None`` means "optimize" and a value means "fix":

.. code-block:: python

   # Fix T=0.5, optimize only nu
   popt = dadi.Inference.optimize_log(
       p0=[1.0, 0.5],
       data=data,
       model_func=func_ex,
       pts=[40, 50, 60],
       fixed_params=[None, 0.5]  # optimize nu, fix T=0.5
   )

A Complete Inference Example
==============================

Here is a complete workflow for fitting a two-epoch model:

.. code-block:: python

   import dadi
   import numpy as np

   # ------- 1. Load or simulate data -------
   # (Here we simulate for demonstration)
   def two_epoch(params, ns, pts):
       nu, T = params
       xx = dadi.Numerics.default_grid(pts)
       phi = dadi.PhiManip.phi_1D(xx)
       phi = dadi.Integration.one_pop(phi, xx, T, nu=nu)
       return dadi.Spectrum.from_phi(phi, ns, (xx,))

   func_ex = dadi.Numerics.make_extrap_log_func(two_epoch)

   # "True" parameters: 2x expansion, 0.1 time units ago
   true_params = [2.0, 0.1]
   true_model = func_ex(true_params, ns=[20], pts=[40, 50, 60])

   # Generate synthetic data (Poisson sampling)
   data = true_model * 1000  # scale by theta
   data = dadi.Spectrum(np.random.poisson(data))

   # ------- 2. Optimize -------
   p0 = [1.0, 0.5]  # initial guess (deliberately wrong)
   popt = dadi.Inference.optimize_log(
       p0, data, func_ex,
       pts=[40, 50, 60],
       lower_bound=[1e-2, 1e-3],
       upper_bound=[100, 3],
       verbose=0,
       maxiter=100
   )

   # ------- 3. Evaluate fit -------
   model = func_ex(popt, ns=[20], pts=[40, 50, 60])
   ll_opt = dadi.Inference.ll_multinom(model, data)
   theta = dadi.Inference.optimal_sfs_scaling(model, data)

   print(f"Best-fit: nu={popt[0]:.3f}, T={popt[1]:.4f}")
   print(f"Log-likelihood: {ll_opt:.2f}")
   print(f"Optimal theta: {theta:.2f}")

Uncertainty: The Godambe Information Matrix
=============================================

Point estimates are incomplete without uncertainty quantification. ``dadi``
uses the **Godambe Information Matrix** (GIM), a sandwich estimator that
accounts for linkage between sites.

The standard Fisher Information Matrix (FIM) assumes independent sites:

.. math::

   \mathcal{I}(\boldsymbol{\Theta}) = -E\left[\frac{\partial^2 \ell}{\partial \Theta_i \partial \Theta_j}\right]

But sites in a genome are not independent -- they are linked. The GIM corrects
for this using bootstrap replicates of the SFS:

.. math::

   G = H \, J^{-1} \, H

where :math:`H` is the Hessian of the log-likelihood (computed by finite
differences) and :math:`J` is the variance of the score vector (estimated from
bootstrap replicates):

.. math::

   J = \frac{1}{B}\sum_{b=1}^{B} U_b U_b^T

Here :math:`U_b = \nabla \ell(\boldsymbol{\Theta}; \mathbf{D}_b)` is the
gradient of the log-likelihood evaluated on bootstrap replicate :math:`b`.

The standard errors are :math:`\text{SE}(\hat{\Theta}_i) = \sqrt{(G^{-1})_{ii}}`:

.. code-block:: python

   import dadi

   # all_boot: list of bootstrap SFS replicates
   # p0: best-fit parameters
   # data: observed SFS
   uncerts = dadi.Godambe.GIM_uncert(
       func_ex,
       pts=[40, 50, 60],
       all_boot=all_boot,
       p0=popt,
       data=data,
       multinom=True,
       eps=0.01    # step size for finite differences
   )
   print(f"Uncertainties: {uncerts}")

Model Comparison
==================

With uncertainties in hand, you can compare competing demographic models using:

**Likelihood Ratio Test (LRT):**

For nested models (one is a special case of the other), twice the
log-likelihood difference follows a :math:`\chi^2` distribution:

.. math::

   \Lambda = 2(\ell_{\text{complex}} - \ell_{\text{simple}})

The Godambe-adjusted LRT (``Godambe.LRT_adjust``) corrects the test statistic
for linkage.

**AIC (Akaike Information Criterion):**

For non-nested models:

.. math::

   \text{AIC} = 2k - 2\ell

where :math:`k` is the number of parameters. Lower AIC indicates a better
model (balancing fit against complexity).

DFE Inference
===============

A powerful extension of ``dadi`` is inference of the **Distribution of Fitness
Effects** (DFE) -- the distribution of selection coefficients across new
mutations.

The approach:

1. **Cache spectra:** compute the expected SFS at many values of the selection
   coefficient :math:`\gamma`, using a demographic model fit to synonymous
   (neutral) sites:

   .. code-block:: python

      import dadi.DFE

      cache = dadi.DFE.Cache1D(
          params=popt,           # demographic parameters
          ns=[20],               # sample sizes
          demo_sel_func=my_sel_model,  # demographic+selection model
          pts=[40, 50, 60],
          gamma_bounds=(1e-4, 2000),
          gamma_pts=500
      )

2. **Integrate over DFE:** weight the cached spectra by a parametric
   distribution (e.g., gamma, lognormal) and sum:

   .. code-block:: python

      # Integrate cached spectra over a gamma DFE
      sel_model = cache.integrate(
          params=dfe_params,
          ns=[20],
          sel_dist=dadi.DFE.PDFs.gamma,
          theta=theta_ns
      )

3. **Optimize DFE parameters:** fit the DFE shape and scale to the
   non-synonymous SFS.

This two-step approach (demographic model from synonymous sites, DFE from
non-synonymous sites) cleanly separates demography from selection.

Ancestral Misidentification
=============================

Polarizing the SFS (assigning derived vs. ancestral alleles) requires an
outgroup. Errors in polarization -- **ancestral misidentification** -- fold
some of the SFS entries across the spectrum. ``dadi`` corrects for this with
``Spectrum.apply_anc_state_misid(p_misid)``, which mixes each SFS entry
:math:`j` with its complement :math:`n - j` according to a misidentification
probability :math:`p`:

.. math::

   F_j^{\text{obs}} = (1-p) \, F_j^{\text{true}} + p \, F_{n-j}^{\text{true}}

This can be applied to the model SFS before comparing to data, adding
:math:`p` as an additional parameter to optimize.

Demes Integration
===================

For standardized model specification, ``dadi`` integrates with the **Demes**
format (Gower et al. 2022), which specifies demographic models as structured
YAML files. This allows the same model to be used across different inference
tools (``dadi``, ``moments``, ``momi2``):

.. code-block:: python

   import demes
   import dadi

   # Load a Demes-format model
   graph = demes.load("my_model.yaml")

   # Convert to a dadi model function
   # and compute the expected SFS
   fs = dadi.Spectrum.from_demes(graph, sampled_demes=["Pop1", "Pop2"],
                                  sample_sizes=[20, 20])

Summary
========

The inference machinery of ``dadi`` combines:

- **Poisson composite likelihood** from the PRF model (shared with ``moments``)
- **BFGS optimization** in log-parameter space for efficient gradient-based search
- **Godambe Information Matrix** for linkage-corrected uncertainty
- **DFE inference** via cached spectra and parametric integration
- **Ancestral misidentification correction** for imperfect polarization
- **Demes integration** for standardized model specification

Together with the diffusion equation solver and Richardson extrapolation from
the previous chapters, these components form a complete, mature framework for
demographic inference from the site frequency spectrum -- the framework that
``moments`` and ``momi2`` were designed to complement and, in some respects,
improve upon.
