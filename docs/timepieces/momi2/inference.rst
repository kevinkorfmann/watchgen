.. _momi2_inference:

=============================================
Automatic Differentiation & Inference
=============================================

   *The mainspring of the watch: autograd-powered optimization turns the expected SFS into inferred history.*

.. epigraph::

   "Automatic differentiation provides exact gradients through the entire
   computation, enabling efficient optimization of complex demographic models."

   -- Kamm, Terhorst, Song, and Durbin (2017)

Step 1: The Likelihood Function
================================

Given an observed SFS :math:`\mathbf{D}` and a demographic model that predicts
expected SFS :math:`\mathbf{M}(\boldsymbol{\Theta})`, ``momi2`` computes the
**composite log-likelihood** by treating each SNP as an independent draw from
a multinomial distribution over allele configurations.

.. math::

   \ell(\boldsymbol{\Theta}) = \sum_{\text{configs } c} D_c \ln M_c(\boldsymbol{\Theta})

where :math:`D_c` is the observed count of configuration :math:`c` and
:math:`M_c(\boldsymbol{\Theta})` is the expected proportion of SNPs with that
configuration under the model.

Alternatively, ``momi2`` can use a **Poisson likelihood** that also fits the
total number of SNPs:

.. math::

   \ell_{\text{Poisson}}(\boldsymbol{\Theta}) = \sum_c \left[ D_c \ln M_c(\boldsymbol{\Theta}) - M_c(\boldsymbol{\Theta}) \right]

.. code-block:: python

   import numpy as np

   def multinomial_log_likelihood(observed_sfs, expected_sfs):
       """Composite log-likelihood under the multinomial model.

       observed_sfs: array of observed configuration counts
       expected_sfs: array of expected proportions (normalized to sum to 1)
       """
       # normalize expected SFS to probabilities
       expected_probs = expected_sfs / expected_sfs.sum()
       # avoid log(0) by masking zero-count entries
       mask = observed_sfs > 0
       ll = np.sum(observed_sfs[mask] * np.log(expected_probs[mask]))
       return ll

   def poisson_log_likelihood(observed_sfs, expected_sfs):
       """Composite log-likelihood under the Poisson model.

       expected_sfs: array of expected counts (not normalized)
       """
       mask = observed_sfs > 0
       ll = np.sum(
           observed_sfs[mask] * np.log(expected_sfs[mask])
           - expected_sfs[mask]
       )
       return ll

.. admonition:: Probability Aside -- Multinomial vs. Poisson likelihood

   The multinomial likelihood conditions on the total number of SNPs
   :math:`S = \sum_c D_c`, making it insensitive to the overall mutation
   rate :math:`\theta`. This is useful when :math:`\theta` is a nuisance
   parameter. The Poisson likelihood fits :math:`S` as well, providing
   information about the absolute scale. In ``momi2``, the default is
   multinomial (``normalized=True``), but the Poisson option is available
   for joint estimation of :math:`\theta`.

Step 2: Automatic Differentiation with autograd
=================================================

The key innovation in ``momi2``'s inference machinery is the use of
**automatic differentiation** (AD) via the ``autograd`` library. Every
numerical operation in the SFS computation -- eigendecomposition, matrix
exponentials, convolutions, tensor products -- is traced by ``autograd``,
which can then compute exact gradients of the log-likelihood with respect
to all demographic parameters.

.. math::

   \nabla_{\boldsymbol{\Theta}} \ell = \left( \frac{\partial \ell}{\partial \Theta_1}, \frac{\partial \ell}{\partial \Theta_2}, \ldots \right)

This is computed by **reverse-mode AD** (backpropagation): a single forward
pass computes :math:`\ell(\boldsymbol{\Theta})`, then a single backward pass
computes all partial derivatives. The cost is roughly 2--3 times the forward
pass, regardless of the number of parameters.

.. code-block:: python

   import autograd
   import autograd.numpy as np

   def make_objective(observed_sfs, demographic_model_func):
       """Create a differentiable objective function.

       demographic_model_func: maps parameter vector -> DemographicModel
       """
       def objective(params):
           model = demographic_model_func(params)
           expected = compute_expected_sfs(model)
           return -multinomial_log_likelihood(observed_sfs, expected)

       # autograd computes the gradient automatically
       grad_objective = autograd.grad(objective)
       val_and_grad = autograd.value_and_grad(objective)

       return objective, grad_objective, val_and_grad

The advantages over alternative gradient methods:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Method
     - Accuracy
     - Cost per gradient
     - Implementation effort
   * - Finite differences
     - :math:`O(\epsilon)` truncation error
     - :math:`O(p)` forward passes
     - Trivial
   * - Hand-coded derivatives
     - Exact (if correct)
     - 1 backward pass
     - Very high, error-prone
   * - Automatic differentiation
     - Exact (to machine precision)
     - 1 backward pass
     - None (autograd handles it)

.. admonition:: Calculus Aside -- How autograd works

   ``autograd`` replaces ``numpy`` operations with traced versions that record
   the computation graph. Each primitive operation (addition, multiplication,
   ``exp``, ``einsum``, etc.) has a registered **vector-Jacobian product** (VJP)
   rule. During the backward pass, autograd chains these VJPs using the chain
   rule to propagate gradients from the output back to the inputs.

   For operations not natively supported (like Cython-accelerated convolutions
   or ``scipy.special.expi``), ``momi2`` registers custom VJPs:

   .. code-block:: python

      from autograd.extend import primitive, defvjp

      @primitive
      def convolve_sum_axes(A, B):
          # Cython implementation
          ...

      defvjp(convolve_sum_axes,
          lambda ans, A, B: lambda g: transposed_convolve(g, B),
          lambda ans, A, B: lambda g: transposed_convolve(g.T, A))

Step 3: Deterministic Optimization
====================================

For small to medium problems, ``momi2`` uses **deterministic gradient-based
optimization** via ``scipy.optimize.minimize``. The ``find_mle`` method wraps
this with autograd-computed gradients:

.. code-block:: python

   from scipy.optimize import minimize

   def find_mle(objective, x0, method='tnc', bounds=None):
       """Find maximum likelihood estimates using deterministic optimization.

       method: 'tnc' (truncated Newton) or 'L-BFGS-B'
       bounds: parameter bounds (lower, upper) for each parameter
       """
       val_and_grad = autograd.value_and_grad(objective)

       def scipy_objective(x):
           val, grad = val_and_grad(x)
           return float(val), np.array(grad, dtype=float)

       result = minimize(
           scipy_objective,
           x0,
           method=method,
           jac=True,   # we provide the gradient
           bounds=bounds,
       )
       return result.x, result.fun

The two recommended methods:

- **TNC** (Truncated Newton Conjugate-gradient): Uses Hessian-vector products
  (which autograd can also compute) for second-order convergence. Good for
  problems with 10--50 parameters.
- **L-BFGS-B**: A quasi-Newton method that approximates the Hessian from
  gradient history. Good for larger parameter spaces. Supports box constraints.

Step 4: Stochastic Optimization
================================

For large datasets with many SNP configurations, evaluating the full likelihood
at every step can be expensive. ``momi2`` supports **stochastic gradient
methods** that subsample the SNPs:

**ADAM** -- Adaptive moment estimation with bias correction:

.. math::

   m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t

   v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2

   \hat{m}_t &= m_t / (1-\beta_1^t)

   \hat{v}_t &= v_t / (1-\beta_2^t)

   \Theta_{t+1} &= \Theta_t - \alpha \, \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)

**SVRG** (Stochastic Variance Reduced Gradient) -- Reduces the variance of
stochastic gradients by periodically computing the full gradient and using it
as a control variate:

.. math::

   g^{\text{SVRG}}_t = \nabla_t f_i(\Theta_t) - \nabla_t f_i(\bar{\Theta}) + \nabla f(\bar{\Theta})

where :math:`\bar{\Theta}` is a "pivot" at which the full gradient
:math:`\nabla f(\bar{\Theta})` was computed.

.. code-block:: python

   def stochastic_find_mle(objective, x0, data, method='adam',
                           snps_per_minibatch=100, stepsize=0.01,
                           num_epochs=100):
       """Stochastic optimization for large datasets.

       Subsamples SNPs into minibatches for gradient estimation.
       """
       x = x0.copy()

       if method == 'adam':
           m = np.zeros_like(x)
           v = np.zeros_like(x)
           b1, b2, eps = 0.9, 0.999, 1e-8

           for epoch in range(num_epochs):
               for batch in make_minibatches(data, snps_per_minibatch):
                   g = autograd.grad(objective)(x, batch)
                   m = b1 * m + (1 - b1) * g
                   v = b2 * v + (1 - b2) * g**2
                   mhat = m / (1 - b1**(epoch + 1))
                   vhat = v / (1 - b2**(epoch + 1))
                   x = x - stepsize * mhat / (np.sqrt(vhat) + eps)
                   x = np.clip(x, bounds[:, 0], bounds[:, 1])

       return x

.. admonition:: When to use stochastic methods

   Deterministic methods (TNC, L-BFGS-B) are preferred for most problems
   because they converge more reliably. Use stochastic methods when:

   - The dataset has millions of SNPs and full-gradient evaluation is slow
   - You want to explore the likelihood surface broadly before refining
   - The number of unique configurations is very large (many populations)

Step 5: Parameter Constraints and Transforms
==============================================

Demographic parameters have natural constraints: population sizes must be
positive, admixture fractions must be between 0 and 1, and split times must
respect the temporal ordering of events.

``momi2`` handles these through:

1. **Box constraints**: Passed directly to L-BFGS-B or TNC via the ``bounds``
   argument
2. **Parameter transforms**: Working in log-space for sizes and times, or
   logit-space for fractions, so the optimizer works in an unconstrained space
3. **Temporal ordering**: Built into the demographic model specification, not
   the optimizer

.. code-block:: python

   def transform_params(params, param_types):
       """Transform parameters to unconstrained space.

       param_types: list of 'log' (positive), 'logit' (0-1), or 'none'
       """
       transformed = np.zeros_like(params)
       for i, (p, ptype) in enumerate(zip(params, param_types)):
           if ptype == 'log':
               transformed[i] = np.log(p)
           elif ptype == 'logit':
               transformed[i] = np.log(p / (1 - p))
           else:
               transformed[i] = p
       return transformed

   def inverse_transform(transformed, param_types):
       """Transform back to natural parameter space."""
       params = np.zeros_like(transformed)
       for i, (t, ptype) in enumerate(zip(transformed, param_types)):
           if ptype == 'log':
               params[i] = np.exp(t)
           elif ptype == 'logit':
               params[i] = 1.0 / (1.0 + np.exp(-t))
           else:
               params[i] = t
       return params

Step 6: Uncertainty Quantification
====================================

After finding the MLE :math:`\hat{\boldsymbol{\Theta}}`, ``momi2`` can
estimate uncertainty via:

**The Hessian matrix.** Autograd can compute the full Hessian of the
log-likelihood:

.. math::

   H_{ij} = \frac{\partial^2 \ell}{\partial \Theta_i \partial \Theta_j} \bigg|_{\hat{\boldsymbol{\Theta}}}

The inverse Hessian gives the asymptotic covariance matrix, and
:math:`\sqrt{(-H^{-1})_{ii}}` gives approximate standard errors.

.. code-block:: python

   def compute_standard_errors(objective, mle_params):
       """Compute approximate standard errors from the Hessian."""
       hessian_func = autograd.hessian(objective)
       H = hessian_func(mle_params)
       # negative Hessian of log-likelihood = observed Fisher information
       cov = np.linalg.inv(-H)
       se = np.sqrt(np.diag(cov))
       return se, cov

**Bootstrap.** For more robust uncertainty estimates (especially when the
composite likelihood assumption is violated due to linkage), ``momi2``
supports parametric and nonparametric bootstrap:

1. Resample SNPs (or blocks of SNPs) with replacement
2. Re-estimate parameters on each bootstrap sample
3. Use the distribution of bootstrap estimates for confidence intervals

.. admonition:: Probability Aside -- Why the Hessian can mislead

   The composite log-likelihood treats SNPs as independent, but linked SNPs
   are correlated. This means the Hessian-based standard errors are typically
   **too small** (overconfident). Block bootstrap, which resamples contiguous
   genomic regions, accounts for linkage and gives more realistic uncertainty
   estimates. This is analogous to the Godambe Information Matrix correction
   used in ``moments`` (see :ref:`demographic_inference`).

Step 7: Goodness-of-Fit Statistics
====================================

Beyond the SFS itself, ``momi2`` can compute **f-statistics** and other
summary statistics to assess model fit:

.. list-table::
   :header-rows: 1
   :widths: 15 45 40

   * - Statistic
     - Definition
     - What it measures
   * - :math:`f_2(A, B)`
     - :math:`E[(p_A - p_B)^2]`
     - Genetic drift between A and B
   * - :math:`f_3(C; A, B)`
     - :math:`E[(p_C - p_A)(p_C - p_B)]`
     - Whether C is admixed between A and B (negative = admixture)
   * - :math:`f_4(A, B; C, D)`
     - :math:`E[(p_A - p_B)(p_C - p_D)]`
     - Shared drift / treeness test
   * - Patterson's D
     - :math:`\frac{\text{ABBA} - \text{BABA}}{\text{ABBA} + \text{BABA}}`
     - Gene flow between non-sister taxa

These are all **linear functions** of the SFS (see :ref:`coalescent_sfs`,
Step 6), so ``momi2`` computes them by passing the appropriate weight vectors
through the tensor machinery. Autograd gradients are available, enabling
optimization of demographic models to fit these statistics.

.. code-block:: python

   def f2_weights(n_A, n_B):
       """Weight vector for f2(A, B) = E[(p_A - p_B)^2].

       Returns a (n_A+1) x (n_B+1) weight matrix.
       """
       p_A = np.arange(n_A + 1) / n_A
       p_B = np.arange(n_B + 1) / n_B
       # f2 = E[(p_A - p_B)^2] = sum over configs of (i/n_A - j/n_B)^2 * SFS[i,j]
       W = np.outer(p_A, np.ones(n_B + 1)) - np.outer(np.ones(n_A + 1), p_B)
       return W**2

   def f3_weights(n_C, n_A, n_B):
       """Weight vector for f3(C; A, B) = E[(p_C - p_A)(p_C - p_B)].

       Negative f3 indicates admixture of C from A and B.
       """
       p_C = np.arange(n_C + 1) / n_C
       p_A = np.arange(n_A + 1) / n_A
       p_B = np.arange(n_B + 1) / n_B
       # 3-way outer product
       W = np.zeros((n_C + 1, n_A + 1, n_B + 1))
       for ic in range(n_C + 1):
           for ia in range(n_A + 1):
               for ib in range(n_B + 1):
                   W[ic, ia, ib] = (p_C[ic] - p_A[ia]) * (p_C[ic] - p_B[ib])
       return W

Step 8: The Complete Inference Pipeline
=========================================

Putting it all together, a typical ``momi2`` analysis follows this workflow:

.. code-block:: python

   # Step 1: Load data
   import momi

   sfs = momi.Sfs.load("observed_sfs.npz")

   # Step 2: Define the demographic model
   model = momi.DemographicModel(N_e=1e4, gen_time=25, muts_per_gen=1.25e-8)
   model.add_leaf("EUR", N=1e4)
   model.add_leaf("AFR", N=1e4)
   model.set_size("EUR", t=0, N=1e5, g=0.01)  # recent growth
   model.move_lineages("EUR", "AFR", t=2000)    # split at 2000 gen ago
   model.set_size("AFR", t=2000, N=2e4)         # ancestral size

   # Step 3: Set up optimization
   model.add_size_param("N_EUR", "EUR")
   model.add_time_param("T_split")
   model.add_growth_param("g_EUR")

   # Step 4: Fit the model
   model.set_data(sfs)
   result = model.optimize(method="TNC")

   # Step 5: Get results
   print("MLE parameters:", result.parameters)
   print("Log-likelihood:", result.log_likelihood)

   # Step 6: Uncertainty (via bootstrap or Hessian)
   se = result.standard_errors()
   print("Standard errors:", se)

.. admonition:: Practical tips for momi2 inference

   - **Start simple**: Begin with a model with few parameters and add
     complexity gradually. Check that each added parameter improves the fit.
   - **Multiple starts**: Run the optimizer from several different starting
     points to avoid local optima.
   - **Check f-statistics**: Even if the SFS fit looks good, compute
     f-statistics to verify the model captures the right population
     relationships.
   - **Use block bootstrap**: Hessian-based standard errors underestimate
     uncertainty. Use genomic block bootstrap for publication-quality
     confidence intervals.

Exercises
=========

.. admonition:: Exercise 1: Gradient verification

   For a simple two-population model, compute the gradient of the log-likelihood
   using (a) autograd, (b) finite differences with step size
   :math:`\epsilon = 10^{-5}`. Verify they agree to several decimal places.

.. admonition:: Exercise 2: Optimizer comparison

   Fit a three-parameter model (ancestral size, derived size, split time) using
   TNC and L-BFGS-B. Compare the number of function evaluations and the final
   log-likelihood. Do they converge to the same optimum?

.. admonition:: Exercise 3: Uncertainty calibration

   Simulate 100 datasets under a known demographic model, fit each one, and
   check whether the 95% confidence intervals (from the Hessian) contain the
   true parameter values 95% of the time. How does block bootstrap compare?

.. admonition:: Exercise 4: f-statistics as model diagnostics

   Fit a tree model (no admixture) to data simulated under a model *with*
   admixture. Compute :math:`f_3` for the admixed population. Does the
   statistic detect the model misspecification?
