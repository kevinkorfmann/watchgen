.. _demographic_inference:

========================
Demographic Inference
========================

   *The case and dial: reading the time from the mechanism.*

We've built the engine (moment equations, :ref:`moment_equations`) and the data
structure (SFS, :ref:`the_frequency_spectrum`). Now we connect them: given
observed genetic data, find the demographic history that best explains it.
This is **demographic inference** -- the reason ``moments`` exists.

In the watch metaphor, the previous chapters designed the gear train (the ODEs)
and described the dial face (the SFS).  This chapter is where we **adjust
parameters until the predicted dial matches observation** -- turning the
crown until the model watch shows the same time as the real one.

By the end of this chapter, you will understand the likelihood function, the
optimization algorithms, and how to quantify uncertainty in your estimates.


Step 1: The Likelihood Function
================================

We observe an SFS :math:`\mathbf{D}` from real data. We compute a model SFS
:math:`\mathbf{M}(\boldsymbol{\Theta})` for a given set of demographic parameters
:math:`\boldsymbol{\Theta}`. The question: how likely is the observed data under
this model?

.. admonition:: Probability Aside -- What is a likelihood?

   In everyday language, "likelihood" and "probability" are synonyms.  In
   statistics they are different.  The **probability** of data :math:`D`
   given parameters :math:`\Theta` is :math:`P(D \mid \Theta)` -- it
   answers "how probable is this data if the model is true?"  The
   **likelihood** of parameters :math:`\Theta` given data :math:`D` is
   the *same function* but viewed from the other direction:
   :math:`L(\Theta) = P(D \mid \Theta)`.  It answers "how well do these
   parameters explain the data I already observed?"

   **Maximum likelihood estimation** (MLE) finds the parameters
   :math:`\hat{\Theta}` that maximize :math:`L(\Theta)`.  Equivalently
   (and more conveniently), we maximize the **log-likelihood**
   :math:`\ell(\Theta) = \ln L(\Theta)`, because logarithms turn products
   into sums and are numerically more stable.

The **Poisson Random Field** (PRF) approximation treats each SFS entry as an
independent Poisson observation:

.. math::

   \text{SFS}[j] \sim \text{Poisson}(M_j)

where :math:`M_j` is the expected count under the model. The log-likelihood is:

.. math::

   \ell(\boldsymbol{\Theta}) = \sum_{j=1}^{n-1}
   \left[ D_j \ln M_j - M_j - \ln(D_j!) \right]

Since :math:`\ln(D_j!)` doesn't depend on the parameters, we can drop it for
optimization:

.. math::

   \ell(\boldsymbol{\Theta}) = \sum_{j=1}^{n-1}
   \left[ D_j \ln M_j - M_j \right] + \text{const.}

**Why Poisson?** Under the infinite-sites model, mutations at different sites are
independent. Each site has a small probability of being segregating and, if
segregating, of having a specific allele count. When events are rare and
independent, the Poisson distribution is the natural model.

.. admonition:: Probability Aside -- From Poisson to the SFS likelihood, step by step

   1. Under the infinite-sites model, each new mutation occurs at a unique
      genomic position.  The number of mutations falling in any frequency
      class :math:`j` is therefore a count of independent rare events across
      many sites -- the classic setup for a Poisson distribution.
   2. The Poisson probability mass function is
      :math:`P(k; \lambda) = e^{-\lambda}\lambda^k / k!`.  Taking the
      logarithm: :math:`\ln P = k \ln\lambda - \lambda - \ln(k!)`.
   3. Setting :math:`k = D_j` (observed count) and :math:`\lambda = M_j`
      (model prediction), and summing over all frequency bins, yields the
      log-likelihood above.
   4. The :math:`-\ln(D_j!)` term is a constant that does not change with
      :math:`\Theta`, so it can be dropped during optimization.

.. code-block:: python

   import numpy as np

   def poisson_log_likelihood(data_sfs, model_sfs):
       """Compute the Poisson log-likelihood of data given model.

       Parameters
       ----------
       data_sfs : ndarray
           Observed SFS (counts).
       model_sfs : ndarray
           Expected SFS under the model.

       Returns
       -------
       ll : float
           Log-likelihood (up to a constant).
       """
       n = len(data_sfs) - 1
       ll = 0.0
       for j in range(1, n):
           if model_sfs[j] <= 0:
               if data_sfs[j] > 0:
                   return -np.inf  # impossible observation => -infinity
               continue
           if data_sfs[j] > 0:
               ll += data_sfs[j] * np.log(model_sfs[j])  # D_j * ln(M_j)
           ll -= model_sfs[j]                              # - M_j
       return ll

   # Example: neutral model is the MLE for neutral data
   n = 20
   theta_true = 1000

   # "Observed" data from neutral model (no noise -- the expected SFS itself)
   data = expected_sfs_neutral(n, theta_true)

   # Compare log-likelihood at different theta values
   print("Log-likelihood at different theta values:")
   for theta_test in [500, 800, 1000, 1200, 1500]:
       model = expected_sfs_neutral(n, theta_test)
       ll = poisson_log_likelihood(data, model)
       marker = " <-- true value" if theta_test == theta_true else ""
       print(f"  theta = {theta_test:5d}: ll = {ll:10.2f}{marker}")

.. admonition:: The likelihood is maximized at the true parameters

   In the example above, the log-likelihood is highest at
   :math:`\theta = 1000`, confirming that the Poisson likelihood correctly
   identifies the true parameter. This may seem trivial for a one-parameter
   model, but the same principle scales to models with dozens of parameters.

The likelihood connects the gear train to the observed dial.  Next, we exploit
a special property of the SFS to simplify the optimization.


Step 2: Optimal Theta Scaling
===============================

A key simplification: the SFS scales **linearly** with :math:`\theta`. If you
double the mutation rate, every SFS entry doubles. This means we can separate
:math:`\theta` from the demographic parameters.

Given a model SFS computed at :math:`\theta = 1` (unit-scaled), the optimal
:math:`\theta` is:

.. math::

   \hat{\theta}_{\text{opt}} =
   \frac{\sum_{j=1}^{n-1} D_j}{\sum_{j=1}^{n-1} M_j^{(\theta=1)}}
   = \frac{S_{\text{data}}}{S_{\text{model}}}

where :math:`S` is the total number of segregating sites. This is the
maximum-likelihood estimator of the scaling factor.

.. admonition:: Calculus Aside -- Deriving the optimal scaling analytically

   Let the model SFS at unit :math:`\theta` be :math:`M_j^{(1)}`.  For a
   global scaling constant :math:`c` (so :math:`M_j = c \cdot M_j^{(1)}`),
   the log-likelihood is:

   .. math::

      \ell(c) = \sum_j D_j \ln(c \cdot M_j^{(1)}) - c \cdot M_j^{(1)}
      = \ln c \sum_j D_j + \sum_j D_j \ln M_j^{(1)} - c \sum_j M_j^{(1)}

   Taking the derivative with respect to :math:`c` and setting it to zero:

   .. math::

      \frac{d\ell}{dc} = \frac{\sum_j D_j}{c} - \sum_j M_j^{(1)} = 0
      \quad\Longrightarrow\quad
      \hat{c} = \frac{\sum_j D_j}{\sum_j M_j^{(1)}}

   The second derivative :math:`d^2\ell/dc^2 = -\sum_j D_j / c^2 < 0`
   confirms this is a maximum.  Because the optimal :math:`\theta` can be
   found in closed form, the optimizer only needs to search over the
   demographic parameters (:math:`\nu, T, m, \ldots`), reducing the
   dimensionality of the search by one.

**Why this works**: Taking the derivative of :math:`\ell` with respect to a
global scaling constant :math:`c` (so :math:`M_j \to c \cdot M_j^{(1)}`):

.. math::

   \frac{d\ell}{dc} = \sum_j \frac{D_j}{c} - \sum_j M_j^{(1)} = 0

Solving: :math:`c = \sum_j D_j / \sum_j M_j^{(1)}`.

.. code-block:: python

   def optimal_theta_scaling(data_sfs, model_sfs_unit):
       """Find the optimal theta to scale the model SFS.

       Parameters
       ----------
       data_sfs : ndarray
           Observed SFS.
       model_sfs_unit : ndarray
           Model SFS computed at theta = 1.

       Returns
       -------
       theta_opt : float
       """
       n = len(data_sfs) - 1
       S_data = data_sfs[1:n].sum()       # total observed segregating sites
       S_model = model_sfs_unit[1:n].sum() # total expected at theta=1
       return S_data / S_model if S_model > 0 else 1.0

   # Example
   n = 20
   theta_true = 1000
   data = expected_sfs_neutral(n, theta_true)
   model_unit = expected_sfs_neutral(n, theta=1.0)

   theta_opt = optimal_theta_scaling(data, model_unit)
   print(f"Optimal theta: {theta_opt:.2f} (true: {theta_true})")

With theta handled analytically, we are ready to build a complete inference
pipeline.


Step 3: A Complete Inference Example
=======================================

Let's put it all together: simulate data under a known demographic model, then
try to recover the parameters.

.. code-block:: python

   import moments
   import numpy as np

   # --- Step 1: Define the demographic model ---
   def two_epoch_model(params, ns):
       """Two-epoch model: constant then expansion.

       Parameters
       ----------
       params : (nu, T)
           nu : expansion factor
           T : time since expansion (2*Ne generations)
       ns : (n,)
           Sample size.
       """
       nu, T = params
       fs = moments.Demographics1D.snm(ns)  # start from equilibrium
       fs.integrate([nu], T)                  # integrate through the size change
       return fs

   # --- Step 2: Generate "observed" data ---
   # True parameters: 5-fold expansion, 0.2 * 2*Ne generations ago
   nu_true, T_true = 5.0, 0.2
   theta_true = 2000  # genome-wide

   n = 30
   model_true = two_epoch_model([nu_true, T_true], [n])
   data = model_true * theta_true  # scale by theta to get expected counts

   # Add Poisson noise to simulate real data
   np.random.seed(42)
   data_noisy = np.zeros(n + 1)
   for j in range(1, n):
       data_noisy[j] = np.random.poisson(data[j])  # each bin drawn from Poisson

   print("True parameters: nu=5.0, T=0.2, theta=2000")
   print(f"Total segregating sites: {data_noisy[1:n].sum():.0f}")

   # --- Step 3: Define the objective function ---
   def objective(log_params, data_sfs, ns):
       """Negative log-likelihood for optimization.

       Parameters are in log-space to enforce positivity:
       log_params = [log(nu), log(T)].
       """
       params = np.exp(log_params)       # back-transform to natural scale
       nu, T = params

       model = two_epoch_model([nu, T], ns)

       # Optimally scale theta (closed-form, see Step 2)
       theta_opt = optimal_theta_scaling(data_sfs, model)
       model_scaled = model * theta_opt

       ll = poisson_log_likelihood(data_sfs, model_scaled)
       return -ll  # minimize negative log-likelihood = maximize log-likelihood

   # --- Step 4: Optimize ---
   from scipy.optimize import minimize

   # Initial guess (deliberately wrong to test convergence)
   log_p0 = np.log([2.0, 0.5])

   result = minimize(objective, log_p0, args=(data_noisy, [n]),
                      method='Nelder-Mead',
                      options={'maxiter': 1000, 'xatol': 1e-4})

   nu_hat, T_hat = np.exp(result.x)
   model_hat = two_epoch_model([nu_hat, T_hat], [n])
   theta_hat = optimal_theta_scaling(data_noisy, model_hat)

   print(f"\nEstimated: nu={nu_hat:.3f}, T={T_hat:.4f}, theta={theta_hat:.1f}")
   print(f"True:      nu={nu_true:.3f}, T={T_true:.4f}, theta={theta_true:.1f}")
   print(f"Converged: {result.success}")

.. admonition:: Probability Aside -- Why optimize in log-space?

   Demographic parameters like :math:`\nu` and :math:`T` must be positive.
   Optimizing :math:`\log(\nu)` and :math:`\log(T)` instead of the raw
   values has two benefits: (1) positivity is automatically enforced (the
   exponential back-transform always gives a positive number), and (2) the
   likelihood surface is often smoother in log-space because demographic
   parameters can span orders of magnitude.  This is analogous to using a
   logarithmic scale on a watch's tachymeter -- it compresses the range and
   makes equal *ratios* equally spaced.


Step 4: Using moments' Built-in Optimization
==============================================

``moments`` provides convenience functions that handle the optimization loop,
including log-space transformations, parameter bounds, and multiple optimizer
choices.

.. code-block:: python

   import moments

   # Define model function (moments convention: returns SFS for theta=1)
   def model_func(params, ns):
       nu, T = params
       fs = moments.Demographics1D.snm(ns)  # equilibrium starting point
       fs.integrate([nu], T)                  # forward integration via ODEs
       return fs

   # Optimize using moments' built-in optimizer
   p0 = [2.0, 0.5]  # initial guess
   lower = [0.01, 0.001]   # lower bounds on nu and T
   upper = [100.0, 5.0]    # upper bounds on nu and T

   # moments.Inference.optimize_log uses L-BFGS-B in log-space
   popt = moments.Inference.optimize_log(
       p0, data_noisy, model_func, pts=None,
       lower_bound=lower, upper_bound=upper,
       verbose=0, maxiter=100
   )

   print(f"moments optimizer result: nu={popt[0]:.3f}, T={popt[1]:.4f}")

   # Compute the log-likelihood at the optimum
   model_opt = model_func(popt, [n])
   ll = moments.Inference.ll_multinom(model_opt, data_noisy)
   print(f"Log-likelihood: {ll:.2f}")

.. admonition:: ``ll_multinom`` vs ``ll``

   ``moments`` provides two likelihood functions:

   - ``ll(model, data)``: Poisson log-likelihood (requires theta scaling)
   - ``ll_multinom(model, data)``: Multinomial log-likelihood (theta-independent)

   The multinomial version treats the SFS as a vector of *proportions* rather
   than counts. It is invariant to theta scaling, which means you can optimize
   demographic parameters without worrying about theta. The Poisson version is
   slightly more informative (it also constrains the total mutation rate), but
   the multinomial version is more robust and is often preferred in practice.

.. admonition:: Calculus Aside -- Multinomial vs. Poisson likelihoods

   The Poisson likelihood treats each SFS bin as an independent count.  The
   multinomial likelihood conditions on the total number of segregating sites
   :math:`S = \sum_j D_j` and models only the *proportions*
   :math:`p_j = D_j / S`.  Mathematically:

   .. math::

      \ell_{\text{multi}} = \sum_j D_j \ln\!\left(\frac{M_j}{\sum_k M_k}\right)

   Because the denominator :math:`\sum_k M_k` cancels any global scaling,
   :math:`\theta` drops out entirely.  The Poisson likelihood is strictly more
   informative (it uses the total count :math:`S` as well), but the
   multinomial likelihood is more robust to violations of the Poisson
   assumption (e.g., when sites are not perfectly independent due to linkage).

With the best-fit parameters in hand, the next question is: how much should we
trust them?


Step 5: Uncertainty Quantification
=====================================

Finding the best-fit parameters is only half the job. We also need to know
**how confident** we are in those estimates. ``moments`` provides two approaches.

Fisher Information Matrix (FIM)
--------------------------------

The FIM measures the curvature of the likelihood surface at the optimum. Steeper
curvature = tighter constraint = smaller uncertainty.

.. math::

   I_{ij} = -E\left[\frac{\partial^2 \ell}{\partial \Theta_i \partial \Theta_j}\right]

The standard errors are:

.. math::

   \text{SE}(\hat{\Theta}_i) = \sqrt{(I^{-1})_{ii}}

And the 95% confidence interval:

.. math::

   \hat{\Theta}_i \pm 1.96 \cdot \text{SE}(\hat{\Theta}_i)

.. admonition:: Calculus Aside -- The Fisher Information Matrix intuitively

   Picture the log-likelihood as a landscape with a peak at the MLE.  The
   **second derivatives** (the Hessian matrix) measure how sharply the
   landscape curves away from the peak in each direction.  The FIM is the
   negative expected Hessian -- it captures the average curvature.

   * A tall, narrow peak (large second derivative) means the data strongly
     constrain the parameter: even a small change away from the MLE causes a
     big drop in likelihood.  The standard error is small.
   * A broad, flat peak (small second derivative) means the data provide
     little information about that parameter.  The standard error is large.

   Inverting the FIM gives the variance-covariance matrix of the parameter
   estimates, from which confidence intervals follow.

.. code-block:: python

   def fisher_information_numerical(params, data_sfs, model_func, ns, eps=0.01):
       """Compute the Fisher Information Matrix by numerical differentiation.

       Parameters
       ----------
       params : array-like
           Optimized parameters.
       data_sfs : ndarray
           Observed SFS.
       model_func : callable
           Function(params, ns) -> model SFS.
       ns : list
           Sample sizes.
       eps : float
           Relative step size for finite differences.

       Returns
       -------
       FIM : ndarray of shape (k, k)
           Fisher Information Matrix.
       """
       k = len(params)
       FIM = np.zeros((k, k))

       def neg_ll(p):
           model = model_func(p, ns)
           theta_opt = optimal_theta_scaling(data_sfs, model)
           model_scaled = model * theta_opt
           return -poisson_log_likelihood(data_sfs, model_scaled)

       # Central differences for second derivatives
       for i in range(k):
           for j in range(i, k):
               # Evaluate neg_ll at four perturbed points (++, +-, -+, --)
               p_pp = params.copy(); p_pp[i] *= (1 + eps); p_pp[j] *= (1 + eps)
               p_pm = params.copy(); p_pm[i] *= (1 + eps); p_pm[j] *= (1 - eps)
               p_mp = params.copy(); p_mp[i] *= (1 - eps); p_mp[j] *= (1 + eps)
               p_mm = params.copy(); p_mm[i] *= (1 - eps); p_mm[j] *= (1 - eps)

               # Finite-difference approximation to the second derivative
               d2 = (neg_ll(p_pp) - neg_ll(p_pm) - neg_ll(p_mp) + neg_ll(p_mm))
               d2 /= (params[i] * eps * 2) * (params[j] * eps * 2)

               FIM[i, j] = d2
               FIM[j, i] = d2  # symmetric

       return FIM

   # Example
   params_opt = np.array([nu_hat, T_hat])
   FIM = fisher_information_numerical(params_opt, data_noisy, model_func, [n])

   if np.linalg.det(FIM) > 0:
       cov = np.linalg.inv(FIM)          # covariance matrix = inverse FIM
       se = np.sqrt(np.diag(cov))         # standard errors = sqrt of diagonal
       print("Parameter estimates with 95% CI (FIM):")
       names = ['nu', 'T']
       for name, val, s in zip(names, params_opt, se):
           print(f"  {name}: {val:.4f} +/- {1.96*s:.4f} "
                 f"({val - 1.96*s:.4f}, {val + 1.96*s:.4f})")

Godambe Information Matrix (GIM)
----------------------------------

The FIM assumes the Poisson model is exactly correct. In reality, nearby sites
are correlated because of linkage. The **Godambe Information Matrix** corrects
for this using bootstrap resampling.

The idea: divide the genome into blocks (e.g., 100 kb windows). Resample blocks
with replacement to create bootstrap replicates of the SFS. The variance of the
score function across bootstraps captures the correlation structure.

.. math::

   \text{GIM} = H^{-1} J H^{-1}

where:

- :math:`H` = Hessian of the log-likelihood (same as FIM)
- :math:`J` = empirical variance of the score across bootstraps

The GIM-based standard errors are always **larger** than FIM-based ones (because
linkage makes sites non-independent, so there's less information than the Poisson
model assumes).

.. admonition:: Probability Aside -- Why linkage inflates uncertainty

   The Poisson likelihood assumes each SFS entry is an independent count.
   In reality, nearby sites on a chromosome are inherited together
   (they are in linkage disequilibrium; see :ref:`linkage_disequilibrium`).
   If 100 sites are in strong LD, they behave more like 10 independent
   observations than 100 -- their allele counts are correlated.  The FIM
   doesn't know this, so it overestimates the information content of the
   data and underestimates the standard errors.  The GIM corrects for this
   by measuring the *actual* variance of the score (gradient of the
   log-likelihood) across bootstrap replicates of genomic blocks.  The
   resulting standard errors are always at least as large as the FIM ones,
   and often 2-5 times larger for whole-genome data.

.. code-block:: python

   def godambe_uncertainty(params_opt, data_sfs, model_func, ns,
                            bootstrap_sfss, eps=0.01):
       """Compute parameter uncertainties using the Godambe Information Matrix.

       Parameters
       ----------
       params_opt : array-like
           Optimized parameters.
       data_sfs : ndarray
           Full observed SFS.
       model_func : callable
       ns : list
       bootstrap_sfss : list of ndarray
           SFS from bootstrap resampling of genomic blocks.
       eps : float
           Step size for numerical derivatives.

       Returns
       -------
       se_godambe : ndarray
           Standard errors from GIM.
       """
       k = len(params_opt)

       # H: Hessian (= FIM) from the full data
       H = fisher_information_numerical(params_opt, data_sfs, model_func, ns, eps)

       # Score function: gradient of log-likelihood at the MLE
       def score(p, data):
           grad = np.zeros(k)
           for i in range(k):
               p_plus = p.copy(); p_plus[i] *= (1 + eps)
               p_minus = p.copy(); p_minus[i] *= (1 - eps)

               model_p = model_func(p_plus, ns)
               model_m = model_func(p_minus, ns)
               theta_p = optimal_theta_scaling(data, model_p)
               theta_m = optimal_theta_scaling(data, model_m)

               ll_p = poisson_log_likelihood(data, model_p * theta_p)
               ll_m = poisson_log_likelihood(data, model_m * theta_m)

               # Central-difference approximation to the partial derivative
               grad[i] = (ll_p - ll_m) / (p[i] * 2 * eps)
           return grad

       # J: empirical variance of the score across bootstraps
       scores = np.array([score(params_opt, bs) for bs in bootstrap_sfss])
       J = np.cov(scores, rowvar=False) * len(bootstrap_sfss)

       # GIM = H^{-1} J H^{-1}  (sandwich estimator)
       H_inv = np.linalg.inv(H)
       GIM = H_inv @ J @ H_inv

       return np.sqrt(np.diag(GIM))

.. admonition:: When to use GIM vs FIM

   **Always prefer GIM** when you have real data. The FIM is appropriate only
   when sites are truly independent (e.g., simulation studies where you simulated
   unlinked loci). For real genomic data, linkage between nearby sites inflates
   the variance, and the FIM will underestimate your uncertainty -- sometimes
   dramatically.

With parameter estimates and confidence intervals in hand, we often want to
compare *models* -- not just parameters.


Step 6: Model Comparison
==========================

Often the question isn't "what are the parameters?" but "which model is better?"
For example: does a two-epoch model fit significantly better than a constant-size
model?

Likelihood Ratio Test
-----------------------

If model A is nested within model B (i.e., model A is model B with some parameters
fixed), the likelihood ratio statistic:

.. math::

   \Lambda = 2(\ell_B - \ell_A)

is approximately :math:`\chi^2`-distributed with :math:`k_B - k_A` degrees of
freedom, where :math:`k` is the number of free parameters.

.. admonition:: Probability Aside -- The chi-squared approximation

   The likelihood ratio test rests on **Wilks' theorem**: under the null
   hypothesis (the simpler model), the statistic :math:`\Lambda` converges
   to a :math:`\chi^2` distribution as the sample size grows.  Intuitively,
   near the MLE the log-likelihood is approximately quadratic (by a Taylor
   expansion), and a quadratic form in Gaussian variables is chi-squared.
   The degrees of freedom equal the number of additional parameters in the
   complex model, because each extra parameter "uses up" one dimension of
   the likelihood surface.

.. code-block:: python

   from scipy.stats import chi2

   def likelihood_ratio_test(ll_simple, ll_complex, df):
       """Likelihood ratio test for nested models.

       Parameters
       ----------
       ll_simple : float
           Log-likelihood of the simpler model.
       ll_complex : float
           Log-likelihood of the more complex model.
       df : int
           Difference in number of free parameters.

       Returns
       -------
       p_value : float
       """
       lr = 2 * (ll_complex - ll_simple)  # likelihood ratio statistic
       p_value = 1 - chi2.cdf(lr, df)      # p-value from chi-squared distribution
       return p_value

   # Example: constant size (0 params) vs two-epoch (2 params: nu, T)
   n = 30
   theta = 2000

   # "Data" from a two-epoch model
   data = two_epoch_model([5.0, 0.2], [n]) * theta
   np.random.seed(42)
   data_noisy = np.array([np.random.poisson(max(0, d)) for d in data])

   # Fit constant-size model (no free parameters)
   model_const = moments.Demographics1D.snm([n])
   theta_const = optimal_theta_scaling(data_noisy, model_const)
   ll_const = poisson_log_likelihood(data_noisy, model_const * theta_const)

   # Fit two-epoch model (2 free parameters)
   model_2epoch = two_epoch_model([nu_hat, T_hat], [n])
   theta_2epoch = optimal_theta_scaling(data_noisy, model_2epoch)
   ll_2epoch = poisson_log_likelihood(data_noisy, model_2epoch * theta_2epoch)

   p_val = likelihood_ratio_test(ll_const, ll_2epoch, df=2)
   print(f"Log-likelihood (constant): {ll_const:.2f}")
   print(f"Log-likelihood (2-epoch):  {ll_2epoch:.2f}")
   print(f"LRT p-value: {p_val:.2e}")
   print(f"Two-epoch significantly better: {p_val < 0.05}")

AIC
----

For non-nested models, use the **Akaike Information Criterion**:

.. math::

   \text{AIC} = 2k - 2\ell

where :math:`k` is the number of parameters and :math:`\ell` is the
log-likelihood. Lower AIC = better model (penalizing complexity).

.. admonition:: Calculus Aside -- Why AIC penalizes complexity

   The AIC is derived from an asymptotic approximation to the
   **Kullback--Leibler divergence** between the true distribution and the
   fitted model.  The :math:`-2\ell` term measures how well the model fits
   the data (smaller is better).  The :math:`2k` term corrects for the fact
   that a model with more parameters can always fit the *training* data
   better, even if the extra parameters are capturing noise rather than
   signal.  The AIC balances fit against parsimony: it prefers the simplest
   model that explains the data adequately.

Beyond the SFS-based likelihood, ``moments`` can also use demographic models
specified in the ``demes`` format.


Step 7: Inference with Demes
==============================

``moments`` integrates with the ``demes`` standard for specifying demographic
models in human-readable YAML format. This is the recommended approach for
complex models.

.. code-block:: python

   import moments
   import demes

   # Define a model using demes
   b = demes.Graph(description="Two-epoch expansion model")
   b.add_deme("ancestral", epochs=[
       demes.Epoch(start_size=10000, end_time=5000)
   ])
   b.add_deme("modern", ancestors=["ancestral"], epochs=[
       demes.Epoch(start_size=50000, end_time=0)
   ])
   graph = b.asdict_simplified()

   # Compute expected SFS directly from the demes graph
   fs = moments.Spectrum.from_demes(
       graph,
       samples={"modern": 20},
       theta=4 * 10000 * 1.5e-8 * 1e6  # 4*Ne*mu*L for 1 Mb
   )

   print(f"SFS shape: {fs.shape}")
   print(f"Segregating sites: {fs[1:-1].sum():.1f}")

.. admonition:: The demes advantage

   With ``demes``, you specify your model once in a standardized format that
   works across tools (``moments``, ``msprime``, ``dadi``, ``fwdpy11``).
   ``moments`` automatically translates the demes graph into the right sequence
   of integration steps, splits, and migration matrices.

Finally, a practical concern that can bias inference if ignored: ancestral
misidentification.


Step 8: Ancestral Misidentification
======================================

In practice, the outgroup used to polarize mutations isn't perfect. Some fraction
:math:`p_{\text{misid}}` of sites have the ancestral/derived labels swapped. This
flips :math:`j \to n - j` in the SFS for those sites.

The observed SFS is a mixture:

.. math::

   \text{SFS}_{\text{obs}}[j] = (1 - p_{\text{misid}}) \cdot \text{SFS}_{\text{true}}[j]
   + p_{\text{misid}} \cdot \text{SFS}_{\text{true}}[n - j]

``moments`` can jointly estimate :math:`p_{\text{misid}}` along with the
demographic parameters.

.. code-block:: python

   def apply_misidentification(sfs, p_misid):
       """Apply ancestral misidentification to an SFS.

       Parameters
       ----------
       sfs : ndarray of shape (n+1,)
       p_misid : float
           Fraction of sites with ancestral/derived labels swapped.

       Returns
       -------
       sfs_obs : ndarray of shape (n+1,)
       """
       n = len(sfs) - 1
       sfs_obs = np.zeros(n + 1)
       for j in range(n + 1):
           # Mix correctly polarized and flipped contributions
           sfs_obs[j] = (1 - p_misid) * sfs[j] + p_misid * sfs[n - j]
       return sfs_obs

   # Example: 2% misidentification
   n = 20
   sfs_true = expected_sfs_neutral(n, theta=1000)
   sfs_misid = apply_misidentification(sfs_true, 0.02)

   print("Effect of 2% ancestral misidentification:")
   print(f"{'j':>3} {'True':>10} {'Observed':>10} {'Diff%':>8}")
   for j in range(1, 6):
       diff_pct = (sfs_misid[j] - sfs_true[j]) / sfs_true[j] * 100
       print(f"{j:3d} {sfs_true[j]:10.2f} {sfs_misid[j]:10.2f} {diff_pct:+7.2f}%")
   for j in range(n-4, n):
       diff_pct = (sfs_misid[j] - sfs_true[j]) / sfs_true[j] * 100
       print(f"{j:3d} {sfs_true[j]:10.2f} {sfs_misid[j]:10.2f} {diff_pct:+7.2f}%")
   print("(High-frequency bins gain counts from misidentified singletons)")


Exercises
=========

.. admonition:: Exercise 1: Likelihood surface

   For a two-epoch model with :math:`n = 30` and simulated data, compute the
   log-likelihood on a :math:`50 \times 50` grid of :math:`(\nu, T)` values.
   Plot the surface as a contour map. Where is the maximum? Is the surface
   unimodal?

.. admonition:: Exercise 2: FIM vs GIM

   Simulate 100 independent SFS replicates under the same two-epoch model.
   For each, compute the FIM standard errors. Compare the distribution of
   :math:`(\hat{\nu} - \nu_{\text{true}}) / \text{SE}_{\text{FIM}}` to a
   standard normal. Does the FIM correctly predict the spread?

.. admonition:: Exercise 3: Model comparison

   Simulate data under a three-epoch model (bottleneck then expansion). Fit
   both a two-epoch and three-epoch model. Use the LRT to test whether the
   three-epoch model is significantly better.

.. admonition:: Exercise 4: Misidentification bias

   Generate data with 5% ancestral misidentification but fit without accounting
   for it. How biased are the demographic parameter estimates? Now fit with
   ``moments``' built-in misidentification correction. Does the bias disappear?

Solutions
=========

.. admonition:: Solution 1: Likelihood surface

   Compute the Poisson log-likelihood on a grid of :math:`(\nu, T)` values
   and display the result as a contour map.  The surface should be unimodal
   with the maximum near the true parameters.

   .. code-block:: python

      import numpy as np
      import moments

      # --- Generate data ---
      n = 30
      theta_true = 2000
      nu_true, T_true = 5.0, 0.2

      model_true = two_epoch_model([nu_true, T_true], [n])
      data = model_true * theta_true
      np.random.seed(42)
      data_noisy = np.zeros(n + 1)
      for j in range(1, n):
          data_noisy[j] = np.random.poisson(data[j])

      # --- Build a 50 x 50 grid ---
      nu_grid = np.linspace(0.5, 15.0, 50)
      T_grid = np.linspace(0.01, 0.8, 50)
      ll_surface = np.zeros((50, 50))

      for i, nu in enumerate(nu_grid):
          for j, T in enumerate(T_grid):
              model = two_epoch_model([nu, T], [n])
              theta_opt = optimal_theta_scaling(data_noisy, model)
              model_scaled = model * theta_opt
              ll_surface[i, j] = poisson_log_likelihood(data_noisy, model_scaled)

      # --- Find the maximum ---
      idx = np.unravel_index(np.argmax(ll_surface), ll_surface.shape)
      nu_best = nu_grid[idx[0]]
      T_best = T_grid[idx[1]]

      print(f"Grid maximum at nu={nu_best:.2f}, T={T_best:.3f}")
      print(f"True parameters: nu={nu_true}, T={T_true}")
      print(f"Max log-likelihood: {ll_surface[idx]:.2f}")

      # --- Contour plot (if matplotlib available) ---
      try:
          import matplotlib.pyplot as plt
          fig, ax = plt.subplots(figsize=(7, 5))
          NU, TT = np.meshgrid(nu_grid, T_grid, indexing='ij')
          levels = np.linspace(ll_surface.max() - 20, ll_surface.max(), 15)
          cs = ax.contourf(NU, TT, ll_surface, levels=levels, cmap='viridis')
          ax.plot(nu_true, T_true, 'r*', markersize=14, label='True')
          ax.plot(nu_best, T_best, 'wx', markersize=10, label='Grid max')
          ax.set_xlabel(r'$\nu$ (expansion factor)')
          ax.set_ylabel(r'$T$ (time since expansion)')
          ax.set_title('Log-likelihood surface')
          ax.legend()
          plt.colorbar(cs, label='Log-likelihood')
          plt.tight_layout()
          plt.savefig('likelihood_surface.png', dpi=150)
          plt.show()
      except ImportError:
          print("(matplotlib not available -- skipping plot)")

   The surface should be unimodal.  A ridge running diagonally indicates a
   correlation between :math:`\nu` and :math:`T`: a larger expansion can be
   partially compensated by a more recent onset.  This correlation is captured
   by the off-diagonal elements of the Fisher Information Matrix.

.. admonition:: Solution 2: FIM vs GIM

   Simulate 100 independent SFS replicates, fit each one, compute
   FIM-based z-scores, and check whether they follow a standard normal.

   .. code-block:: python

      import numpy as np
      import moments
      from scipy.optimize import minimize

      n = 30
      nu_true, T_true = 5.0, 0.2
      theta_true = 2000

      model_true = two_epoch_model([nu_true, T_true], [n])
      data_expected = model_true * theta_true

      z_scores_nu = []
      z_scores_T = []

      for rep in range(100):
          np.random.seed(rep)
          # Simulate Poisson data
          data_rep = np.zeros(n + 1)
          for j in range(1, n):
              data_rep[j] = np.random.poisson(data_expected[j])

          # Fit the two-epoch model
          log_p0 = np.log([3.0, 0.3])
          result = minimize(
              objective, log_p0, args=(data_rep, [n]),
              method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-4}
          )
          nu_hat, T_hat = np.exp(result.x)
          params_hat = np.array([nu_hat, T_hat])

          # Compute FIM and standard errors
          FIM = fisher_information_numerical(
              params_hat, data_rep, two_epoch_model, [n]
          )
          if np.linalg.det(FIM) > 0:
              cov = np.linalg.inv(FIM)
              se = np.sqrt(np.diag(cov))
              z_scores_nu.append((nu_hat - nu_true) / se[0])
              z_scores_T.append((T_hat - T_true) / se[1])

      z_nu = np.array(z_scores_nu)
      z_T = np.array(z_scores_T)

      print(f"z-scores for nu: mean={z_nu.mean():.3f}, std={z_nu.std():.3f}")
      print(f"z-scores for T:  mean={z_T.mean():.3f}, std={z_T.std():.3f}")
      print(f"Expected for N(0,1): mean=0.000, std=1.000")

   If the FIM correctly predicts the spread, the z-score standard deviation
   should be close to 1.0.  With Poisson-sampled data (truly independent
   sites), the FIM is the correct information measure, so the z-scores
   should indeed follow a standard normal.  With real genomic data, linkage
   would inflate the true variance beyond what the FIM predicts, making
   the z-score distribution wider than :math:`N(0,1)` -- this is precisely
   why the GIM correction is needed for real data.

.. admonition:: Solution 3: Model comparison

   Simulate data under a three-epoch model (bottleneck then expansion),
   fit both a two-epoch and three-epoch model, and use the LRT.

   .. code-block:: python

      import numpy as np
      import moments
      from scipy.optimize import minimize
      from scipy.stats import chi2

      n = 30
      theta_true = 3000

      # --- Three-epoch model: ancestral -> bottleneck -> expansion ---
      def three_epoch_model(params, ns):
          nu_B, nu_E, T_B, T_E = params
          fs = moments.Demographics1D.snm(ns)
          fs.integrate([nu_B], T_B)   # bottleneck
          fs.integrate([nu_E], T_E)   # expansion
          return fs

      # True parameters: bottleneck to 0.1x, then expansion to 5x
      params_true_3 = [0.1, 5.0, 0.05, 0.2]
      model_true_3 = three_epoch_model(params_true_3, [n])
      data = model_true_3 * theta_true

      np.random.seed(42)
      data_noisy = np.zeros(n + 1)
      for j in range(1, n):
          data_noisy[j] = np.random.poisson(data[j])

      # --- Fit the two-epoch model (2 free parameters) ---
      def obj_2epoch(log_p, data_sfs, ns):
          nu, T = np.exp(log_p)
          model = two_epoch_model([nu, T], ns)
          theta_opt = optimal_theta_scaling(data_sfs, model)
          return -poisson_log_likelihood(data_sfs, model * theta_opt)

      res_2 = minimize(obj_2epoch, np.log([2.0, 0.3]), args=(data_noisy, [n]),
                       method='Nelder-Mead', options={'maxiter': 2000})
      nu2, T2 = np.exp(res_2.x)
      model_2 = two_epoch_model([nu2, T2], [n])
      theta_2 = optimal_theta_scaling(data_noisy, model_2)
      ll_2 = poisson_log_likelihood(data_noisy, model_2 * theta_2)

      # --- Fit the three-epoch model (4 free parameters) ---
      def obj_3epoch(log_p, data_sfs, ns):
          params = np.exp(log_p)
          model = three_epoch_model(params, ns)
          theta_opt = optimal_theta_scaling(data_sfs, model)
          return -poisson_log_likelihood(data_sfs, model * theta_opt)

      res_3 = minimize(obj_3epoch, np.log([0.5, 3.0, 0.1, 0.1]),
                       args=(data_noisy, [n]),
                       method='Nelder-Mead', options={'maxiter': 5000})
      p3 = np.exp(res_3.x)
      model_3 = three_epoch_model(p3, [n])
      theta_3 = optimal_theta_scaling(data_noisy, model_3)
      ll_3 = poisson_log_likelihood(data_noisy, model_3 * theta_3)

      # --- Likelihood ratio test (df = 4 - 2 = 2) ---
      lr_stat = 2 * (ll_3 - ll_2)
      p_value = 1 - chi2.cdf(lr_stat, df=2)

      print(f"Two-epoch:   ll = {ll_2:.2f}  (nu={nu2:.3f}, T={T2:.4f})")
      print(f"Three-epoch: ll = {ll_3:.2f}  (nuB={p3[0]:.3f}, nuE={p3[1]:.3f}, "
            f"TB={p3[2]:.4f}, TE={p3[3]:.4f})")
      print(f"LR statistic: {lr_stat:.2f}")
      print(f"p-value: {p_value:.2e}")
      print(f"Three-epoch significantly better (p < 0.05): {p_value < 0.05}")

   Since the data were generated under a three-epoch model, the three-epoch
   fit should have a significantly higher likelihood.  The LRT with 2 degrees
   of freedom should reject the simpler two-epoch model at the 0.05 level.

.. admonition:: Solution 4: Misidentification bias

   Generate data with 5% ancestral misidentification, fit without correction,
   then fit with the ``apply_misidentification`` correction and compare.

   .. code-block:: python

      import numpy as np
      import moments
      from scipy.optimize import minimize

      n = 30
      nu_true, T_true = 5.0, 0.2
      theta_true = 2000
      p_misid_true = 0.05

      # Generate true SFS and apply misidentification
      model_true = two_epoch_model([nu_true, T_true], [n])
      sfs_true = model_true * theta_true
      sfs_misid = apply_misidentification(np.array(sfs_true), p_misid_true)

      np.random.seed(42)
      data_obs = np.zeros(n + 1)
      for j in range(1, n):
          data_obs[j] = np.random.poisson(sfs_misid[j])

      # --- Fit WITHOUT misidentification correction ---
      def obj_no_correction(log_p, data_sfs, ns):
          nu, T = np.exp(log_p)
          model = two_epoch_model([nu, T], ns)
          theta_opt = optimal_theta_scaling(data_sfs, model)
          return -poisson_log_likelihood(data_sfs, model * theta_opt)

      res_no = minimize(obj_no_correction, np.log([3.0, 0.3]),
                        args=(data_obs, [n]),
                        method='Nelder-Mead', options={'maxiter': 2000})
      nu_no, T_no = np.exp(res_no.x)

      # --- Fit WITH misidentification correction ---
      def obj_with_correction(log_p, data_sfs, ns):
          # params: [log(nu), log(T), logit(p_misid)]
          nu, T = np.exp(log_p[:2])
          p_misid = 1.0 / (1.0 + np.exp(-log_p[2]))  # sigmoid for (0, 1)
          model = two_epoch_model([nu, T], ns)
          theta_opt = optimal_theta_scaling(data_sfs, model)
          model_scaled = np.array(model * theta_opt)
          model_corrected = apply_misidentification(model_scaled, p_misid)
          return -poisson_log_likelihood(data_sfs, model_corrected)

      # Initial guess: logit(0.02) for p_misid
      logit_p0 = np.log(0.02 / (1 - 0.02))
      res_with = minimize(obj_with_correction,
                          np.array([np.log(3.0), np.log(0.3), logit_p0]),
                          args=(data_obs, [n]),
                          method='Nelder-Mead', options={'maxiter': 3000})
      nu_with, T_with = np.exp(res_with.x[:2])
      p_misid_hat = 1.0 / (1.0 + np.exp(-res_with.x[2]))

      print("True parameters:    nu=5.000, T=0.2000, p_misid=0.050")
      print(f"Without correction: nu={nu_no:.3f}, T={T_no:.4f}")
      print(f"  Bias in nu: {(nu_no - nu_true)/nu_true*100:+.1f}%")
      print(f"  Bias in T:  {(T_no - T_true)/T_true*100:+.1f}%")
      print(f"With correction:    nu={nu_with:.3f}, T={T_with:.4f}, "
            f"p_misid={p_misid_hat:.3f}")
      print(f"  Bias in nu: {(nu_with - nu_true)/nu_true*100:+.1f}%")
      print(f"  Bias in T:  {(T_with - T_true)/T_true*100:+.1f}%")

   Without correction, the 5% misidentification inflates the high-frequency
   bins (singletons are "flipped" to appear as :math:`n-1` counts), biasing
   the inferred demography -- typically making the population appear less
   expanded than it truly is.  With the correction, the optimizer jointly
   recovers :math:`\nu`, :math:`T`, and :math:`p_{\text{misid}}`, removing
   the bias.

Next: :ref:`linkage_disequilibrium` -- a second source of information about
demography, captured by two-locus statistics.  Where the SFS is our primary
dial, LD is a **second pendulum** that constrains parameters the SFS alone
cannot resolve.
