.. _tsdate_variational_gamma:

=============================================
Variational Gamma (Expectation Propagation)
=============================================

   *The master gear: approximate every node's age as a gamma distribution, then
   refine by passing messages until the whole mechanism converges.*

The inside-outside method (:ref:`tsdate_inside_outside`) passes messages
through the gear train using a discrete grid. It works, but the grid imposes
limits on resolution and speed. The variational gamma method is tsdate's
default and most accurate algorithm. Instead of discretizing time into a grid,
it approximates each node's posterior age as a **gamma distribution** and
refines it iteratively using **Expectation Propagation** (EP) -- a
message-passing algorithm from machine learning (Minka, 2001).

In the watch metaphor, this is still **messages flowing through the gear
train**, but now each message is a gamma distribution rather than a probability
vector. The gear train is the same; only the language of the messages has
changed -- from a list of numbers (grid probabilities) to two numbers
(:math:`\alpha`, :math:`\beta`) that encode a continuous belief about each
node's age.

This chapter builds EP from scratch, one piece at a time. By the end you will
understand cavity distributions, moment matching, and damping -- the three
pillars of EP -- and see how they fit together to date a tree sequence.

.. admonition:: Biology Aside -- Why dating matters

   Assigning dates to ancestral nodes in a genealogy answers some of the most
   fundamental questions in evolutionary biology: *When did the most recent
   common ancestor of all humans live? When did a specific population split
   occur? How old is a particular beneficial mutation?* The tree sequence
   from ``tsinfer`` gives us the topology (who is related to whom), but
   the branch lengths -- the time spans between ancestor and descendant --
   must be estimated from the density of mutations along each branch. More
   mutations imply more time. The variational gamma method performs this
   estimation for every node simultaneously, propagating information through
   the entire genealogy to produce the most consistent set of dates.

.. admonition:: Prerequisites

   This chapter builds on three earlier ones. The coalescent prior
   (:ref:`tsdate_coalescent_prior`) provides the initial gamma beliefs; the
   mutation likelihood (:ref:`tsdate_mutation_likelihood`) defines the Poisson
   factors that link parent and child nodes; and the inside-outside chapter
   (:ref:`tsdate_inside_outside`) introduces the idea of two-pass message
   passing. If any of those concepts feel shaky, revisit the relevant chapter
   before continuing.


Why Move Beyond Inside-Outside?
==================================

The inside-outside method (previous chapter) has three practical limitations:

1. The **time grid** limits resolution: you can't distinguish ages that fall
   in the same cell.
2. The cost is **quadratic** in grid size per edge: :math:`O(K^2 \cdot E)`.
3. The grid boundaries are somewhat arbitrary.

The variational gamma method solves all three:

- **Continuous time**: no grid, no resolution limit.
- **Two parameters per node**: :math:`(\alpha, \beta)` for the gamma shape and
  rate, so the cost is :math:`O(E)` per iteration.
- **Natural parameterization**: the gamma family captures the right range of
  posterior shapes for coalescence times.


The Big Picture
=================

Here's the algorithm in one paragraph:

   Represent each node's posterior age as :math:`\text{Gamma}(\alpha_u, \beta_u)`.
   For each edge :math:`e=(u,v)`, compute a "message" that says how the Poisson
   mutation likelihood on :math:`e` updates the gamma beliefs for :math:`u` and
   :math:`v`. Apply these messages by **moment matching**: compute the exact
   moments of the updated distribution, then find the gamma that matches those
   moments. Iterate over all edges until convergence.

And here it is as a diagram:

.. code-block:: text

   Initialize: q(t_u) = Gamma(alpha_u, beta_u) for each node u
                                |
                +---------------+---------------+
                |                               |
                v                               |
   For each edge e = (u, v):                    |
     1. Remove old message from q(t_u), q(t_v)  |
     2. Compute exact moments of:               |
        q(t_u) * q(t_v) * Poisson(m_e|...)      |
     3. Moment-match to new gammas               |
     4. Update q(t_u), q(t_v)                   |
                |                               |
                +-----> Converged? ----No-------+
                            |
                           Yes
                            |
                            v
                  Return posterior means

.. admonition:: Probability Aside -- What is variational inference?

   Variational inference is a family of methods for approximating intractable
   probability distributions. The idea: choose a simple family of
   distributions :math:`\mathcal{Q}` (here, products of gamma distributions)
   and find the member :math:`q^* \in \mathcal{Q}` that is "closest" to the
   true posterior :math:`p`, measured by KL divergence. The name "variational"
   comes from the calculus of variations, because we are optimizing over
   functions (distributions) rather than finite-dimensional parameters. EP is
   one specific variational method; standard variational Bayes (mean-field) is
   another. They differ in which KL divergence they minimize and how they
   process factors.


The Natural Parameterization
==============================

A gamma distribution :math:`\text{Gamma}(\alpha, \beta)` has density:

.. math::

   p(t) = \frac{\beta^\alpha}{\Gamma(\alpha)} t^{\alpha - 1} e^{-\beta t},
   \quad t > 0

In the **natural parameter** (exponential family) form, this is:

.. math::

   p(t) \propto \exp\bigl((\alpha - 1) \log t - \beta t\bigr)

The natural parameters are :math:`\eta_1 = \alpha - 1` and :math:`\eta_2 = -\beta`.
The sufficient statistics are :math:`\log t` and :math:`t`.

.. admonition:: Biology Aside -- What :math:`\alpha` and :math:`\beta` mean for node ages

   For each ancestor in the genealogy, the gamma distribution
   :math:`\text{Gamma}(\alpha, \beta)` encodes our belief about when that
   ancestor lived. The **mean** :math:`\alpha/\beta` is our best estimate
   of the ancestor's age (in generations or coalescent time units). The
   **variance** :math:`\alpha/\beta^2` measures how uncertain we are. A
   node deep in the tree (many mutations on incident edges, many descendant
   samples) will have a tight gamma with large :math:`\alpha` -- we are
   confident about its age. A node with few mutations and few descendants
   will have a diffuse gamma -- we know little about when it lived. The EP
   algorithm refines these beliefs by passing information along edges,
   iteratively sharpening each node's gamma.

**Why natural parameters?** Because products of gamma-like terms correspond to
*additions* of natural parameters. Let us show this step by step. Start with two
gamma-shaped factors:

.. math::

   f_1(t) \propto t^{\alpha_1 - 1} e^{-\beta_1 t}, \quad
   f_2(t) \propto t^{\alpha_2 - 1} e^{-\beta_2 t}

Multiply them together, using the rules :math:`t^a \cdot t^b = t^{a+b}` and
:math:`e^{-c_1 t} \cdot e^{-c_2 t} = e^{-(c_1+c_2)t}`:

.. math::

   f_1(t) \cdot f_2(t) &\propto t^{\alpha_1 - 1} \cdot t^{\alpha_2 - 1}
   \cdot e^{-\beta_1 t} \cdot e^{-\beta_2 t} \\
   &= t^{(\alpha_1 - 1) + (\alpha_2 - 1)} \cdot e^{-(\beta_1 + \beta_2) t} \\
   &= t^{(\alpha_1 + \alpha_2 - 2)} \cdot e^{-(\beta_1 + \beta_2) t} \\
   &= t^{(\alpha_1 + \alpha_2 - 1) - 1} \cdot e^{-(\beta_1 + \beta_2) t}

This is the kernel of :math:`\text{Gamma}(\alpha_1 + \alpha_2 - 1, \beta_1 + \beta_2)`.
In natural parameters :math:`(\eta_1, \eta_2) = (\alpha - 1, -\beta)`, the
product corresponds to elementwise addition:

.. math::

   (\eta_1^{(1)} + \eta_1^{(2)}, \; \eta_2^{(1)} + \eta_2^{(2)})
   = (\alpha_1 - 1 + \alpha_2 - 1, \; -\beta_1 - \beta_2)

which gives natural parameters for the product
:math:`\text{Gamma}(\alpha_1 + \alpha_2 - 1, \beta_1 + \beta_2)`.

.. code-block:: python

   # Verify the product rule numerically
   import numpy as np
   from scipy.stats import gamma as gamma_dist

   a1, b1 = 3.0, 2.0
   a2, b2 = 2.0, 1.5

   x = np.linspace(0.01, 5.0, 1000)

   # Product of two gamma PDFs (unnormalized)
   f1 = gamma_dist.pdf(x, a=a1, scale=1/b1)
   f2 = gamma_dist.pdf(x, a=a2, scale=1/b2)
   product = f1 * f2

   # The result should be proportional to Gamma(a1+a2-1, b1+b2)
   a_new, b_new = a1 + a2 - 1, b1 + b2
   f_new = gamma_dist.pdf(x, a=a_new, scale=1/b_new)

   # Check proportionality: ratio should be constant
   ratio = product / f_new
   ratio = ratio[f_new > 1e-10]  # avoid division by near-zero
   print(f"Product is Gamma({a_new}, {b_new})")
   print(f"Ratio min={ratio.min():.6f}, max={ratio.max():.6f} (should be constant)")

This addition rule is the foundation of EP updates. In the gear train, each
factor (edge likelihood, coalescent prior) contributes a "torque" in natural
parameter space, and the total torque on a node is simply the sum.

.. admonition:: Calculus Aside -- Exponential families

   A distribution belongs to the *exponential family* if its density can be
   written as :math:`p(x|\eta) = h(x) \exp(\eta \cdot T(x) - A(\eta))`,
   where :math:`\eta` is the natural parameter vector, :math:`T(x)` is the
   sufficient statistic vector, and :math:`A(\eta)` is the log-normalizer.
   For the gamma: :math:`T(t) = (\log t, t)`,
   :math:`\eta = (\alpha - 1, -\beta)`, and
   :math:`A(\eta) = \log\Gamma(\alpha) - \alpha \log\beta`. The key property
   is that products of factors in the same exponential family yield a member
   of the same family (with summed natural parameters). This is why gamma
   posteriors can be updated by simple addition.

.. code-block:: python

   import numpy as np
   from scipy.special import gammaln, digamma, polygamma

   class GammaDistribution:
       """A gamma distribution in natural parameterization.

       Natural parameters: eta1 = alpha - 1, eta2 = -beta
       Standard parameters: alpha (shape), beta (rate)
       """
       def __init__(self, alpha=1.0, beta=1.0):
           self.alpha = alpha   # shape parameter
           self.beta = beta     # rate parameter

       @property
       def eta1(self):
           """First natural parameter: alpha - 1."""
           return self.alpha - 1

       @property
       def eta2(self):
           """Second natural parameter: -beta."""
           return -self.beta

       @property
       def mean(self):
           """E[t] = alpha / beta."""
           return self.alpha / self.beta

       @property
       def variance(self):
           """Var(t) = alpha / beta^2."""
           return self.alpha / self.beta**2

       @property
       def log_mean(self):
           """E[log t] = digamma(alpha) - log(beta)"""
           return digamma(self.alpha) - np.log(self.beta)

       def multiply(self, other):
           """Multiply two gamma factors (add natural parameters).

           In natural parameter space: (eta1, eta2) + (eta1', eta2')
           In standard parameters: alpha_new = alpha + alpha' - 1,
                                   beta_new = beta + beta'
           """
           new_alpha = self.alpha + other.alpha - 1
           new_beta = self.beta + other.beta
           return GammaDistribution(new_alpha, new_beta)

       def divide(self, other):
           """Divide by a gamma factor (subtract natural parameters).

           This is the inverse of multiply: removing a factor's contribution.
           """
           new_alpha = self.alpha - other.alpha + 1
           new_beta = self.beta - other.beta
           return GammaDistribution(new_alpha, new_beta)

       @classmethod
       def from_moments(cls, mean, variance):
           """Create from mean and variance via moment matching.

           Uses the standard method-of-moments estimator:
           beta = mean / variance, alpha = mean * beta
           """
           beta = mean / variance
           alpha = mean * beta
           return cls(alpha, beta)


The EP Algorithm Step by Step
===============================

EP maintains the following state:

- **Posterior approximation** :math:`q(t_u) = \text{Gamma}(\alpha_u, \beta_u)`
  for each node :math:`u`
- **Edge factors** :math:`f_e^{\to u}` and :math:`f_e^{\to v}` for each edge
  :math:`e = (u, v)`: gamma-shaped "messages" from the edge likelihood

The posterior for node :math:`u` is the product of its prior and all incoming
edge messages:

.. math::

   q(t_u) \propto \text{prior}(t_u) \cdot \prod_{e \ni u} f_e^{\to u}(t_u)

In natural parameters, this is a sum:

.. math::

   (\alpha_u - 1, -\beta_u) = (\alpha_{\text{prior}} - 1, -\beta_{\text{prior}})
   + \sum_{e \ni u} (\alpha_{f_e} - 1, -\beta_{f_e})

.. admonition:: Probability Aside -- Expectation Propagation vs. Variational Bayes

   EP and variational Bayes (VB) are both methods for approximating a posterior
   :math:`p` with a simpler distribution :math:`q`. The difference lies in
   *which* KL divergence they minimize:

   - **VB** minimizes :math:`\text{KL}(q \| p)` (the "exclusive" or
     "reverse" KL). This tends to make :math:`q` concentrate on a single mode
     and underestimate uncertainty.
   - **EP** minimizes :math:`\text{KL}(p \| q)` (the "inclusive" or
     "forward" KL). This forces :math:`q` to *cover* all of :math:`p`,
     typically yielding better-calibrated uncertainty.

   For tsdate, EP's inclusive KL is important: we want the gamma approximation
   to reflect the full spread of each node's age uncertainty, not just the
   mode.

Initialization
----------------

1. Set each node's posterior to its coalescent prior:
   :math:`q(t_u) = \text{Gamma}(\alpha_{\text{prior}}, \beta_{\text{prior}})`

2. Set all edge factors to "uninformative":
   :math:`f_e^{\to u} = \text{Gamma}(1, 0)` (i.e., natural parameters :math:`(0, 0)`)

.. code-block:: python

   def initialize_ep(ts, prior_grid):
       """Initialize EP state.

       Parameters
       ----------
       ts : tskit.TreeSequence
       prior_grid : dict
           {node_id: (alpha, beta)} from the coalescent prior.

       Returns
       -------
       posteriors : dict
           {node_id: GammaDistribution}
       edge_factors : dict
           {(edge_id, direction): GammaDistribution}
           direction is 'rootward' (to parent) or 'leafward' (to child)
       """
       posteriors = {}
       for node in ts.nodes():
           if node.id in prior_grid:
               alpha, beta = prior_grid[node.id]
               # Start with the coalescent prior as the initial belief
               posteriors[node.id] = GammaDistribution(alpha, beta)
           else:
               # Sample nodes: fixed at time 0 (very tight distribution)
               posteriors[node.id] = GammaDistribution(1.0, 1e10)

       # Initialize all edge factors to uninformative Gamma(1, 0)
       # In natural parameters this is (0, 0) -- contributes nothing
       edge_factors = {}
       for edge in ts.edges():
           edge_factors[(edge.id, 'rootward')] = GammaDistribution(1.0, 0.0)
           edge_factors[(edge.id, 'leafward')] = GammaDistribution(1.0, 0.0)

       return posteriors, edge_factors


The EP Update for One Edge
----------------------------

This is the heart of the algorithm. For edge :math:`e = (u, v)` with
:math:`m_e` mutations and span-weighted rate :math:`\lambda_e`:

.. admonition:: Biology Aside -- What an EP update does, biologically

   Each edge in the tree sequence connects a parent (ancestor) to a child
   (descendant). The edge carries mutations whose count constrains the time
   difference between parent and child. The EP update for one edge asks:
   *given what we currently believe about the parent's age and the child's
   age, and given the number of mutations on this edge, how should we revise
   our beliefs?* If many mutations sit on a short edge, the parent must be
   much older than the child. If no mutations sit on a long edge, parent and
   child are probably close in time. The four steps below formalize this
   intuition as a sequence of mathematical operations.

**Step 1: Compute the "cavity" distributions.**

Remove the current edge's messages from the parent and child posteriors:

.. math::

   q_{\setminus e}(t_u) = \frac{q(t_u)}{f_e^{\to u}(t_u)}, \quad
   q_{\setminus e}(t_v) = \frac{q(t_v)}{f_e^{\to v}(t_v)}

In natural parameters, this is subtraction.

**Intuition**: The cavity is "what we'd believe about this node if we forgot
everything this particular edge told us." It's the belief from all *other*
sources of information. In the gear train, it is like temporarily disengaging
one spring to see where the gear would sit under the tension of all the other
springs.

**Step 2: Compute the "tilted" distribution.**

The tilted distribution is the cavity times the *true* edge likelihood:

.. math::

   \tilde{p}(t_u, t_v) = q_{\setminus e}(t_u) \cdot q_{\setminus e}(t_v)
   \cdot \frac{(\lambda_e (t_u - t_v))^{m_e}}{m_e!} e^{-\lambda_e(t_u - t_v)}
   \cdot \mathbb{1}[t_u > t_v]

This is the *exact* posterior we'd get for these two nodes if this were the only
edge in the graph. It's generally **not** a product of two gammas -- the
:math:`(t_u - t_v)^{m_e}` term couples the variables.

**Step 3: Moment matching.**

Compute the marginal moments of the tilted distribution:

.. math::

   \tilde{\mu}_u = \mathbb{E}_{\tilde{p}}[t_u], \quad
   \tilde{\sigma}^2_u = \text{Var}_{\tilde{p}}(t_u)

.. math::

   \tilde{\mu}_v = \mathbb{E}_{\tilde{p}}[t_v], \quad
   \tilde{\sigma}^2_v = \text{Var}_{\tilde{p}}(t_v)

Then find the gamma distributions that match these moments:

.. math::

   q_{\text{new}}(t_u) = \text{Gamma}\left(\frac{\tilde{\mu}_u^2}{\tilde{\sigma}^2_u},
   \frac{\tilde{\mu}_u}{\tilde{\sigma}^2_u}\right)

.. admonition:: Probability Aside -- What is moment matching?

   Moment matching is the simplest way to project a complex distribution onto
   a simpler family. Given a distribution :math:`\tilde{p}` (the tilted
   distribution, which is not gamma), we compute its mean and variance, then
   find the unique Gamma(:math:`\alpha`, :math:`\beta`) with the same mean
   and variance. This is the gamma that is "closest" to :math:`\tilde{p}` in
   the sense of matching first and second moments. It is the same
   method-of-moments idea used for the coalescent prior
   (:ref:`tsdate_coalescent_prior`), but here applied at every EP iteration
   to every edge.

**Step 4: Update the edge factors.**

The new message from edge :math:`e` to node :math:`u` is:

.. math::

   f_e^{\to u, \text{new}} = \frac{q_{\text{new}}(t_u)}{q_{\setminus e}(t_u)}

In natural parameters: subtract the cavity from the new posterior.

.. code-block:: python

   def ep_update_edge(edge, posteriors, edge_factors, m_e, lambda_e, damping=0.5):
       """Perform one EP update for a single edge.

       Parameters
       ----------
       edge : tskit.Edge
       posteriors : dict of GammaDistribution
       edge_factors : dict of GammaDistribution
       m_e : int
           Mutations on this edge.
       lambda_e : float
           Span-weighted mutation rate.
       damping : float
           Damping factor in [0, 1]. 1 = no damping, 0.5 = half step.

       Returns
       -------
       posteriors, edge_factors : updated in place.
       """
       u, v = edge.parent, edge.child

       # Step 1: Compute cavities (remove this edge's old message)
       cavity_u = posteriors[u].divide(edge_factors[(edge.id, 'rootward')])
       cavity_v = posteriors[v].divide(edge_factors[(edge.id, 'leafward')])

       # Step 2 & 3: Compute tilted moments and moment-match
       # This is the expensive part: we need E[t_u], Var(t_u), E[t_v], Var(t_v)
       # under the tilted distribution
       moments = compute_tilted_moments(cavity_u, cavity_v, m_e, lambda_e)

       if moments is None:
           return  # numerical failure, skip this edge

       mu_u, var_u, mu_v, var_v = moments

       # Moment-match to gammas (find the gamma with these moments)
       new_post_u = GammaDistribution.from_moments(mu_u, var_u)
       new_post_v = GammaDistribution.from_moments(mu_v, var_v)

       # Step 4: Compute new edge factors = new_posterior / cavity
       new_factor_u = new_post_u.divide(cavity_u)
       new_factor_v = new_post_v.divide(cavity_v)

       # Apply damping: interpolate between old and new factors
       # in natural parameter space to prevent oscillation
       old_factor_u = edge_factors[(edge.id, 'rootward')]
       old_factor_v = edge_factors[(edge.id, 'leafward')]

       damped_u = GammaDistribution(
           old_factor_u.alpha + damping * (new_factor_u.alpha - old_factor_u.alpha),
           old_factor_u.beta + damping * (new_factor_u.beta - old_factor_u.beta)
       )
       damped_v = GammaDistribution(
           old_factor_v.alpha + damping * (new_factor_v.alpha - old_factor_v.alpha),
           old_factor_v.beta + damping * (new_factor_v.beta - old_factor_v.beta)
       )

       # Update the stored edge factors
       edge_factors[(edge.id, 'rootward')] = damped_u
       edge_factors[(edge.id, 'leafward')] = damped_v

       # Recompute posteriors: cavity * new_factor
       posteriors[u] = cavity_u.multiply(damped_u)
       posteriors[v] = cavity_v.multiply(damped_v)


Computing the Tilted Moments
===============================

The hardest part of EP is computing the moments of the tilted distribution.
For the Poisson-gamma case, this involves integrals of the form:

.. math::

   \mathbb{E}_{\tilde{p}}[t_u] = \frac{
   \int_0^\infty \int_0^{t_u} t_u \cdot q_{\setminus e}(t_u) \cdot q_{\setminus e}(t_v) \cdot
   (\lambda_e(t_u - t_v))^{m_e} e^{-\lambda_e(t_u-t_v)} \, dt_v \, dt_u
   }{
   \int_0^\infty \int_0^{t_u} q_{\setminus e}(t_u) \cdot q_{\setminus e}(t_v) \cdot
   (\lambda_e(t_u - t_v))^{m_e} e^{-\lambda_e(t_u-t_v)} \, dt_v \, dt_u
   }

.. admonition:: Plain-language summary -- Why these integrals are hard

   The difficulty arises because the parent must be older than the child
   (:math:`t_u > t_v`), and the number of mutations depends on the time
   *difference* :math:`t_u - t_v`. This couples the two variables: you
   cannot estimate the parent's age independently of the child's. The
   integral averages over all possible (parent age, child age) combinations
   that are consistent with both the mutation data on this edge *and* the
   information from all other edges (encoded in the cavity). Computing this
   average exactly would require evaluating a two-dimensional integral for
   each of the millions of edges in a tree sequence -- which is why
   approximations are essential.

These integrals don't have closed forms in general. tsdate evaluates them using
a combination of:

1. **Laplace approximation**: Find the mode of the tilted distribution and
   approximate with a Gaussian around it.

2. **Numerical quadrature**: For edges with very few mutations, use direct
   numerical integration.

3. **Special cases**: When :math:`m_e = 0`, the likelihood simplifies to a pure
   exponential, and some integrals become tractable.

The Laplace Approach
----------------------

The Laplace approximation finds the mode :math:`(\hat{t}_u, \hat{t}_v)` of the
tilted distribution by solving:

.. math::

   \frac{\partial}{\partial t_u} \log \tilde{p}(t_u, t_v) = 0, \quad
   \frac{\partial}{\partial t_v} \log \tilde{p}(t_u, t_v) = 0

Then approximates the tilted distribution as a bivariate Gaussian centered at
the mode, with covariance given by the negative inverse Hessian:

.. math::

   \tilde{p}(t_u, t_v) \approx \mathcal{N}\left(
   \begin{pmatrix} \hat{t}_u \\ \hat{t}_v \end{pmatrix},
   \mathbf{H}^{-1}
   \right)

where :math:`\mathbf{H}` is the Hessian of :math:`-\log \tilde{p}` at the mode.

.. admonition:: Calculus Aside -- The Laplace approximation

   The Laplace approximation is a technique for approximating integrals of
   the form :math:`\int e^{f(x)} dx`. The idea: expand :math:`f(x)` in a
   Taylor series around its maximum :math:`\hat{x}`:

   .. math::

      f(x) \approx f(\hat{x}) + \frac{1}{2}(x - \hat{x})^T H (x - \hat{x})

   where :math:`H = \nabla^2 f(\hat{x})` is the Hessian (matrix of second
   derivatives). The integral then becomes a Gaussian integral with known
   closed form. In our case :math:`f = \log \tilde{p}`, so the Laplace
   approximation replaces the tilted distribution with a Gaussian centered at
   its mode. The quality of this approximation improves as the tilted
   distribution becomes more peaked (more data), which is why it works well
   for edges with many mutations.

.. code-block:: python

   from scipy.optimize import minimize

   def compute_tilted_moments(cavity_u, cavity_v, m_e, lambda_e):
       """Compute moments of the tilted distribution via Laplace approximation.

       Parameters
       ----------
       cavity_u, cavity_v : GammaDistribution
           Cavity distributions for parent and child.
       m_e : int
           Mutation count.
       lambda_e : float
           Span-weighted mutation rate.

       Returns
       -------
       mu_u, var_u, mu_v, var_v : float
           Moments of the tilted marginals, or None if numerical failure.
       """
       def neg_log_tilted(params):
           """Negative log of the tilted distribution (to be minimized)."""
           t_u, t_v = params
           if t_u <= t_v or t_u <= 0 or t_v < 0:
               return 1e20  # constraint violation

           delta = t_u - t_v

           # Log cavity contributions (gamma log-pdf, unnormalized)
           log_cavity_u = (cavity_u.alpha - 1) * np.log(t_u) - cavity_u.beta * t_u
           log_cavity_v = (cavity_v.alpha - 1) * np.log(max(t_v, 1e-20)) - cavity_v.beta * t_v

           # Log Poisson likelihood: m*log(lambda*delta) - lambda*delta
           log_lik = m_e * np.log(lambda_e * delta) - lambda_e * delta

           return -(log_cavity_u + log_cavity_v + log_lik)

       # Initial guess: cavity means
       t_u_init = max(cavity_u.mean, 1e-6)
       t_v_init = max(cavity_v.mean, 1e-6)
       if t_u_init <= t_v_init:
           t_u_init = t_v_init + 1.0  # ensure parent is older than child

       result = minimize(neg_log_tilted, [t_u_init, t_v_init],
                        method='Nelder-Mead')

       if not result.success:
           return None

       t_u_hat, t_v_hat = result.x

       # Compute Hessian numerically for the Laplace approximation
       H = numerical_hessian(neg_log_tilted, [t_u_hat, t_v_hat])

       try:
           cov = np.linalg.inv(H)  # covariance = inverse Hessian
       except np.linalg.LinAlgError:
           return None

       # Marginal moments from the Gaussian approximation
       mu_u = t_u_hat                    # mode ~ mean for peaked distributions
       var_u = max(cov[0, 0], 1e-20)    # diagonal of covariance = marginal variance
       mu_v = t_v_hat
       var_v = max(cov[1, 1], 1e-20)

       return mu_u, var_u, mu_v, var_v

   def numerical_hessian(f, x, eps=1e-5):
       """Compute the Hessian of f at x via finite differences.

       Uses the standard 4-point formula for mixed partial derivatives:
       d^2f/dxidxj ~ (f(+,+) - f(+,-) - f(-,+) + f(-,-)) / (4*eps^2)
       """
       n = len(x)
       H = np.zeros((n, n))
       f0 = f(x)
       for i in range(n):
           for j in range(i, n):
               x_pp = x.copy()
               x_pp[i] += eps
               x_pp[j] += eps
               x_pm = x.copy()
               x_pm[i] += eps
               x_pm[j] -= eps
               x_mp = x.copy()
               x_mp[i] -= eps
               x_mp[j] += eps
               x_mm = x.copy()
               x_mm[i] -= eps
               x_mm[j] -= eps
               H[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * eps**2)
               H[j, i] = H[i, j]  # Hessian is symmetric
       return H


Damping: Preventing Oscillation
==================================

EP updates can overshoot, causing oscillation or divergence. tsdate uses
**damping** to stabilize convergence: instead of fully replacing the old
message with the new one, it takes a weighted average.

In natural parameter space:

.. math::

   \eta^{\text{damped}} = (1 - d) \cdot \eta^{\text{old}} + d \cdot \eta^{\text{new}}

where :math:`d \in (0, 1]` is the damping factor. A typical value is
:math:`d = 0.5` (half-step).

**Why does this help?** Each EP update is based on a *local* approximation
(one edge at a time). If the approximation is poor, the update might push the
parameters too far. Damping ensures we only move a fraction of the way, giving
the other edges a chance to "catch up" before the next iteration.

In the watch metaphor, damping is like the balance wheel's hairspring: it
prevents the mechanism from swinging too far in response to a single impulse,
allowing it to settle smoothly into the correct position.


Convergence
=============

.. admonition:: Biology Aside -- Convergence means consistent dating

   Convergence of EP means that the inferred ages of all nodes in the
   genealogy have become mutually consistent. Each node's age agrees with the
   mutation evidence on every incident edge, with the coalescent prior (old
   nodes should have ages consistent with the expected coalescent times), and
   with the ages of its parents and children. When the algorithm converges,
   the age assignments satisfy all these constraints simultaneously -- or at
   least as well as the gamma approximation allows. In practice, ~25
   iterations suffice for tree sequences with millions of nodes.

EP iterates over all edges multiple times. Convergence is monitored by checking
whether the posteriors change significantly between iterations:

.. math::

   \max_u \frac{|\alpha_u^{(t+1)} - \alpha_u^{(t)}|}{|\alpha_u^{(t)}|} < \epsilon

tsdate defaults to 25 iterations, which is usually sufficient.

.. code-block:: python

   def run_ep(ts, mutation_rate, prior_grid, max_iter=25, damping=0.5, tol=1e-6):
       """Run the full EP algorithm.

       Parameters
       ----------
       ts : tskit.TreeSequence
       mutation_rate : float
       prior_grid : dict
       max_iter : int
       damping : float
       tol : float

       Returns
       -------
       posteriors : dict of GammaDistribution
       """
       posteriors, edge_factors = initialize_ep(ts, prior_grid)

       # Count mutations per edge (once, before the iteration loop)
       mut_per_edge = np.zeros(ts.num_edges, dtype=int)
       for mut in ts.mutations():
           if mut.edge >= 0:
               mut_per_edge[mut.edge] += 1

       for iteration in range(max_iter):
           max_change = 0.0  # track largest parameter change for convergence

           for edge in ts.edges():
               m_e = mut_per_edge[edge.id]
               span = edge.right - edge.left
               lambda_e = mutation_rate * span  # span-weighted mutation rate

               old_alpha = posteriors[edge.parent].alpha  # save for convergence check

               # The core EP update: cavity -> tilted -> moment match -> new factor
               ep_update_edge(edge, posteriors, edge_factors,
                             m_e, lambda_e, damping)

               # Track convergence: relative change in alpha
               change = abs(posteriors[edge.parent].alpha - old_alpha)
               rel_change = change / max(abs(old_alpha), 1e-10)
               max_change = max(max_change, rel_change)

           if max_change < tol:
               print(f"EP converged after {iteration + 1} iterations")
               break

       return posteriors


What EP Minimizes: The KL Divergence
=======================================

EP's fixed point (when it converges) approximately minimizes the **inclusive
Kullback-Leibler divergence**:

.. math::

   \text{KL}(p \| q) = \int p(\mathbf{t}) \log \frac{p(\mathbf{t})}{q(\mathbf{t})} \, d\mathbf{t}

where :math:`p` is the true posterior and :math:`q` is the gamma approximation.

**Why "inclusive" KL?** This is :math:`\text{KL}(p \| q)`, not
:math:`\text{KL}(q \| p)`. The inclusive KL penalizes :math:`q` for having
zero density where :math:`p` has mass. This means the approximation tends to
**cover** the true posterior rather than concentrating on a single mode.

**Contrast with variational Bayes**: Standard variational inference minimizes
:math:`\text{KL}(q \| p)` (the "exclusive" KL), which tends to underestimate
uncertainty. EP's inclusive KL typically gives better-calibrated uncertainty
estimates.

.. admonition:: Probability Aside -- KL divergence in 60 seconds

   The Kullback-Leibler divergence :math:`\text{KL}(p \| q)` measures how
   much information is lost when we use :math:`q` to approximate :math:`p`.
   It is always :math:`\geq 0`, and equals zero only when :math:`p = q`.
   Importantly, it is *not* symmetric: :math:`\text{KL}(p \| q) \neq
   \text{KL}(q \| p)`. The inclusive direction
   :math:`\text{KL}(p \| q)` heavily penalizes :math:`q` for placing zero
   density where :math:`p` is positive (missing mass), so the minimizer
   :math:`q^*` spreads out to cover :math:`p`. The exclusive direction
   :math:`\text{KL}(q \| p)` penalizes :math:`q` for placing density where
   :math:`p` is zero (extra mass), so the minimizer :math:`q^*` concentrates
   inside :math:`p`. For uncertainty quantification, the inclusive direction
   (used by EP) is generally more conservative and better calibrated.


Comparison with Inside-Outside
=================================

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Feature
     - Inside-Outside
     - Variational Gamma
   * - Time representation
     - Discrete grid (:math:`K` points)
     - Continuous (gamma distribution)
   * - Parameters per node
     - :math:`K` (probability vector)
     - 2 (:math:`\alpha, \beta`)
   * - Cost per edge per iteration
     - :math:`O(K^2)`
     - :math:`O(1)`
   * - Resolution
     - Limited by grid
     - Unlimited
   * - Posterior output
     - Full distribution (on grid)
     - Mean + variance (gamma)
   * - Convergence
     - 1 pass (exact on trees)
     - ~25 iterations
   * - Handles loops
     - Approximate
     - Approximate (EP)


Putting It All Together
=========================

.. code-block:: python

   def variational_gamma_date(ts, mutation_rate, Ne=1.0, max_iter=25):
       """Date a tree sequence using the variational gamma method.

       Parameters
       ----------
       ts : tskit.TreeSequence
       mutation_rate : float
       Ne : float
       max_iter : int

       Returns
       -------
       node_times : np.ndarray
       """
       # Build coalescent prior (Gear 1)
       prior_grid = {}
       for node in ts.nodes():
           if node.id not in set(ts.samples()):
               k = count_sample_descendants(ts, node.id)
               # Mean and variance from conditional coalescent
               mean = sum(2.0 / (j * (j-1)) for j in range(2, max(k, 2) + 1))
               var = sum(4.0 / (j * (j-1))**2 for j in range(2, max(k, 2) + 1))
               # Convert to gamma parameters via method of moments
               alpha = mean**2 / var
               beta = mean / var
               prior_grid[node.id] = (alpha, beta)

       # Run EP (messages flow through the gear train until convergence)
       posteriors = run_ep(ts, mutation_rate, prior_grid, max_iter)

       # Extract posterior means as point estimates
       node_times = np.zeros(ts.num_nodes)
       for u in range(ts.num_nodes):
           if u in posteriors:
               node_times[u] = posteriors[u].mean

       # Fix samples at time 0 (known ages)
       for s in ts.samples():
           node_times[s] = 0.0

       return node_times


Summary
========

The variational gamma method dates nodes through:

1. **Gamma approximation**: Each node's posterior is
   :math:`q(t_u) = \text{Gamma}(\alpha_u, \beta_u)`

2. **Expectation propagation**: For each edge, compute the exact moments of
   the tilted distribution (cavity :math:`\times` true likelihood), then
   moment-match back to gammas

3. **Damping**: Stabilize updates by interpolating between old and new messages

4. **Iteration**: Repeat over all edges until convergence (~25 iterations)

The key equations:

.. math::

   q_{\setminus e}(t_u) = q(t_u) / f_e^{\to u}(t_u) \quad \text{(cavity)}

.. math::

   q_{\text{new}}(t_u) = \text{Gamma}\bigl(\tilde{\mu}_u^2/\tilde{\sigma}_u^2, \;\;
   \tilde{\mu}_u / \tilde{\sigma}_u^2\bigr) \quad \text{(moment match)}

.. math::

   f_e^{\to u, \text{new}} = q_{\text{new}}(t_u) / q_{\setminus e}(t_u) \quad \text{(new message)}

This method is faster, more accurate, and higher-resolution than inside-outside,
which is why it's the default in modern tsdate. In the watch metaphor, it is
the same gear train carrying the same messages, but now the messages speak a
more efficient language -- two numbers per gear instead of a whole grid -- and
the mechanism converges more quickly to the correct time.

Next: the final gear, rescaling -- adjusting the inferred times to match the
empirical mutation clock (:ref:`tsdate_rescaling`).
