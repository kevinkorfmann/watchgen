.. _tsdate_mutation_likelihood:

========================
The Mutation Likelihood
========================

   *The ticking of the clock: every mutation is a beat, every edge a measured interval.*

The coalescent prior (previous chapter, :ref:`tsdate_coalescent_prior`) tells
us how old a node *should* be, absent any data -- the expected beat rate from
coalescent theory. Now we bring in the data: the mutations observed on each
edge of the tree sequence. This is the **likelihood** -- the probability of the
data given the node ages.

If the prior is the expected beat rate, the likelihood is **evidence from the
mutation clock**: each mutation is a tick, and the number of ticks on an edge
tells us how long that edge lasted. Together, prior and likelihood will be
combined by Bayes' rule to produce the posterior -- the calibrated age
estimates.


The Molecular Clock
=====================

The molecular clock is one of the most powerful ideas in evolutionary genetics.
It says:

   **Mutations accumulate at a roughly constant rate per base pair per generation.**

If an edge in the genealogy spans :math:`\Delta t` generations and covers
:math:`\ell` base pairs, then the expected number of mutations on that edge is:

.. math::

   \mathbb{E}[\text{mutations}] = \mu \cdot \ell \cdot \Delta t

where :math:`\mu` is the per-base-pair, per-generation mutation rate. This is a
linear relationship: longer branches accumulate more mutations.

**Why is this useful?** Because we *observe* the mutations (they show up as
differences between sequences), and we want to infer :math:`\Delta t`. The
molecular clock converts observed mutations into time estimates.

Think of it as a metronome inside each branch of the tree: it ticks at rate
:math:`\mu \cdot \ell`, and the number of ticks recorded on the branch is the
mutation count :math:`m_e`. Counting ticks tells us how long the metronome ran.


The Poisson Model
==================

tsdate models the number of mutations on each edge as a **Poisson random
variable**. This is a natural choice because:

1. Mutations are rare events occurring along a long sequence -- the classic
   Poisson regime.
2. Each base pair mutates independently (approximately).
3. The Poisson distribution has a single parameter (the mean), which is directly
   proportional to the branch length.

For edge :math:`e` connecting parent :math:`u` to child :math:`v`:

.. math::

   m_e \sim \text{Poisson}(\lambda_e \cdot \Delta t_e)

where:

- :math:`m_e` = observed number of mutations on this edge
- :math:`\lambda_e = \mu \cdot \ell_e` = the "span-weighted mutation rate"
- :math:`\Delta t_e = t_u - t_v` = the branch length (what we want to infer)
- :math:`\ell_e` = the genomic span of the edge in base pairs

The Poisson probability mass function gives us:

.. math::

   P(m_e \mid t_u, t_v) = \frac{(\lambda_e \Delta t_e)^{m_e}}{m_e!}
   \exp(-\lambda_e \Delta t_e)

.. admonition:: Probability Aside -- The Poisson distribution

   The Poisson distribution models the number of events occurring in a fixed
   interval when events happen independently at a constant rate. If the
   expected number of events is :math:`\lambda`, then
   :math:`P(k) = \lambda^k e^{-\lambda} / k!`. Its mean and variance are
   both :math:`\lambda`. In our context the "interval" is the branch length
   in base-pair-generations, the "rate" is :math:`\mu`, and the "events" are
   mutations. The Poisson is appropriate whenever :math:`\mu` is small and the
   number of sites :math:`\ell` is large -- exactly the regime of genomic
   mutations.

.. code-block:: python

   import numpy as np
   from scipy.stats import poisson

   def edge_likelihood(m_e, lambda_e, t_parent, t_child):
       """Poisson likelihood for mutations on a single edge.

       Parameters
       ----------
       m_e : int
           Observed mutation count on this edge.
       lambda_e : float
           Span-weighted mutation rate (mu * span_bp).
       t_parent : float
           Age of parent node.
       t_child : float
           Age of child node.

       Returns
       -------
       likelihood : float
           P(m_e | t_parent, t_child).
       """
       delta_t = t_parent - t_child          # branch length in generations
       if delta_t <= 0:
           return 0.0                         # parent must be older than child
       expected = lambda_e * delta_t          # Poisson mean = rate * time
       return poisson.pmf(m_e, expected)      # evaluate the Poisson PMF

   # Example: an edge spanning 10,000 bp with mu = 1e-8
   mu = 1e-8
   span_bp = 10_000
   lambda_e = mu * span_bp  # = 1e-4 mutations per generation

   # If parent is 500 generations older than child:
   delta_t = 500
   expected_mutations = lambda_e * delta_t  # = 0.05

   print(f"Expected mutations: {expected_mutations:.4f}")
   print(f"P(0 mutations) = {edge_likelihood(0, lambda_e, 500, 0):.6f}")
   print(f"P(1 mutation)  = {edge_likelihood(1, lambda_e, 500, 0):.6f}")
   print(f"P(2 mutations) = {edge_likelihood(2, lambda_e, 500, 0):.6f}")


The Full Likelihood
=====================

The total likelihood is the product over all edges (assuming mutations on
different edges are independent):

.. math::

   P(\mathbf{D} \mid \mathbf{t}) = \prod_{e \in \mathcal{E}}
   \frac{(\lambda_e \Delta t_e)^{m_e}}{m_e!} \exp(-\lambda_e \Delta t_e)

Taking the log:

.. math::

   \log P(\mathbf{D} \mid \mathbf{t}) = \sum_{e \in \mathcal{E}}
   \left[ m_e \log(\lambda_e \Delta t_e) - \lambda_e \Delta t_e - \log(m_e!) \right]

.. admonition:: Calculus Aside -- Why work with the log-likelihood?

   Products of many small probabilities lead to numerical underflow (numbers
   too small for floating point). By taking logarithms, products become sums,
   which are numerically stable. Additionally, the log turns the exponential
   in the Poisson PMF into a linear term :math:`-\lambda_e \Delta t_e`,
   making derivatives easier to compute. Throughout tsdate, nearly all
   computations happen in log space.

.. code-block:: python

   from scipy.special import gammaln

   def total_log_likelihood(ts, node_times, mutation_rate):
       """Compute the total log-likelihood of observed mutations.

       Parameters
       ----------
       ts : tskit.TreeSequence
           Tree sequence with mutations.
       node_times : np.ndarray
           Proposed times for each node.
       mutation_rate : float
           Per-bp per-generation mutation rate.

       Returns
       -------
       log_lik : float
           Total log-likelihood.
       """
       # Count mutations per edge
       mut_per_edge = np.zeros(ts.num_edges, dtype=int)
       for mut in ts.mutations():
           if mut.edge >= 0:
               mut_per_edge[mut.edge] += 1   # tally mutations on their parent edge

       log_lik = 0.0
       for edge in ts.edges():
           m_e = mut_per_edge[edge.id]        # observed mutation count for this edge
           span = edge.right - edge.left      # genomic span in base pairs
           lambda_e = mutation_rate * span     # span-weighted rate
           delta_t = node_times[edge.parent] - node_times[edge.child]  # branch length

           if delta_t <= 0:
               return -np.inf  # invalid: parent younger than child

           expected = lambda_e * delta_t                              # Poisson mean
           # Poisson log-pmf: m*log(expected) - expected - log(m!)
           log_lik += m_e * np.log(expected) - expected - gammaln(m_e + 1)

       return log_lik


The Likelihood as a Function of Branch Length
================================================

For a single edge, the likelihood as a function of :math:`\Delta t` has a
clean shape. Let's see why.

Fix :math:`m_e` and :math:`\lambda_e`. The log-likelihood as a function of
:math:`\Delta t` is:

.. math::

   \ell(\Delta t) = m_e \log(\lambda_e \Delta t) - \lambda_e \Delta t - \log(m_e!)

Taking the derivative and setting to zero:

.. math::

   \frac{d\ell}{d(\Delta t)} = \frac{m_e}{\Delta t} - \lambda_e = 0
   \quad \Rightarrow \quad
   \widehat{\Delta t}_{\text{MLE}} = \frac{m_e}{\lambda_e}

.. admonition:: Calculus Aside -- Finding the MLE

   The maximum likelihood estimate (MLE) is the parameter value that
   maximizes the likelihood (or equivalently, the log-likelihood). We
   differentiate :math:`\ell(\Delta t) = m_e \log(\lambda_e \Delta t)
   - \lambda_e \Delta t + C` with respect to :math:`\Delta t`. The
   :math:`\log` term gives :math:`m_e / \Delta t` (chain rule), and the
   linear term gives :math:`-\lambda_e`. Setting the sum to zero yields
   :math:`\Delta t = m_e / \lambda_e`. The second derivative is
   :math:`-m_e / (\Delta t)^2 < 0`, confirming this is a maximum.

**The MLE is beautifully intuitive**: the best estimate for the branch length is
the observed mutation count divided by the expected rate. More mutations = longer
branch.

The second derivative is :math:`-m_e / (\Delta t)^2 < 0`, confirming this is a
maximum.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   def plot_edge_likelihood(m_e=3, lambda_e=0.01):
       """Plot the likelihood curve for a single edge."""
       delta_t = np.linspace(0.01, 800, 1000)
       # Log-likelihood (up to a constant) as a function of branch length
       log_lik = m_e * np.log(lambda_e * delta_t) - lambda_e * delta_t

       fig, ax = plt.subplots(figsize=(8, 5))
       # Plot relative likelihood (normalized so max = 1)
       ax.plot(delta_t, np.exp(log_lik - log_lik.max()), lw=2)
       ax.axvline(m_e / lambda_e, color='red', ls='--',
                  label=f'MLE = {m_e/lambda_e:.0f} gen')
       ax.set_xlabel('Branch length (generations)')
       ax.set_ylabel('Relative likelihood')
       ax.set_title(f'Edge likelihood: {m_e} mutations, lambda={lambda_e}')
       ax.legend()
       return fig

   # plot_edge_likelihood()


The Edge Factor in the Factor Graph
======================================

Having established the Poisson likelihood for a single edge, let us now see
how it fits into the factor graph introduced in :ref:`tsdate_overview`. Each
edge :math:`e = (u, v)` contributes a **factor** that connects the parent age
:math:`t_u` and child age :math:`t_v`:

.. math::

   \phi_e(t_u, t_v) = \frac{(\lambda_e (t_u - t_v))^{m_e}}{m_e!}
   \exp(-\lambda_e (t_u - t_v)) \cdot \mathbb{1}[t_u > t_v]

This factor is **bivariate**: it depends on both the parent and child ages.
The indicator :math:`\mathbb{1}[t_u > t_v]` enforces the constraint that
parents must be older than children.

For message passing, we need to compute **messages** from this factor to each
connected node. A message from factor :math:`\phi_e` to node :math:`u` is
obtained by "integrating out" the other variable:

.. math::

   \text{msg}_{e \to u}(t_u) = \int_0^{t_u} \phi_e(t_u, t_v) \cdot q(t_v) \, dt_v

where :math:`q(t_v)` is the current belief (approximate posterior) for node
:math:`v`. This integral is what the inside-outside and EP algorithms compute.
In the watch metaphor, each edge factor is a spring connecting two gears: it
transmits information (force) between them, and the message is the resulting
torque on each gear.

.. admonition:: Calculus Aside -- Marginalizing by integration

   "Integrating out" a variable means computing
   :math:`\int f(x, y) \, dy` to obtain a function of :math:`x` alone.
   This is how we go from a joint distribution over two variables (parent and
   child age) to a marginal message about one variable. In the discrete
   inside-outside method this integral becomes a sum over grid points; in the
   variational gamma method it is evaluated approximately via the Laplace
   method or numerical quadrature.


Gamma-Poisson Conjugacy
==========================

Here's where the choice of gamma priors pays off. If the prior on :math:`\Delta t`
is :math:`\text{Gamma}(\alpha, \beta)`, and we observe :math:`m` mutations with
rate :math:`\lambda`, then the posterior on :math:`\Delta t` is:

.. math::

   P(\Delta t \mid m) \propto \underbrace{(\Delta t)^{m} e^{-\lambda \Delta t}}_{\text{Poisson likelihood}} \cdot
   \underbrace{(\Delta t)^{\alpha - 1} e^{-\beta \Delta t}}_{\text{Gamma prior}}

.. math::

   = (\Delta t)^{(\alpha + m) - 1} e^{-(\beta + \lambda) \Delta t}

This is a :math:`\text{Gamma}(\alpha + m, \beta + \lambda)` distribution!

**This is conjugacy**: the posterior is in the same family as the prior. The
update rules are simply:

.. math::

   \alpha_{\text{post}} = \alpha_{\text{prior}} + m_e, \qquad
   \beta_{\text{post}} = \beta_{\text{prior}} + \lambda_e

.. admonition:: Probability Aside -- What is conjugacy?

   A prior family is *conjugate* to a likelihood if the posterior belongs to
   the same family as the prior. For the Gamma-Poisson pair: a Gamma prior on
   the rate of a Poisson observation yields a Gamma posterior. Why does this
   matter? Because the update from prior to posterior reduces to adding
   numbers (:math:`\alpha + m`, :math:`\beta + \lambda`) instead of
   evaluating integrals. In tsdate, conjugacy means that -- for a single
   isolated edge -- the posterior on branch length is immediately available in
   closed form. The complication in the full problem is that edges share
   nodes, so the conjugacy cannot be applied independently to each edge.
   That is why we need message passing.

.. code-block:: python

   def gamma_poisson_update(alpha_prior, beta_prior, m_e, lambda_e):
       """Update gamma parameters given Poisson observations.

       The gamma-Poisson conjugacy gives a closed-form posterior.

       Parameters
       ----------
       alpha_prior, beta_prior : float
           Prior gamma parameters.
       m_e : int
           Observed mutations.
       lambda_e : float
           Span-weighted mutation rate.

       Returns
       -------
       alpha_post, beta_post : float
           Posterior gamma parameters.
       """
       # Conjugate update: just add mutation count to shape, rate to rate
       return alpha_prior + m_e, beta_prior + lambda_e

   # Example: prior Gamma(2, 3), observe 5 mutations with rate 0.01
   alpha_prior, beta_prior = 2.0, 3.0
   m_e, lambda_e = 5, 0.01
   alpha_post, beta_post = gamma_poisson_update(alpha_prior, beta_prior, m_e, lambda_e)

   print(f"Prior:     Gamma({alpha_prior}, {beta_prior})")
   print(f"  mean = {alpha_prior/beta_prior:.4f}")
   print(f"Posterior: Gamma({alpha_post}, {beta_post})")
   print(f"  mean = {alpha_post/beta_post:.4f}")

.. admonition:: A subtlety: it's not quite this simple

   The conjugacy result above applies when the branch length :math:`\Delta t`
   is a free variable with a gamma prior. But in tsdate, the branch length
   is :math:`t_u - t_v`, a *difference* of two random variables. The parent
   and child are not independent -- they're connected through the tree.
   This is why we need message passing: the prior on :math:`t_u` gets
   "messages" from all edges connected to :math:`u`, not just one.

   The true posterior on :math:`t_u` is a product of gamma-like factors from
   each connected edge plus the coalescent prior. This product is generally
   *not* gamma, but the variational gamma method approximates it as one.


Handling Edges with Zero Mutations
=====================================

Many edges in a tree sequence have zero mutations. This is not a problem for
the Poisson model -- in fact, it's the most common case for short edges or
edges with small spans.

When :math:`m_e = 0`:

.. math::

   P(0 \mid \lambda_e, \Delta t_e) = \exp(-\lambda_e \Delta t_e)

This is a pure exponential decay: longer branches are exponentially less
likely to have zero mutations. The likelihood still provides information --
it says "this branch is probably short."

The MLE for :math:`\Delta t` when :math:`m_e = 0` is :math:`\hat{\Delta t} = 0`,
which is degenerate. This is where the prior matters most: it pulls the estimate
away from zero to a biologically reasonable age. In our watch metaphor, an edge
with zero mutations is like a spring that shows no wear -- the mutation clock
recorded no ticks, so we lean on our prior knowledge (the expected beat rate
from coalescent theory) to estimate how long it ran.


Handling Shared Edges
========================

In a tree sequence, nodes can participate in many edges (across different
genomic intervals). A single ancestral node might be the parent of different
children in different regions of the genome.

The total likelihood contribution for node :math:`u` combines information from
*all* edges where :math:`u` is a parent or child:

.. math::

   P(\mathbf{D} \mid t_u, \ldots) = \prod_{e : \text{parent}(e) = u}
   P(m_e \mid t_u, t_{\text{child}(e)})
   \cdot \prod_{e : \text{child}(e) = u}
   P(m_e \mid t_{\text{parent}(e)}, t_u)

This is what makes the problem interconnected: changing :math:`t_u` affects the
likelihood of every edge connected to :math:`u`. In the gear train, each gear
meshes with multiple neighbors, and adjusting one gear shifts them all.

.. code-block:: python

   def node_log_likelihood_contribution(ts, node_id, node_times, mutation_rate):
       """Log-likelihood contribution from all edges connected to a node.

       Parameters
       ----------
       ts : tskit.TreeSequence
       node_id : int
       node_times : np.ndarray
       mutation_rate : float

       Returns
       -------
       log_lik : float
       """
       # First, count mutations per edge
       mut_per_edge = np.zeros(ts.num_edges, dtype=int)
       for mut in ts.mutations():
           if mut.edge >= 0:
               mut_per_edge[mut.edge] += 1

       log_lik = 0.0
       for edge in ts.edges():
           # Only consider edges connected to this node
           if edge.parent == node_id or edge.child == node_id:
               m_e = mut_per_edge[edge.id]
               span = edge.right - edge.left
               lambda_e = mutation_rate * span
               delta_t = node_times[edge.parent] - node_times[edge.child]

               if delta_t <= 0:
                   return -np.inf  # invalid configuration

               expected = lambda_e * delta_t
               log_lik += m_e * np.log(expected) - expected - gammaln(m_e + 1)

       return log_lik


Summary
========

The mutation likelihood is the data-driven half of tsdate's Bayesian framework:

.. math::

   P(m_e \mid t_u, t_v) = \text{Poisson}(m_e; \lambda_e \cdot (t_u - t_v))

Key takeaways:

- Mutations follow a **Poisson process** along each edge
- The expected count is proportional to **branch length** :math:`\times` **span** :math:`\times` **mutation rate**
- The MLE for branch length is simply :math:`m_e / \lambda_e`
- Gamma priors are **conjugate** to the Poisson likelihood, enabling closed-form updates for single edges
- But the tree structure couples all nodes, requiring **message passing** to propagate information

In the watch metaphor, the mutation likelihood is the evidence from the
mutation clock -- the wear marks that tell us how long each spring has been
under tension. Combined with the coalescent prior (the expected beat rate),
these two gears produce the posterior via Bayes' rule.

We now have both halves of Bayes' rule: the prior (Gear 1) and the likelihood
(Gear 2). Next, we'll combine them using belief propagation. First up: the
discrete-time inside-outside algorithm (:ref:`tsdate_inside_outside`).
