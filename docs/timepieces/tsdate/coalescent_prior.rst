.. _tsdate_coalescent_prior:

====================
The Coalescent Prior
====================

   *A watchmaker who knows how clocks age can guess a part's vintage before inspecting it.*

Before looking at any mutational data, we already know something about how old
each node should be. A node with 1000 descendant samples is expected to be
older than a node with only 2. This knowledge comes from **coalescent
theory**, and tsdate encodes it as a **prior distribution** on each node's age.

In the watch metaphor, the prior is **the expected beat rate from coalescent
theory** -- our best guess for when each gear was manufactured, before we open
the case and inspect the wear marks (mutations). Larger descendant sets require
more coalescences below their ancestral node and therefore shift the prior
toward older ages.

This chapter builds the conditional-coalescent moments from first principles.
The discrete methods approximate them with a lognormal distribution by default
or an optional gamma distribution before combining them with the mutation
likelihood.

.. admonition:: Prerequisites

   This chapter assumes familiarity with the standard coalescent model
   (covered in the population genetics fundamentals). You should also
   understand why tsdate needs a tree sequence with known topology -- that
   topology comes from tsinfer (see :ref:`tsinfer_overview`).


The Intuition: More Descendants = Older
=========================================

Under the standard coalescent, a node with :math:`k` descendant leaves in a
sample of :math:`n` coalesces at a time that depends on :math:`k` and :math:`n`.
The key intuition:

- A node that is ancestral to *all* :math:`n` samples (the root) has had
  :math:`n-1` coalescence events below it. It must be old.
- A node that is ancestral to just :math:`k=2` samples only needs one
  coalescence event. It can be young.

More precisely, under the standard coalescent with constant population size
:math:`N_e`, the expected time for :math:`j` lineages to coalesce to :math:`j-1`
is:

.. math::

   \mathbb{E}[T_j] = \frac{2N_e}{j(j-1)/2} = \frac{4N_e}{j(j-1)}

The total time from :math:`n` lineages down to 1 is the sum of waiting times,
and a node ancestral to :math:`k` leaves enters the picture somewhere in this
process.

.. admonition:: Probability Aside -- Why :math:`j(j-1)/2`?

   When there are :math:`j` lineages in the population, any pair can coalesce.
   The number of possible pairs is :math:`\binom{j}{2} = j(j-1)/2`. Each pair
   coalesces independently at rate :math:`1/(2N_e)`, so the total coalescence
   rate is :math:`\binom{j}{2} / (2N_e)`. The waiting time until the next
   coalescence is Exponential with this rate, giving the mean
   :math:`4N_e / (j(j-1))`. As :math:`j` grows, there are many more pairs,
   coalescence is faster, and the waiting time is shorter.


The Conditional Coalescent
============================

tsdate uses the **conditional coalescent** (Wiuf & Donnelly, 1999) to derive
the prior. The question is:

   Given a tree with :math:`n` total leaves, what is the distribution of the
   age of a node that is ancestral to exactly :math:`k` of those leaves?

This is not a simple closed-form expression. It requires integrating over the
possible number of **extant ancestors** :math:`a` -- the number of lineages
that exist at the time this particular subtree coalesces.

With the intuition established, let us now work through the mathematics of the
conditional coalescent. The key difficulty is that when a subtree of size
:math:`k` finishes coalescing, the number of remaining lineages in the rest of
the tree is random.

The mean and variance
-----------------------

The conditional coalescent gives us :math:`\mathbb{E}[t \mid k, n]` and
:math:`\text{Var}(t \mid k, n)`. These are computed by marginalizing over
the number of ancestors.

When a subtree of size :math:`k` coalesces (going back in time from the
present), there are :math:`a` total lineages remaining. The probability
of having :math:`a` ancestors given :math:`k` and :math:`n` follows a
hypergeometric-like distribution (Wiuf & Donnelly, 1999). Conditional on
:math:`a`, node age is a **sum** of exponential waiting times, not one
exponential draw:

.. math::

   T \mid a \;\sim\; \sum_{j=a}^{n} \text{Exp}\left(\binom{j}{2}\right)

in standard coalescent units. Marginalizing the resulting hypoexponential
moments over :math:`P(a\mid k,n)` gives the variance used by tsdate. The mean
simplifies to an exact expression:

.. math::

   \mathbb{E}[T \mid k,n] =
   \begin{cases}
   (k-1)/n, & k<n,\\
   2(1-1/n), & k=n.
   \end{cases}

and the conditional variance includes both the variance within each :math:`a`
class and the variance between classes (law of total variance).

.. admonition:: Probability Aside -- The law of total variance

   The law of total variance (sometimes called Eve's law) says that for any
   two random variables :math:`X` and :math:`Y`:

   .. math::

      \text{Var}(X) = \mathbb{E}[\text{Var}(X \mid Y)] + \text{Var}(\mathbb{E}[X \mid Y])

   In our case, :math:`X` is the coalescence time :math:`T` and :math:`Y` is
   the number of ancestors :math:`a`. The first term captures the randomness
   *within* each value of :math:`a` (the exponential waiting time), and the
   second term captures the randomness *between* values of :math:`a` (different
   numbers of lineages lead to different expected times). This decomposition
   is how tsdate computes the variance of the conditional coalescent without
   needing the full distribution.

.. code-block:: python

   from watchgen.mini_tsdate import conditional_coalescent_mean

   # Exact values used by tsdate's conditional-coalescent lookup.
   assert conditional_coalescent_mean(2, 4) == 0.25
   assert conditional_coalescent_mean(3, 4) == 0.5
   assert conditional_coalescent_mean(4, 4) == 1.5


The Recursive Computation
---------------------------

The key to computing :math:`P(a \mid k, n)` efficiently is a log-space
**recursive relationship** over decreasing :math:`k`. The production code
updates the complete ancestor-count probability vector; it is not the simple
two-state transition shown in earlier versions of this chapter.

The base case is :math:`k = n-1`, where the subtree is the second-to-last
to coalesce, and there are exactly :math:`a = 2` ancestors.

In practice, tsdate precomputes a lookup table of :math:`(\text{mean}, \text{variance})`
indexed by :math:`k` (number of descendants), for a given :math:`n` (total tips).

Now let us translate this recursion into code. The implementation walks backward
from :math:`k = n-1` down to :math:`k = 2`, building the probability table one
row at a time.

.. code-block:: python

   import numpy as np
   from watchgen.mini_tsdate import conditional_coalescent_moments

   moments = conditional_coalescent_moments(4)
   assert np.isclose(moments[2][1], 11 / 144)
   assert np.isclose(moments[3][1], 5 / 36)
   assert np.isclose(moments[4][1], 41 / 36)

The mini implementation follows the production log-space recursion and sums
the appropriate exponential waiting-time moments. Keeping one tested
implementation avoids the incomplete transition formula that previously
appeared in this chapter.


From Moments to Gamma Parameters
===================================

With the mean and variance of the conditional coalescent in hand, the next step
is to convert them into a form that a discrete dating algorithm can use. Current
tsdate uses a **lognormal approximation by default** for these priors and also
supports a gamma approximation. The mini implementation demonstrates the
optional gamma moment match; variational gamma uses a different learned prior.

Given mean :math:`\mu` and variance :math:`\sigma^2`, the gamma parameters are:

.. math::

   \alpha = \frac{\mu^2}{\sigma^2}, \qquad
   \beta = \frac{\mu}{\sigma^2}

This is the standard method-of-moments estimator. Let's verify:

.. math::

   \mathbb{E}[\text{Gamma}(\alpha, \beta)] = \frac{\alpha}{\beta}
   = \frac{\mu^2/\sigma^2}{\mu/\sigma^2} = \mu \quad \checkmark

.. math::

   \text{Var}[\text{Gamma}(\alpha, \beta)] = \frac{\alpha}{\beta^2}
   = \frac{\mu^2/\sigma^2}{\mu^2/\sigma^4} = \sigma^2 \quad \checkmark

.. admonition:: Calculus Aside -- Method of moments

   Method of moments is one of the oldest techniques in statistics. The idea:
   set the theoretical moments of a distribution equal to the observed (or
   computed) moments, then solve for the parameters. For a Gamma(:math:`\alpha`,
   :math:`\beta`), the first two moments are :math:`\mu_1 = \alpha/\beta` and
   :math:`\mu_2 = \alpha(\alpha+1)/\beta^2`. From the mean :math:`\mu_1` and
   variance :math:`\sigma^2 = \mu_2 - \mu_1^2 = \alpha/\beta^2`, we solve
   the two equations in two unknowns to get :math:`\alpha = \mu_1^2/\sigma^2`
   and :math:`\beta = \mu_1/\sigma^2`. This is simple and fast, which is why
   tsdate uses it instead of maximum likelihood estimation for the prior
   parameters.

.. code-block:: python

   def gamma_params_from_moments(mean, variance):
       """Convert mean and variance to gamma distribution parameters.

       Parameters
       ----------
       mean : float
           E[T] from the conditional coalescent.
       variance : float
           Var[T] from the conditional coalescent.

       Returns
       -------
       alpha : float
           Shape parameter (controls peakedness of the distribution).
       beta : float
           Rate parameter (controls how quickly the density decays).
       """
       alpha = mean**2 / variance   # shape = mean^2 / variance
       beta = mean / variance       # rate  = mean / variance
       return alpha, beta

   # Example: node with k=3 descendants in a sample of n=100
   # The conditional coalescent gives approximate values:
   k, n = 3, 100
   # For small k relative to n, the mean is approximately 2/(k*(k-1))
   approx_mean = 2.0 / (k * (k - 1))  # = 0.333 in coalescent units
   approx_var = approx_mean**2          # exponential: var = mean^2

   alpha, beta = gamma_params_from_moments(approx_mean, approx_var)
   print(f"k={k}: mean={approx_mean:.4f}, var={approx_var:.4f}")
   print(f"  Gamma prior: alpha={alpha:.4f}, beta={beta:.4f}")


The Approximate Prior for Large :math:`n`
============================================

Computing exact conditional coalescent moments for every possible
:math:`(k, n)` pair is expensive when :math:`n` is large. tsdate uses a
**lookup table with interpolation**:

1. Precompute exact moments for :math:`k = 2, 3, \ldots, n` (or a subsample)
2. Store as arrays indexed by :math:`k`
3. For nodes with the same :math:`k`, reuse the same prior

The key array in tsdate's implementation is a **prior grid**: for each possible
number of descendant leaves :math:`k`, store :math:`(\alpha_k, \beta_k, \mu_k, \sigma^2_k)`.

.. code-block:: python

   import numpy as np

   def build_prior_grid(n, Ne=1.0):
       """Build a lookup table of gamma priors indexed by descendant count.

       Parameters
       ----------
       n : int
           Total number of sample leaves.
       Ne : float
           Effective population size.

       Returns
       -------
       prior_grid : np.ndarray, shape (n+1, 4)
           Columns: [alpha, beta, mean, variance]
           Row k gives the prior for a node with k descendants.
           Rows 0 and 1 are unused (no node has 0 or 1 non-self descendants).
       """
       grid = np.zeros((n + 1, 4))
       moments = conditional_coalescent_moments(n, Ne)  # compute all (mean, var) pairs

       for k in range(2, n + 1):
           mean, var = moments[k]
           alpha, beta = gamma_params_from_moments(mean, var)
           grid[k] = [alpha, beta, mean, var]  # store both parameterizations

       return grid


Special Cases
===============

Before moving on, let us address two boundary cases that arise in every tree
sequence: the root (which is the oldest node) and the leaves (whose ages are
known).

Roots
------

For the root of a tree (or a connected component in the tree sequence), tsdate
assigns an **exponential prior** rather than a conditional coalescent prior.
The exponential distribution is :math:`\text{Gamma}(1, \beta)`, and the rate
:math:`\beta` is set so the mean matches the expected TMRCA.

For the variational gamma method, root priors are handled differently: they get
a weakly informative mixture prior that allows for a wide range of ages.

Leaves (samples)
------------------

Leaf nodes have known ages. Modern samples are at time 0. Ancient samples
(e.g., from aDNA) have their age set to the sample's radiocarbon date. These
are **fixed nodes** -- they don't need priors because their ages are observed.

.. code-block:: python

   def assign_node_priors(ts, prior_grid):
       """Assign a gamma prior to each non-leaf node.

       Parameters
       ----------
       ts : tskit.TreeSequence
           The input tree sequence (topology from tsinfer).
       prior_grid : np.ndarray
           From build_prior_grid().

       Returns
       -------
       priors : dict
           {node_id: (alpha, beta)} for each non-fixed node.
       """
       priors = {}
       fixed_nodes = set(ts.samples())  # samples have known ages -- no prior needed

       for node in ts.nodes():
           if node.id in fixed_nodes:
               continue  # known age, no prior needed

           # Count descendants: number of samples below this node
           k = count_sample_descendants(ts, node.id)

           if k >= 2 and k <= ts.num_samples:
               # Look up the precomputed gamma prior for this descendant count
               alpha, beta = prior_grid[k, 0], prior_grid[k, 1]
               priors[node.id] = (alpha, beta)
           else:
               # Fallback: exponential prior for nodes with unusual topology
               priors[node.id] = (1.0, 1.0)

       return priors

   def count_sample_descendants(ts, node_id):
       """Count the number of sample leaves descended from a node."""
       samples = set(ts.samples())
       count = 0
       for tree in ts.trees():
           for leaf in tree.leaves(node_id):
               if leaf in samples:
                   count += 1
           break  # only need one tree (approximate for polytomies)
       return count


Putting It Together: A Visualization
=======================================

Let's visualize what the prior looks like for different descendant counts. This
will make concrete the central idea of this chapter: nodes with more descendants
get priors shifted toward older ages.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from scipy.stats import gamma

   def plot_coalescent_priors(n=50, Ne=1.0):
       """Plot gamma priors for nodes with different numbers of descendants."""
       fig, ax = plt.subplots(figsize=(10, 6))
       t = np.linspace(0, 4, 500)  # time axis in coalescent units

       descendant_counts = [2, 5, 10, 25, 49]
       colors = plt.cm.viridis(np.linspace(0, 0.9, len(descendant_counts)))

       for k, color in zip(descendant_counts, colors):
           # Approximate moments for illustration
           # Mean age ~ sum of 1/(j choose 2) for j = k down to 2
           mean = sum(2.0 / (j * (j - 1)) for j in range(2, k + 1))
           var = sum(4.0 / (j * (j - 1))**2 for j in range(2, k + 1))

           alpha = mean**2 / var   # shape from method of moments
           beta = mean / var       # rate from method of moments

           pdf = gamma.pdf(t, a=alpha, scale=1.0/beta)
           ax.plot(t, pdf, color=color, lw=2, label=f'k={k} (mean={mean:.2f})')

       ax.set_xlabel('Node age (coalescent units)')
       ax.set_ylabel('Prior density')
       ax.set_title(f'Coalescent Prior for Different Descendant Counts (n={n})')
       ax.legend()
       ax.set_xlim(0, 4)

       return fig

   # plot_coalescent_priors()

**What you should see**: Nodes with more descendants (larger :math:`k`) have
priors shifted to the right (older ages), with more spread. Nodes with :math:`k=2`
have a tight, exponential-like prior near the present. The root (:math:`k=n`)
has the broadest, most right-shifted prior.

Think of it this way: a gear deep inside the movement (ancestral to many parts)
must have been installed early in the watch's construction. A gear near the
dial (ancestral to just two leaves) could have been added at any stage.


Summary
========

The coalescent prior gives tsdate a principled starting point for each node:

.. math::

   t_u \sim \text{Gamma}(\alpha_k, \beta_k) \quad \text{where } k = |\text{descendants}(u)|

The parameters :math:`(\alpha_k, \beta_k)` come from fitting gamma distributions
to the mean and variance of the conditional coalescent. This prior encodes the
simple but powerful idea: **nodes ancestral to more samples are expected to be
older**.

In our watch metaphor, the coalescent prior is the expected beat rate -- the
baseline rhythm we expect from population genetics before any mutation data
enters the picture. It sets the initial position of every hand on the dial.

Next, we need the other half of Bayes' rule: the likelihood. How do observed
mutations inform us about branch lengths? That's the subject of the next
chapter: :ref:`tsdate_mutation_likelihood`.
