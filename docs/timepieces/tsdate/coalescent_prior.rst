.. _tsdate_coalescent_prior:

====================
The Coalescent Prior
====================

   *A watchmaker who knows how clocks age can guess a part's vintage before inspecting it.*

Before looking at any mutational data, we already know something about how old
each node should be. A node with 1000 descendant samples is almost certainly
much younger than a node with only 2. This knowledge comes from **coalescent
theory**, and tsdate encodes it as a **prior distribution** on each node's age.

In the watch metaphor, the prior is **the expected beat rate from coalescent
theory** -- our best guess for when each gear was manufactured, before we open
the case and inspect the wear marks (mutations). A node with many descendants
is like a mass-produced part: it was probably made recently. A node ancestral
to only two samples is like a rare vintage component: it could be quite old.

This chapter builds the coalescent prior from first principles. By the end you
will be able to assign a Gamma(:math:`\alpha`, :math:`\beta`) distribution to
every internal node of a tree sequence, ready to combine with the mutation
likelihood in the chapters that follow.

.. admonition:: Prerequisites

   This chapter assumes familiarity with the standard coalescent model
   (covered in the population genetics fundamentals). You should also
   understand why tsdate needs a tree sequence with known topology -- that
   topology comes from tsinfer (see :ref:`tsinfer_overview`).


The Intuition: More Descendants = Younger
============================================

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
hypergeometric-like distribution (Wiuf & Donnelly, 1999), and the coalescence
time conditioned on :math:`a` is:

.. math::

   T \mid a \;\sim\; \text{Exp}\left(\frac{a(a-1)}{4N_e}\right)

So the conditional mean is:

.. math::

   \mathbb{E}[T \mid k, n] = \sum_{a=2}^{n-k+1} P(a \mid k, n) \cdot \frac{4N_e}{a(a-1)}

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

   import numpy as np
   from scipy.special import comb

   def conditional_coalescent_mean(k, n, Ne=1.0):
       """Mean age of a node with k descendants in a sample of n.

       Under the conditional coalescent (Wiuf & Donnelly, 1999), averaged
       over the number of extant ancestors.

       Parameters
       ----------
       k : int
           Number of descendant leaves of this node.
       n : int
           Total number of leaves in the tree.
       Ne : float
           Effective population size (in coalescent units, 2*Ne generations).

       Returns
       -------
       mean : float
           Expected age in units of 2*Ne generations.
       """
       if k == n:
           # The root: must wait for all n lineages to coalesce
           # Mean is sum of 1/(j choose 2) for j = n down to 2
           return sum(2.0 / (j * (j - 1)) for j in range(2, n + 1))

       # P(a ancestors | k descendants coalesce, n total tips)
       # computed recursively
       mean = 0.0
       for a in range(2, n - k + 2):
           # Probability of a ancestors when subtree of size k merges
           p_a = _pr_ancestors(a, k, n)
           # Expected coalescence time given a lineages
           expected_time = 2.0 / (a * (a - 1))
           mean += p_a * expected_time

       return mean

   def _pr_ancestors(a, k, n):
       """Probability of a extant ancestors when subtree of size k coalesces.

       This follows Wiuf & Donnelly (1999). For a subtree of size k in a
       tree of n tips, the number of other lineages when k coalesces to 1
       ranges from 1 to n-k. So total ancestors a ranges from 2 to n-k+1.
       """
       if k == 2:
           # Special case: the pair coalesces when a-1 other lineages exist
           # at that time, so a total. This has a known distribution.
           # P(a | k=2, n) = (n-1) * C(n-2, a-2) * C(a-1, 1)
           #                / (C(n, 2) * product terms)
           # ... simplified via the recursion in Wiuf & Donnelly
           pass
       # In practice, tsdate computes this recursively using the relationship:
       # P(a | k, n) can be computed from P(a | k+1, n) using
       # binomial coefficient identities.
       # For educational purposes, here's a direct simulation approach:
       raise NotImplementedError(
           "See the recursive implementation below for the full computation."
       )


The Recursive Computation
---------------------------

The key to computing :math:`P(a \mid k, n)` efficiently is a **recursive
relationship** over decreasing :math:`k`. tsdate's implementation uses the
identity:

.. math::

   P(a \mid k, n) = \sum_{a'=a}^{n-k+1} P(a' \mid k+1, n) \cdot
   \frac{\binom{a'-1}{1}}{\binom{a'+1}{2}}

The base case is :math:`k = n-1`, where the subtree is the second-to-last
to coalesce, and there are exactly :math:`a = 2` ancestors.

In practice, tsdate precomputes a lookup table of :math:`(\text{mean}, \text{variance})`
indexed by :math:`k` (number of descendants), for a given :math:`n` (total tips).

Now let us translate this recursion into code. The implementation walks backward
from :math:`k = n-1` down to :math:`k = 2`, building the probability table one
row at a time.

.. code-block:: python

   def conditional_coalescent_moments(n, Ne=1.0):
       """Compute mean and variance of node age for all possible descendant counts.

       Parameters
       ----------
       n : int
           Total number of tips.
       Ne : float
           Effective population size.

       Returns
       -------
       moments : dict
           {k: (mean, variance)} for k = 2, 3, ..., n.
       """
       # Precompute unconditional coalescence time moments for a lineages
       # E[T | a] = 2/(a*(a-1)),  Var[T | a] = E[T|a]^2 = 4/(a*(a-1))^2
       max_a = n
       t_mean = np.zeros(max_a + 1)   # t_mean[a] = expected coalescence time given a lineages
       t_var = np.zeros(max_a + 1)    # t_var[a] = variance of coalescence time given a lineages
       for a in range(2, max_a + 1):
           rate = a * (a - 1) / 2.0   # coalescence rate with a lineages (num pairs)
           t_mean[a] = 1.0 / rate     # exponential mean = 1/rate
           t_var[a] = 1.0 / rate**2   # exponential variance = 1/rate^2

       # Build P(a | k, n) table recursively from k=n-1 down to k=2
       # Start: when k = n-1, there must be a=2 ancestors (only 2 lineages left)
       pr_a = {}
       pr_a[n-1] = np.zeros(max_a + 1)
       pr_a[n-1][2] = 1.0  # certain: exactly 2 ancestors

       for k in range(n - 2, 1, -1):
           pr_a[k] = np.zeros(max_a + 1)
           for a in range(2, n - k + 2):
               # Recursive formula from Wiuf & Donnelly
               for a_prime in range(a, n - k + 1):
                   # Transition probability from (k+1, a') to (k, a)
                   # depends on coalescent rates
                   if pr_a[k+1][a_prime] > 0:
                       transition = _transition_prob(a_prime, a)
                       pr_a[k][a] += pr_a[k+1][a_prime] * transition

           # Normalize to ensure probabilities sum to 1
           total = pr_a[k].sum()
           if total > 0:
               pr_a[k] /= total

       # Compute moments by averaging over a (law of total expectation/variance)
       moments = {}
       for k in range(2, n):
           mean = np.sum(pr_a[k] * t_mean)             # E[T] = sum_a P(a) * E[T|a]
           e_t_sq = np.sum(pr_a[k] * (t_var + t_mean**2))  # E[T^2] via law of total expectation
           variance = e_t_sq - mean**2                  # Var = E[T^2] - (E[T])^2
           moments[k] = (mean, variance)

       # Root (k=n): sum of all waiting times from n lineages down to 1
       root_mean = sum(2.0 / (j * (j - 1)) for j in range(2, n + 1))
       root_var = sum(4.0 / (j * (j - 1))**2 for j in range(2, n + 1))
       moments[n] = (root_mean, root_var)

       return moments

   def _transition_prob(a_prime, a):
       """Transition probability in the Wiuf-Donnelly recursion.

       Probability that when one more pair coalesces (decreasing k by 1),
       the number of ancestors changes from a' to a.
       """
       if a > a_prime or a < 2:
           return 0.0
       if a == a_prime:
           # The coalescing pair was entirely within the subtree
           # (no change in total ancestor count -- it was the subtree
           # that coalesced, reducing subtree lineages, but a new
           # subtree-root lineage appears)
           return (a_prime - 1) / (a_prime + 1)
       if a == a_prime - 1:
           # One of the coalescing lineages was in the subtree,
           # the other was not, reducing total ancestors by 1
           return 2.0 / (a_prime + 1)
       return 0.0


From Moments to Gamma Parameters
===================================

With the mean and variance of the conditional coalescent in hand, the next step
is to convert them into a form that the dating algorithm can use. tsdate takes
the mean and variance and fits a **gamma distribution** to them. This is the
prior for each node.

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
