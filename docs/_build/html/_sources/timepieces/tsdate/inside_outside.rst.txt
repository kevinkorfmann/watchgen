.. _tsdate_inside_outside:

=================================
Inside-Outside Belief Propagation
=================================

   *The first algorithm: pass messages up the tree, then back down, and every
   node knows its place in time.*

With the coalescent prior (:ref:`tsdate_coalescent_prior`) and the mutation
likelihood (:ref:`tsdate_mutation_likelihood`) in hand, we have both halves
of Bayes' rule. The challenge now is *combining* them. Each node's age
depends on the ages of its parents and children through the edge likelihoods,
creating a coupled system that cannot be solved node-by-node.

The inside-outside method is tsdate's original dating algorithm (Wohns et al.,
2022). It discretizes time into a grid, represents each node's posterior as a
probability vector over grid points, and propagates information through the tree
using two passes: **inside** (leaves to root) and **outside** (root to leaves).

This is the same algorithmic idea as the forward-backward algorithm for HMMs,
adapted to tree structures. In the watch metaphor, it is **messages flowing
through the gear train**: each gear tells its neighbors what time it thinks it
is, and after two complete sweeps (one upward, one downward), every gear has
heard from every other gear and settled into its calibrated position.

.. admonition:: Probability Aside -- Belief propagation on trees vs. graphs

   Belief propagation (BP) on a tree-shaped graphical model gives *exact*
   marginal distributions in exactly two passes. The inside pass collects
   evidence from leaves to root; the outside pass distributes it back. The
   algorithm is sometimes called the "sum-product algorithm." On a graph
   with loops (like a tree sequence, where nodes are shared across local
   trees), BP becomes *loopy BP* -- an approximation. Loopy BP has no
   guarantee of convergence or exactness, but in practice it works well for
   the sparse, tree-like graphs that tree sequences produce.


Step 1: Discretize Time
=========================

The first decision: what grid of timepoints to use?

tsdate creates a grid :math:`\mathbf{g} = (g_0, g_1, \ldots, g_{K-1})` spanning
from 0 to some maximum time. The grid can be:

- **Linear**: equally spaced in time
- **Logarithmic**: more resolution near the present, less in the deep past

Logarithmic is the default, because most nodes are relatively young and we want
fine resolution there.

.. code-block:: python

   import numpy as np

   def make_time_grid(n, Ne=1.0, num_points=20, grid_type="logarithmic"):
       """Create a time grid for the inside-outside algorithm.

       Parameters
       ----------
       n : int
           Number of samples (sets the expected TMRCA).
       Ne : float
           Effective population size.
       num_points : int
           Number of grid points.
       grid_type : str
           "linear" or "logarithmic".

       Returns
       -------
       grid : np.ndarray
           Array of timepoints, starting at 0.
       """
       # Expected TMRCA under standard coalescent: 2*Ne*(1 - 1/n)
       expected_tmrca = 2 * Ne * (1 - 1.0 / n)
       t_max = expected_tmrca * 4  # go well beyond expected TMRCA

       if grid_type == "linear":
           return np.linspace(0, t_max, num_points)
       else:
           # Log-spaced: more points near 0, fewer far out
           # Start from a small positive number to avoid log(0)
           t_min = t_max / (10 * num_points)
           return np.concatenate([[0], np.geomspace(t_min, t_max, num_points - 1)])

   # Example
   grid = make_time_grid(n=100, num_points=20)
   print(f"Grid: {grid[:5]} ... {grid[-3:]}")
   print(f"Grid spans [0, {grid[-1]:.2f}] with {len(grid)} points")


Step 2: The Likelihood Matrix
================================

For each edge :math:`e`, we need the likelihood of the observed mutations
:math:`m_e` as a function of the parent and child times. On the discrete grid,
this becomes a :math:`K \times K` **lower-triangular matrix** :math:`L_e`:

.. math::

   L_e[i, j] = P(m_e \mid t_{\text{parent}} = g_i, t_{\text{child}} = g_j)
   = \text{Poisson}(m_e; \lambda_e \cdot (g_i - g_j))

for :math:`i > j` (parent older than child), and :math:`L_e[i, j] = 0` otherwise.

This matrix is the discrete version of the bivariate edge factor :math:`\phi_e`
we met in the likelihood chapter. Each entry answers: "if the parent were at
grid point :math:`i` and the child at grid point :math:`j`, how likely are the
observed mutations?"

.. code-block:: python

   from scipy.stats import poisson

   def edge_likelihood_matrix(m_e, lambda_e, grid):
       """Compute the likelihood matrix for an edge on the time grid.

       Parameters
       ----------
       m_e : int
           Mutation count on this edge.
       lambda_e : float
           Span-weighted mutation rate (mu * span_bp).
       grid : np.ndarray
           Time grid.

       Returns
       -------
       L : np.ndarray, shape (K, K)
           L[i, j] = P(m_e | parent_time=grid[i], child_time=grid[j])
           Lower triangular (i >= j).
       """
       K = len(grid)
       L = np.zeros((K, K))

       for i in range(K):
           for j in range(i + 1):  # j <= i (child younger than parent)
               delta_t = grid[i] - grid[j]
               if delta_t > 0:
                   expected = lambda_e * delta_t       # Poisson mean
                   L[i, j] = poisson.pmf(m_e, expected)  # evaluate PMF
               elif m_e == 0:
                   # delta_t = 0, only possible if no mutations
                   L[i, j] = 1.0

       return L

   # Example
   grid = make_time_grid(n=50, num_points=10)
   L = edge_likelihood_matrix(m_e=2, lambda_e=0.001, grid=grid)
   print(f"Likelihood matrix shape: {L.shape}")
   print(f"Max likelihood at parent_idx, child_idx = {np.unravel_index(L.argmax(), L.shape)}")

.. admonition:: Storage optimization

   tsdate doesn't actually store full :math:`K \times K` matrices. Instead, it
   stores the lower triangle as a flattened 1D array of size :math:`K(K+1)/2`.
   This halves the memory requirement.


Step 3: The Inside Pass (Leaves to Root)
==========================================

Now we arrive at the heart of the algorithm. The inside pass computes, for
each node :math:`u`, the probability of all the data *below* :math:`u`,
conditioned on :math:`u`'s age:

.. math::

   \text{inside}(u, g_i) = P(\mathbf{D}_{\text{below } u} \mid t_u = g_i)

Think of this as each gear reporting upward: "given that I am at grid point
:math:`i`, here is the total evidence from everything below me." The messages
flow from the leaves (known time 0) up to the root, accumulating mutation
evidence along the way.

For **leaf nodes** (samples at time 0):

.. math::

   \text{inside}(\text{leaf}, g_i) = \begin{cases}
   1 & \text{if } g_i = 0 \\
   0 & \text{otherwise}
   \end{cases}

For **internal nodes**, the inside value combines information from all child
edges. If node :math:`u` has children :math:`v_1, v_2, \ldots` connected by
edges :math:`e_1, e_2, \ldots`:

.. math::

   \text{inside}(u, g_i) = \prod_{\text{child } v_c}
   \underbrace{\sum_{j=0}^{i} L_{e_c}[i, j] \cdot \text{inside}(v_c, g_j)}_{\text{message from child } v_c}

**Intuition**: For each child, sum over all possible child times (weighted by
the edge likelihood and the child's inside value), then multiply across
children. This is exactly the same logic as the forward algorithm in an HMM,
but on a tree instead of a chain.

.. admonition:: Calculus Aside -- Discrete marginalization

   The inner sum :math:`\sum_{j=0}^{i} L_e[i,j] \cdot \text{inside}(v, g_j)`
   is the discrete analogue of the integral
   :math:`\int_0^{t_u} \phi_e(t_u, t_v) \cdot q(t_v) \, dt_v` that we met
   in the likelihood chapter. On the grid, the integral becomes a
   matrix-vector product: multiply the likelihood matrix row by the child's
   inside vector, then sum. The product over children is the "product rule"
   for independent subtrees.

.. code-block:: python

   import numpy as np

   def inside_pass(ts, grid, mutation_rate, mut_per_edge):
       """Compute inside values for all nodes.

       Parameters
       ----------
       ts : tskit.TreeSequence
       grid : np.ndarray
           Time grid of K points.
       mutation_rate : float
       mut_per_edge : np.ndarray
           Mutation count per edge.

       Returns
       -------
       inside : np.ndarray, shape (num_nodes, K)
           inside[u, i] = P(data below u | t_u = grid[i]).
       """
       K = len(grid)
       inside = np.ones((ts.num_nodes, K))  # start at 1 (multiplicative identity)

       # Initialize leaves: delta at time 0
       for sample_id in ts.samples():
           inside[sample_id, :] = 0.0       # zero everywhere...
           inside[sample_id, 0] = 1.0       # ...except at grid point 0 (present)

       # Process edges from leaves to root (bottom-up)
       # We need a topological ordering: process children before parents
       # tsdate uses the edge table sorted by child time

       # Build adjacency: for each parent, collect (child, edge_id)
       children_of = {}
       for edge in ts.edges():
           if edge.parent not in children_of:
               children_of[edge.parent] = []
           children_of[edge.parent].append((edge.child, edge.id))

       # Topological order: process nodes with smallest time first
       node_order = sorted(range(ts.num_nodes),
                          key=lambda u: ts.node(u).time)

       for u in node_order:
           if u in ts.samples():
               continue  # already initialized

           if u not in children_of:
               continue

           for child_id, edge_id in children_of[u]:
               m_e = mut_per_edge[edge_id]
               edge = ts.edge(edge_id)
               span = edge.right - edge.left
               lambda_e = mutation_rate * span

               # Build the K x K likelihood matrix for this edge
               L = edge_likelihood_matrix(m_e, lambda_e, grid)

               # Message from child to parent:
               # msg[i] = sum_j L[i,j] * inside[child, j]
               msg = np.zeros(K)
               for i in range(K):
                   for j in range(i + 1):  # only j <= i (child younger than parent)
                       msg[i] += L[i, j] * inside[child_id, j]

               # Multiply into parent's inside value (product over children)
               inside[u, :] *= msg

           # Normalize to prevent underflow (does not change relative values)
           total = inside[u, :].sum()
           if total > 0:
               inside[u, :] /= total

       return inside


Step 4: The Outside Pass (Root to Leaves)
==========================================

With the inside pass complete, every node knows about the evidence below it.
But nodes also need evidence from *above* -- what do the parent, grandparent,
and sibling subtrees say? The outside pass sends this information downward.

The outside pass computes, for each node :math:`u`, the probability of all the
data *above* :math:`u`:

.. math::

   \text{outside}(u, g_i) = P(\mathbf{D}_{\text{above } u} \mid t_u = g_i)

For **root nodes**:

.. math::

   \text{outside}(\text{root}, g_i) = \text{prior}(\text{root}, g_i)

The prior comes from the conditional coalescent (Gear 1,
:ref:`tsdate_coalescent_prior`).

For **non-root nodes**, the outside value is computed by combining the parent's
outside value, the edge likelihood, and the inside values of *sibling* subtrees:

.. math::

   \text{outside}(v, g_j) = \sum_{i=j}^{K-1}
   L_e[i, j] \cdot \text{outside}(u, g_i) \cdot
   \prod_{\text{sibling } v' \neq v} \text{msg}_{v' \to u}(g_i)

**Intuition**: To know what the data above :math:`v` tells us about :math:`v`'s
age, we need:

1. The information from above the parent :math:`u` (the outside of :math:`u`)
2. The information from sibling subtrees (the inside messages from siblings)
3. The edge likelihood connecting :math:`u` to :math:`v`

In the gear train, the outside message is the force transmitted *downward*
from the mainspring (root) through the gear train. Each gear receives torque
from above (its parent's outside) modulated by the sibling gears' evidence
(their inside messages).

.. code-block:: python

   def outside_pass(ts, grid, mutation_rate, mut_per_edge, inside, prior_grid):
       """Compute outside values for all nodes.

       Parameters
       ----------
       ts : tskit.TreeSequence
       grid : np.ndarray
       mutation_rate : float
       mut_per_edge : np.ndarray
       inside : np.ndarray, shape (num_nodes, K)
       prior_grid : np.ndarray
           Prior for each node.

       Returns
       -------
       outside : np.ndarray, shape (num_nodes, K)
       """
       K = len(grid)
       outside = np.ones((ts.num_nodes, K))

       # Initialize roots with coalescent prior
       for u in range(ts.num_nodes):
           if is_root(ts, u):
               outside[u, :] = prior_grid[u]  # prior is the "outside" for the root

       # Process nodes from root to leaves (top-down -- oldest first)
       node_order = sorted(range(ts.num_nodes),
                          key=lambda u: -ts.node(u).time)  # oldest first

       # Build parent lookup
       parent_of = {}  # (child, edge_id) -> parent
       children_of = {}
       for edge in ts.edges():
           parent_of[(edge.child, edge.id)] = edge.parent
           if edge.parent not in children_of:
               children_of[edge.parent] = []
           children_of[edge.parent].append((edge.child, edge.id))

       for u in node_order:
           if u not in children_of:
               continue

           # Compute the "inside messages" from each child to u
           child_messages = {}
           for child_id, edge_id in children_of[u]:
               m_e = mut_per_edge[edge_id]
               edge = ts.edge(edge_id)
               span = edge.right - edge.left
               lambda_e = mutation_rate * span
               L = edge_likelihood_matrix(m_e, lambda_e, grid)

               # Standard inside message: sum over child times
               msg = np.zeros(K)
               for i in range(K):
                   for j in range(i + 1):
                       msg[i] += L[i, j] * inside[child_id, j]

               child_messages[(child_id, edge_id)] = msg

           # For each child, compute outside using parent outside
           # and all other children's messages (siblings)
           for child_id, edge_id in children_of[u]:
               m_e = mut_per_edge[edge_id]
               edge = ts.edge(edge_id)
               span = edge.right - edge.left
               lambda_e = mutation_rate * span
               L = edge_likelihood_matrix(m_e, lambda_e, grid)

               # Parent contribution: outside[u] * product of sibling messages
               parent_contrib = outside[u, :].copy()
               for other_child, other_eid in children_of[u]:
                   if other_eid != edge_id:
                       # Multiply in sibling's inside message
                       parent_contrib *= child_messages[(other_child, other_eid)]

               # Message from parent to child (downward):
               # msg[j] = sum_i L[i,j] * parent_contrib[i]
               msg = np.zeros(K)
               for j in range(K):
                   for i in range(j, K):  # i >= j (parent older than child)
                       msg[j] += L[i, j] * parent_contrib[i]

               outside[child_id, :] *= msg  # accumulate outside evidence

               # Normalize
               total = outside[child_id, :].sum()
               if total > 0:
                   outside[child_id, :] /= total

       return outside

   def is_root(ts, node_id):
       """Check if a node is a root (has no parent edges)."""
       for edge in ts.edges():
           if edge.child == node_id:
               return False
       return ts.node(node_id).time > 0


Step 5: Combine to Get the Posterior
======================================

With the inside and outside values computed, combining them is straightforward.
The marginal posterior for each node is the product of inside and outside,
weighted by the prior:

.. math::

   P(t_u = g_i \mid \mathbf{D}) \propto \text{inside}(u, g_i) \cdot \text{outside}(u, g_i)

This is the fundamental identity of the sum-product algorithm: the marginal
at a variable is the product of all evidence arriving from below (inside) and
all evidence arriving from above (outside).

.. code-block:: python

   def compute_posteriors(inside, outside):
       """Combine inside and outside to get marginal posteriors.

       Parameters
       ----------
       inside : np.ndarray, shape (num_nodes, K)
       outside : np.ndarray, shape (num_nodes, K)

       Returns
       -------
       posterior : np.ndarray, shape (num_nodes, K)
           posterior[u, :] is the marginal posterior distribution over
           grid points for node u.
       """
       posterior = inside * outside  # element-wise product

       # Normalize each node's posterior to sum to 1
       row_sums = posterior.sum(axis=1, keepdims=True)
       row_sums[row_sums == 0] = 1.0  # avoid division by zero
       posterior /= row_sums

       return posterior

   def posterior_mean(posterior, grid):
       """Compute posterior mean age for each node.

       Parameters
       ----------
       posterior : np.ndarray, shape (num_nodes, K)
       grid : np.ndarray, shape (K,)

       Returns
       -------
       means : np.ndarray, shape (num_nodes,)
           E[t_u | D] for each node.
       """
       return posterior @ grid  # weighted sum: sum_i posterior[u,i] * grid[i]


Why This Works: The Belief Propagation Guarantee
===================================================

On a **tree** (no loops), the inside-outside algorithm gives **exact** marginal
posteriors. This is a classical result from graphical models: belief propagation
on trees converges in exactly two passes.

But a tree *sequence* is not a tree. When a node appears in multiple local
trees, it creates loops in the factor graph. For example, if node :math:`u` is
the parent of :math:`v` in one genomic region and the grandparent of :math:`v`
in another, there are two paths between :math:`u` and :math:`v` -- a loop.

On loopy graphs, belief propagation is **approximate**. It may:

- Converge to a fixed point that's close to the true posterior (common in practice)
- Oscillate (rare for this type of graph)
- Over-count evidence from repeated paths (the main source of error)

tsdate mitigates this by processing edges in the tree sequence's natural
ordering, which respects the temporal structure and minimizes loop effects.

.. admonition:: Probability Aside -- Why loops cause trouble

   On a tree, each piece of evidence (each mutation on each edge) is counted
   exactly once in every node's posterior. On a graph with loops, messages
   can "circulate" around a loop: node A tells B, B tells C, C tells A
   what A originally said -- as if the same evidence were counted twice.
   This is called "double-counting" and it makes loopy BP an approximation.
   In tree sequences the loops arise because a single ancestor participates
   in different local trees. The loops are typically short (length 2 or 3),
   and empirically the approximation is good.


Log-Space Computation
========================

In practice, the inside and outside values can span many orders of magnitude.
tsdate performs all computations in **log space** to prevent underflow:

.. math::

   \log \text{inside}(u, g_i) = \sum_{\text{children}} \log \left(
   \sum_j \exp\left(\log L_e[i,j] + \log \text{inside}(v, g_j)\right)
   \right)

The inner log-sum-exp is computed using the standard numerical trick:

.. math::

   \log \sum_j e^{x_j} = x_{\max} + \log \sum_j e^{x_j - x_{\max}}

.. admonition:: Calculus Aside -- The log-sum-exp trick

   Naively computing :math:`\log(\sum_j e^{x_j})` can overflow (if any
   :math:`x_j` is very large) or underflow (if all :math:`x_j` are very
   negative). The trick: factor out :math:`e^{x_{\max}}` to get
   :math:`x_{\max} + \log(\sum_j e^{x_j - x_{\max}})`. Now every exponent
   is :math:`\leq 0`, preventing overflow, and at least one exponent is 0,
   preventing underflow. This is the single most important numerical trick
   in probabilistic computation, and it appears throughout tsdate.

.. code-block:: python

   from scipy.special import logsumexp

   def inside_pass_logspace(inside_log, L_log, K):
       """Compute a single inside message in log space.

       Parameters
       ----------
       inside_log : np.ndarray, shape (K,)
           Log inside values for child node.
       L_log : np.ndarray, shape (K, K)
           Log likelihood matrix.

       Returns
       -------
       msg_log : np.ndarray, shape (K,)
           Log message from child to parent.
       """
       msg_log = np.full(K, -np.inf)    # start at log(0) = -inf
       for i in range(K):
           terms = L_log[i, :i+1] + inside_log[:i+1]  # log(L * inside) = log(L) + log(inside)
           msg_log[i] = logsumexp(terms)               # log-sum-exp for numerical stability
       return msg_log


The Standardization Trick
===========================

tsdate also uses **standardization**: after each message computation, the
maximum value is subtracted. This keeps all values in a numerically safe range
without changing the relative proportions.

.. math::

   \tilde{f}(g_i) = f(g_i) - \max_i f(g_i)

In log space, this means :math:`\max_i \tilde{f}(g_i) = 0`.


Putting It All Together
=========================

Here's the complete inside-outside algorithm, assembling all the pieces from
above into a single pipeline.

.. code-block:: python

   def inside_outside_date(ts, mutation_rate, Ne=1.0, num_points=20):
       """Date a tree sequence using the inside-outside algorithm.

       Parameters
       ----------
       ts : tskit.TreeSequence
           Input tree sequence (topology from tsinfer).
       mutation_rate : float
           Per-bp per-generation mutation rate.
       Ne : float
           Effective population size.
       num_points : int
           Number of time grid points.

       Returns
       -------
       node_times : np.ndarray
           Posterior mean age for each node.
       """
       # Step 0: Setup -- build the time grid
       grid = make_time_grid(ts.num_samples, Ne, num_points)
       K = len(grid)

       # Count mutations per edge (used by both passes)
       mut_per_edge = np.zeros(ts.num_edges, dtype=int)
       for mut in ts.mutations():
           if mut.edge >= 0:
               mut_per_edge[mut.edge] += 1

       # Build prior for each node (from coalescent theory, Gear 1)
       prior = build_discrete_prior(ts, grid, Ne)

       # Step 1: Inside pass (leaves to root) -- evidence flows upward
       inside = inside_pass(ts, grid, mutation_rate, mut_per_edge)

       # Step 2: Outside pass (root to leaves) -- evidence flows downward
       outside = outside_pass(ts, grid, mutation_rate, mut_per_edge,
                              inside, prior)

       # Step 3: Combine inside and outside to get marginal posteriors
       posterior = compute_posteriors(inside, outside)

       # Step 4: Extract posterior means as point estimates
       node_times = posterior_mean(posterior, grid)

       # Fix leaf times at 0 (samples have known ages)
       for s in ts.samples():
           node_times[s] = 0.0

       return node_times

   def build_discrete_prior(ts, grid, Ne):
       """Build a discrete prior for each node on the time grid."""
       from scipy.stats import gamma

       K = len(grid)
       prior = np.ones((ts.num_nodes, K))

       for u in range(ts.num_nodes):
           if u in set(ts.samples()):
               # Sample nodes are fixed at time 0
               prior[u, :] = 0.0
               prior[u, 0] = 1.0
               continue

           # Count descendants (simplified: assume binary tree)
           k = 2
           mean = sum(2.0 / (j * (j - 1)) for j in range(2, k + 1))
           var = sum(4.0 / (j * (j - 1))**2 for j in range(2, k + 1))
           alpha = mean**2 / var          # gamma shape from method of moments
           beta_param = mean / var        # gamma rate from method of moments

           # Evaluate gamma pdf at grid points
           for i in range(K):
               if grid[i] > 0:
                   prior[u, i] = gamma.pdf(grid[i], a=alpha, scale=1.0/beta_param)
               else:
                   prior[u, i] = 0.0  # internal nodes can't be at time 0

           # Normalize to a proper probability distribution
           total = prior[u, :].sum()
           if total > 0:
               prior[u, :] /= total

       return prior


Limitations of Inside-Outside
================================

The inside-outside method works well but has some limitations that motivated
the development of the variational gamma method:

1. **Grid resolution**: The posterior is only as fine as the grid. With
   :math:`K=20` points, you can't distinguish between times that fall in the
   same grid cell.

2. **Quadratic per edge**: Computing the likelihood matrix is :math:`O(K^2)`.
   For large :math:`K`, this becomes expensive.

3. **Loopy BP**: On tree sequences with many shared nodes, the approximation
   may degrade.

4. **No natural way to handle constraints**: Enforcing :math:`t_u > t_v` on
   the grid requires zeroing out entries, which can lose probability mass.

These limitations motivated the development of the **variational gamma method**
(:ref:`tsdate_variational_gamma`), which works in continuous time and avoids
the grid entirely. Instead of a probability vector of :math:`K` values per
node, it stores just two numbers (:math:`\alpha`, :math:`\beta`), and instead
of matrix-vector products, it uses moment matching -- a fundamentally different
(and faster) way of passing messages through the gear train.


Summary
========

The inside-outside algorithm dates nodes by:

1. **Discretizing** time into a grid of :math:`K` points
2. **Inside pass**: propagating mutation likelihoods upward from leaves to roots
3. **Outside pass**: propagating prior and sibling information downward
4. **Combining**: multiplying inside and outside to get marginal posteriors

The key equations:

.. math::

   \text{inside}(u, g_i) = \prod_{\text{children}} \sum_j L_e[i,j] \cdot \text{inside}(v, g_j)

.. math::

   \text{outside}(v, g_j) = \sum_i L_e[i,j] \cdot \text{outside}(u, g_i) \cdot \prod_{\text{siblings}} \text{msg}(g_i)

.. math::

   P(t_u = g_i \mid \mathbf{D}) \propto \text{inside}(u, g_i) \cdot \text{outside}(u, g_i)

In the watch metaphor, the inside pass is like winding the mainspring from the
bottom -- evidence accumulates upward from the leaves. The outside pass
releases that energy back down through the gear train. After both passes,
every gear (node) has felt the full tension of the data from every direction,
and its position (age) is set.

Next: the modern default method, variational gamma, which replaces the grid with
continuous gamma approximations (:ref:`tsdate_variational_gamma`).
