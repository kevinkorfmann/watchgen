.. _moran_model:

=========================
The Moran Model
=========================

   *The escapement of the watch: a discrete population model whose eigendecomposition lets us tell time exactly.*

.. epigraph::

   "The Moran model provides a tractable Markov chain whose eigensystem is
   known in closed form."

   -- Kamm, Terhorst, Song, and Durbin (2017)

Step 1: The Moran Model as a Continuous-Time Markov Chain
==========================================================

The **Moran model** describes how allele frequencies change in a finite
population of size :math:`n`. Unlike the Wright-Fisher model (which has
discrete, non-overlapping generations), the Moran model operates in continuous
time: at rate proportional to :math:`n`, one individual is chosen to reproduce
and one to die.

The state of the system is the number of derived alleles :math:`i \in \{0, 1,
\ldots, n\}`. The transition rates are:

.. math::

   q(i, i+1) &= \frac{i(n-i)}{2} \quad \text{(one more derived copy)}

   q(i, i-1) &= \frac{i(n-i)}{2} \quad \text{(one fewer derived copy)}

   q(i, i) &= -i(n-i) \quad \text{(total departure rate)}

The factor :math:`i(n-i)/2` counts the number of ways to pick one derived and
one ancestral individual (divided by 2 because either could be the parent).

.. admonition:: Probability Aside -- Moran vs. Wright-Fisher

   The Moran model and the Wright-Fisher model are different discrete models
   that converge to the same diffusion limit as :math:`n \to \infty`. The key
   advantage of the Moran model for ``momi2`` is that its rate matrix has a
   known eigendecomposition, which allows exact computation of transition
   probabilities for any time :math:`t`. The Wright-Fisher model, being a
   discrete-time chain with binomial transitions, does not have such a clean
   eigensystem.

Step 2: The Rate Matrix
=========================

The rate matrix :math:`Q` is an :math:`(n+1) \times (n+1)` tridiagonal matrix:

.. math::

   Q = \begin{pmatrix}
   0 & 0 & & & \\
   0 \cdot n/2 & -0 \cdot n & 0 \cdot n/2 & & \\
    & 1(n-1)/2 & -1(n-1) & 1(n-1)/2 & \\
    & & \ddots & \ddots & \ddots \\
    & & & 0 & 0
   \end{pmatrix}

States 0 (all ancestral) and :math:`n` (all derived) are **absorbing** -- their
rows are all zeros. This makes sense: once an allele fixes or is lost, there
is no further change.

.. code-block:: python

   import numpy as np
   from scipy import sparse

   def moran_rate_matrix(n):
       """Construct the Moran model rate matrix for sample size n.

       Returns a (n+1) x (n+1) tridiagonal matrix Q where:
       - Q[i, i+1] = i*(n-i)/2   (gain one derived copy)
       - Q[i, i-1] = i*(n-i)/2   (lose one derived copy)
       - Q[i, i]   = -i*(n-i)    (total departure rate)
       """
       i = np.arange(n + 1, dtype=float)
       off_diag = i * (n - i) / 2.0
       diag = -2.0 * off_diag
       Q = (np.diag(off_diag[:-1], k=1)
            + np.diag(diag, k=0)
            + np.diag(off_diag[1:], k=-1))
       return Q

.. code-block:: python

   # Verification: rows sum to zero (a property of all rate matrices)
   n = 10
   Q = moran_rate_matrix(n)
   row_sums = Q.sum(axis=1)
   assert np.allclose(row_sums, 0), f"Row sums: {row_sums}"

   # Verification: Q is symmetric (detailed balance for the Moran model)
   assert np.allclose(Q, Q.T), "Rate matrix should be symmetric"

Step 3: Eigendecomposition
===========================

Because :math:`Q` is a real symmetric tridiagonal matrix, it has a complete set
of real eigenvalues and orthogonal eigenvectors:

.. math::

   Q = V \Lambda V^{-1}

where :math:`\Lambda = \text{diag}(\lambda_0, \lambda_1, \ldots, \lambda_n)` and
:math:`V` is the matrix of eigenvectors.

The eigenvalues of the Moran rate matrix are:

.. math::

   \lambda_j = -\frac{j(j-1)}{2}, \quad j = 0, 1, \ldots, n

Note that :math:`\lambda_0 = 0` and :math:`\lambda_1 = 0`, corresponding to the
two absorbing states (fixation and loss). All other eigenvalues are negative,
meaning the system decays toward absorption.

.. code-block:: python

   def moran_eigensystem(n):
       """Compute the eigendecomposition of the Moran rate matrix.

       Returns (V, eigenvalues, V_inv) where Q = V @ diag(eigenvalues) @ V_inv.
       """
       Q = moran_rate_matrix(n)
       eigenvalues, V = np.linalg.eigh(Q)  # eigh for symmetric matrices
       V_inv = np.linalg.inv(V)
       return V, eigenvalues, V_inv

.. code-block:: python

   # Verification: eigenvalues match the theoretical formula
   n = 10
   V, eigs, V_inv = moran_eigensystem(n)
   j = np.arange(n + 1)
   theoretical_eigs = -j * (j - 1) / 2.0
   # sort both to compare
   assert np.allclose(np.sort(eigs), np.sort(theoretical_eigs), atol=1e-10)

   # Verification: reconstruct Q from eigensystem
   Q_reconstructed = V @ np.diag(eigs) @ V_inv
   Q_original = moran_rate_matrix(n)
   assert np.allclose(Q_reconstructed, Q_original, atol=1e-10)

Step 4: The Transition Matrix
===============================

The transition probability matrix for time :math:`t` is the matrix exponential:

.. math::

   P(t) = e^{Qt} = V \, e^{\Lambda t} \, V^{-1} = V \, \text{diag}(e^{\lambda_0 t}, e^{\lambda_1 t}, \ldots, e^{\lambda_n t}) \, V^{-1}

Entry :math:`P_{ij}(t)` gives the probability that the number of derived alleles
transitions from :math:`i` to :math:`j` in time :math:`t`.

This is the core computational advantage of the Moran model: instead of
numerically integrating an ODE system (as ``moments`` does) or discretizing a
PDE on a grid (as ``dadi`` does), ``momi2`` computes exact transition
probabilities with a single matrix multiplication. The eigendecomposition is
computed once per sample size and cached; applying it for different times
:math:`t` requires only exponentiating the eigenvalues.

.. code-block:: python

   def moran_transition(t, n):
       """Compute the Moran transition matrix P(t) = exp(Q*t).

       t: time (in Moran model units, scaled by 2/N)
       n: sample size

       Returns (n+1) x (n+1) transition probability matrix.
       """
       V, eigs, V_inv = moran_eigensystem(n)
       D = np.diag(np.exp(t * eigs))
       P = V @ D @ V_inv
       # clamp small numerical errors
       P = np.clip(P, 0, None)
       P = P / P.sum(axis=1, keepdims=True)  # normalize rows
       return P

.. code-block:: python

   # Verification: P(0) is the identity matrix
   n = 10
   P0 = moran_transition(0, n)
   assert np.allclose(P0, np.eye(n + 1), atol=1e-10)

   # Verification: rows of P(t) sum to 1
   P1 = moran_transition(1.0, n)
   assert np.allclose(P1.sum(axis=1), 1.0, atol=1e-10)

   # Verification: all entries non-negative
   assert np.all(P1 >= -1e-15)

   # Verification: P(s+t) = P(s) @ P(t) (Chapman-Kolmogorov)
   P_s = moran_transition(0.5, n)
   P_t = moran_transition(0.3, n)
   P_st = moran_transition(0.8, n)
   assert np.allclose(P_s @ P_t, P_st, atol=1e-8)

Step 5: Connecting to the Coalescent
======================================

How does the Moran transition matrix relate to the coalescent? The connection
is through **lineage counting**.

Consider :math:`n` sampled chromosomes in a population of size :math:`N`. Going
backward in time, we track how many distinct lineages remain. The Moran model
(running backward) describes how the configuration of alleles in the sample
evolves as we trace back through the population history.

Specifically, the transition matrix :math:`P(t)` applied to a likelihood vector
:math:`\mathbf{v}` of length :math:`n+1` computes:

.. math::

   [\mathbf{v} \cdot P(t)]_i = \sum_j v_j \, P_{ji}(t)

This transforms the likelihood of observing a given allele configuration at
the bottom of an epoch to the likelihood at the top, accounting for all possible
coalescence and drift events that could have occurred during the epoch.

The **time scaling** is crucial. Real time :math:`T` (in generations) maps to
Moran model time via:

.. math::

   t = \int_0^T \frac{2}{N(s)}\, ds

For constant size: :math:`t = 2T/N`. For exponential growth with rate :math:`g`
and bottom size :math:`N_0`:

.. math::

   t = \frac{2}{N_0} \int_0^T e^{gs}\, ds = \frac{2(e^{gT} - 1)}{N_0 g}

Step 6: Applying Moran Transitions to Tensors
===============================================

In ``momi2``'s tensor framework, the Moran transition is applied to a
**specific axis** of a multi-dimensional likelihood tensor. If the tensor has
one axis per population, and we need to apply the Moran transition for
population :math:`k`, we multiply along axis :math:`k`:

.. math::

   L'_{i_1, \ldots, i_k, \ldots, i_D} = \sum_{j_k} L_{i_1, \ldots, j_k, \ldots, i_D} \, P_{j_k, i_k}(t)

This is an ``einsum`` operation, which ``momi2`` implements via optimized
batched matrix multiplication:

.. code-block:: python

   def moran_action(t, tensor, axis):
       """Apply Moran transition matrix to a tensor along a given axis.

       t: scaled time for this epoch
       tensor: multi-dimensional likelihood tensor
       axis: which axis (population) to apply the transition to

       Returns tensor with the Moran transition applied.
       """
       n = tensor.shape[axis] - 1
       P = moran_transition(t, n)
       # einsum: contract axis of tensor with P
       return np.tensordot(tensor, P.T, axes=([axis], [0]))

.. admonition:: Calculus Aside -- Why eigendecomposition beats ODE integration

   ``moments`` computes the SFS by integrating a system of ODEs forward in
   time: :math:`d\phi/dt = A \phi`. For each epoch, this requires calling an
   ODE solver (like RK45), which takes many small steps. ``momi2`` instead
   computes :math:`\phi(t) = e^{At} \phi(0)` directly via the eigensystem.
   Since the eigensystem is computed once and cached, each epoch requires only
   :math:`O(n^2)` work for the matrix-vector product. This makes ``momi2``
   particularly efficient when the same sample size appears in many epochs.

Step 7: The Moran Model in momi2's Source Code
================================================

In the actual ``momi2`` implementation, the Moran eigensystem is memoized
(cached by sample size :math:`n`) and the transition matrix computation handles
edge cases for numerical stability:

.. code-block:: python

   # Simplified from momi/moran_model.py

   from functools import lru_cache

   @lru_cache(maxsize=None)
   def cached_eigensystem(n):
       """Memoized eigendecomposition -- computed once per sample size."""
       Q = moran_rate_matrix(n)
       eigenvalues, V = np.linalg.eig(Q)
       V_inv = np.linalg.inv(V)
       return V, eigenvalues, V_inv

   def transition_with_checks(t, n):
       """Transition matrix with probability validation."""
       V, eigs, V_inv = cached_eigensystem(n)
       D = np.diag(np.exp(t * eigs))
       P = V @ D @ V_inv
       # numerical cleanup: clamp negatives, normalize rows
       P = np.maximum(P, 0)
       row_sums = P.sum(axis=1, keepdims=True)
       P = P / row_sums
       return P

The memoization is critical for performance: in a typical inference run, the
optimizer evaluates hundreds of parameter combinations, but the sample sizes
remain fixed. The eigensystem is computed once; only the time argument :math:`t`
changes between evaluations.

Exercises
=========

.. admonition:: Exercise 1: Eigenvalue verification

   Compute the eigenvalues of the Moran rate matrix for :math:`n = 5, 10, 20`
   and verify they match the formula :math:`\lambda_j = -j(j-1)/2`.

.. admonition:: Exercise 2: Stationary distribution

   Show that the Moran model has two absorbing states (0 and :math:`n`). For
   a large time :math:`t`, what does :math:`P(t)` converge to? What is the
   probability of fixation starting from :math:`i` derived alleles?

.. admonition:: Exercise 3: Chapman-Kolmogorov

   Verify the semigroup property :math:`P(s+t) = P(s) P(t)` numerically for
   several values of :math:`s, t` and :math:`n`. Why does this property make
   the eigendecomposition approach so natural?

.. admonition:: Exercise 4: Time scaling

   For a population that grows exponentially from :math:`N_0 = 1000` to
   :math:`N_T = 10000` over :math:`T = 500` generations, compute the scaled
   Moran time :math:`t`. How does this compare to the constant-size case
   :math:`t = 2T/N_0`?

Next: :ref:`tensor_machinery`
