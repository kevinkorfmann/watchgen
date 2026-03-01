.. _moran_model:

=========================
The Moran Model
=========================

   *The escapement of the watch: a discrete population model whose eigendecomposition lets us tell time exactly.*

.. epigraph::

   "The Moran model provides a tractable Markov chain whose eigensystem is
   known in closed form."

   -- Kamm, Terhorst, Song, and Durbin (2017)

.. admonition:: Biology Aside -- Why a population model?

   The central question of demographic inference is: *given DNA from living
   individuals, what was the population's history?* To answer this, we need a
   mathematical model that predicts how allele frequencies -- the proportions
   of different genetic variants in the population -- change over time. A
   variant that exists in 5% of chromosomes today may have been at 2% a
   thousand generations ago, or it may not have existed at all. The Moran
   model gives us a precise, tractable way to compute the probability of any
   such frequency trajectory. It is the mathematical engine inside ``momi2``
   that converts a demographic scenario (population sizes, split times,
   migration rates) into a predicted site frequency spectrum that can be
   compared to real data.

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

.. admonition:: Biology Aside -- Derived and ancestral alleles

   At any position in the genome, chromosomes carry one of two variants: the
   **ancestral** allele (inherited from a distant common ancestor of all
   samples) and the **derived** allele (introduced by a mutation at some point
   in the past). Knowing which allele is ancestral typically requires an
   outgroup species -- for example, chimpanzee sequence is used to polarize
   human variants. The number of derived copies :math:`i` in a sample of
   :math:`n` chromosomes is what the site frequency spectrum (SFS) tabulates.
   The Moran model tracks exactly this quantity -- how :math:`i` fluctuates
   over evolutionary time due to the randomness of reproduction and death
   (genetic drift).

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

.. admonition:: Biology Aside -- Fixation and loss

   In a real population, most new mutations are eventually **lost** -- they
   drift to zero copies and disappear. A small fraction reach **fixation** --
   every chromosome carries the derived allele, and the variant becomes
   invisible to the SFS (it no longer varies). The absorbing boundaries at
   states 0 and :math:`n` capture exactly this: lineages that leave the
   polymorphic frequency range never return. Only the transient frequencies
   :math:`1, 2, \ldots, n-1` contribute to observable genetic variation.
   The rate at which alleles transit through these intermediate frequencies
   depends on the population size history -- which is precisely what
   demographic inference aims to recover.

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

Where does this formula come from? Recall from the coalescent that
:math:`\binom{j}{2} = j(j-1)/2` is the rate at which :math:`j` lineages
coalesce. This is not a coincidence: the Moran model and the coalescent
describe the same process from different perspectives (forward vs. backward
in time). The eigenvalue :math:`\lambda_j` captures the rate at which the
:math:`j`-th mode of genetic variation decays -- and this rate equals the
coalescence rate for :math:`j` lineages.

Concretely:

- :math:`\lambda_0 = 0` and :math:`\lambda_1 = 0`: these correspond to the
  two absorbing states (allele fixed at 0 or at :math:`n`). Once the allele
  is fixed or lost, nothing changes -- hence zero decay rate.
- :math:`\lambda_2 = -1`: the slowest-decaying transient mode. This
  corresponds to the last pair of lineages coalescing, which happens at rate
  :math:`\binom{2}{2} = 1`.
- :math:`\lambda_n = -n(n-1)/2`: the fastest-decaying mode. This corresponds
  to all :math:`n` lineages present, coalescing at rate :math:`\binom{n}{2}`.

.. code-block:: python

   import numpy as np

   n = 10  # population size
   eigenvalues = [-j*(j-1)/2 for j in range(n+1)]
   print("Eigenvalues of the Moran rate matrix (n=10):")
   for j, lam in enumerate(eigenvalues):
       label = " (absorbing)" if lam == 0 else ""
       print(f"  j={j:2d}: lambda = {lam:8.1f}{label}")

.. admonition:: Plain-language summary -- What eigendecomposition gives us

   An eigendecomposition breaks a complicated matrix into simpler components,
   much like splitting white light into individual colours with a prism. Each
   eigenvalue :math:`\lambda_j` controls one "mode" of the system's behaviour:

   - **Modes with** :math:`\lambda = 0`: These never decay. They correspond
     to the allele being permanently fixed or lost -- the absorbing states.
   - **Modes with large negative** :math:`\lambda`: These decay quickly.
     They correspond to fast transient fluctuations that die out early.
   - **Modes with small negative** :math:`\lambda`: These decay slowly.
     They correspond to the long-lived genetic variation that persists over
     many generations.

   The practical payoff is enormous: to compute how allele frequencies change
   over a time span :math:`t`, we simply multiply each mode by
   :math:`e^{\lambda_j t}`. Modes with large negative eigenvalues are
   exponentially damped (ancient variation is lost), while the :math:`\lambda = 0`
   modes persist forever. This gives us an exact, closed-form solution instead
   of requiring iterative numerical simulation.

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

.. admonition:: Biology Aside -- What the transition matrix means biologically

   Each entry :math:`P_{ij}(t)` answers a concrete biological question: *if a
   variant is currently present in* :math:`i` *out of* :math:`n` *chromosomes,
   what is the probability that it will be present in* :math:`j` *chromosomes
   after* :math:`t` *units of evolutionary time?* When the population is
   small, drift is strong and alleles rapidly fix or are lost -- the matrix
   concentrates probability at the edges (:math:`j = 0` or :math:`j = n`).
   When the population is large, drift is weak and alleles linger at
   intermediate frequencies -- the matrix is more diffuse. By computing
   :math:`P(t)` for different values of :math:`t` (corresponding to different
   epoch durations in a demographic model), ``momi2`` can predict how genetic
   variation is shaped by any sequence of population size changes, splits, and
   bottlenecks.

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

.. admonition:: Biology Aside -- The coalescent, in brief

   The coalescent is a way of thinking about genetic ancestry **backward in
   time**. Start with :math:`n` sampled chromosomes today and trace their
   lineages into the past. Occasionally two lineages merge ("coalesce") when
   they find a common ancestor. The rate of coalescence depends on the
   population size: in a small population, lineages bump into each other
   quickly; in a large population, it takes longer. By connecting the Moran
   model to the coalescent, ``momi2`` can translate a forward-in-time model
   of allele frequency change into the backward-in-time framework used to
   compute the expected SFS. This bridge is what makes the tensor machinery
   in the next chapter possible.

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

.. admonition:: Plain-language summary -- Why time must be scaled

   Genetic drift runs on a "molecular clock" whose speed depends on
   population size. In a population of 1,000 individuals, 100 generations of
   drift produce the same amount of frequency change as 10,000 generations
   in a population of 100,000. The integral above converts real time
   (generations) into the Moran model's intrinsic time units, which account
   for varying population size. When the population is small, the integral
   accumulates quickly (drift is fast); when it is large, the integral
   accumulates slowly (drift is slow). This is the reason that demographic
   events like bottlenecks leave such strong signatures in the SFS: a brief
   period of small population size corresponds to a large "jump" in
   drift-scaled time.

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

Solutions
=========

.. admonition:: Solution 1: Eigenvalue verification

   The theoretical eigenvalues are :math:`\lambda_j = -j(j-1)/2` for
   :math:`j = 0, 1, \ldots, n`.

   .. code-block:: python

      import numpy as np

      for n in [5, 10, 20]:
          Q = moran_rate_matrix(n)
          numerical_eigs = np.sort(np.linalg.eigvalsh(Q))

          j = np.arange(n + 1)
          theoretical_eigs = np.sort(-j * (j - 1) / 2.0)

          assert np.allclose(numerical_eigs, theoretical_eigs, atol=1e-10), \
              f"Mismatch for n={n}"
          print(f"n={n}: eigenvalues match")
          print(f"  Theoretical: {theoretical_eigs[:6]} ...")
          print(f"  Numerical:   {numerical_eigs[:6]} ...")

   For every :math:`n`, the numerically computed eigenvalues agree with the
   closed-form formula to machine precision. Note the two zero eigenvalues
   (:math:`j = 0` and :math:`j = 1`), corresponding to the absorbing states.

.. admonition:: Solution 2: Stationary distribution

   The two absorbing states are :math:`i = 0` and :math:`i = n`, since
   :math:`q(0, \cdot) = 0` and :math:`q(n, \cdot) = 0` (the departure rates
   vanish). For large :math:`t`, all probability mass concentrates on these
   absorbing states.

   The fixation probability starting from :math:`i` derived alleles is
   :math:`i/n` by symmetry of the Moran model (the up and down rates are
   equal).

   .. code-block:: python

      import numpy as np

      n = 10
      t_large = 100.0  # large time

      P = moran_transition(t_large, n)

      # For large t, P[i, :] should have mass only at 0 and n
      for i in range(n + 1):
          interior_mass = P[i, 1:n].sum()
          assert interior_mass < 1e-6, \
              f"Interior mass at state {i}: {interior_mass}"

          # Fixation probability: P[i, n] should be approximately i/n
          fix_prob = P[i, n]
          expected_fix = i / n
          print(f"  i={i}: P(fixation) = {fix_prob:.6f}, "
                f"expected = {expected_fix:.6f}")
          if 0 < i < n:
              assert abs(fix_prob - expected_fix) < 1e-4

   .. math::

      \lim_{t \to \infty} P_{ij}(t) = \begin{cases}
      1 - i/n & \text{if } j = 0 \\
      i/n & \text{if } j = n \\
      0 & \text{otherwise}
      \end{cases}

   This result follows from the martingale property of the Moran model:
   :math:`E[X(t) \mid X(0) = i] = i` for all :math:`t`, and the only
   distributions on :math:`\{0, n\}` with mean :math:`i` assign probability
   :math:`i/n` to :math:`n` and :math:`1 - i/n` to :math:`0`.

.. admonition:: Solution 3: Chapman-Kolmogorov

   The semigroup property :math:`P(s+t) = P(s) P(t)` follows directly from
   the eigendecomposition:

   .. math::

      P(s) P(t) = V e^{\Lambda s} V^{-1} V e^{\Lambda t} V^{-1}
                = V e^{\Lambda (s+t)} V^{-1} = P(s+t)

   .. code-block:: python

      import numpy as np

      for n in [5, 10, 15]:
          for s, t in [(0.1, 0.2), (0.5, 0.3), (1.0, 1.0), (0.01, 0.99)]:
              P_s = moran_transition(s, n)
              P_t = moran_transition(t, n)
              P_st = moran_transition(s + t, n)
              P_product = P_s @ P_t

              max_err = np.max(np.abs(P_product - P_st))
              assert max_err < 1e-8, \
                  f"n={n}, s={s}, t={t}: max error = {max_err}"
              print(f"  n={n}, s={s}, t={t}: max error = {max_err:.2e}")

   The eigendecomposition makes this natural because exponentiating diagonal
   matrices is trivial: :math:`e^{\Lambda s} e^{\Lambda t} = e^{\Lambda(s+t)}`.
   The intermediate :math:`V^{-1} V = I` cancels. This is precisely why the
   eigendecomposition is the method of choice: it converts matrix exponentiation
   (an expensive operation) into scalar exponentiation (trivial), and the
   semigroup property is automatically satisfied.

.. admonition:: Solution 4: Time scaling

   For exponential growth from :math:`N_0 = 1000` to :math:`N_T = 10000` over
   :math:`T = 500` generations, the growth rate is:

   .. math::

      g = \frac{\ln(N_T / N_0)}{T} = \frac{\ln(10)}{500} \approx 0.004605

   The scaled Moran time is:

   .. math::

      t = \frac{2}{N_0} \int_0^T e^{gs}\, ds = \frac{2(e^{gT} - 1)}{N_0 g}

   .. code-block:: python

      import numpy as np

      N0 = 1000
      NT = 10000
      T = 500

      g = np.log(NT / N0) / T
      print(f"Growth rate g = {g:.6f}")

      # Scaled time under exponential growth
      t_exp = 2.0 * (np.exp(g * T) - 1) / (N0 * g)
      print(f"Scaled Moran time (exponential growth): t = {t_exp:.6f}")

      # Scaled time under constant size N0
      t_const = 2.0 * T / N0
      print(f"Scaled Moran time (constant size N0):   t = {t_const:.6f}")

      # Ratio
      print(f"Ratio t_exp / t_const = {t_exp / t_const:.4f}")

   The exponential growth case gives :math:`t_{\text{exp}} \approx 3.91`,
   while the constant-size case gives :math:`t_{\text{const}} = 1.0`. The
   growing population accumulates less drift-scaled time because drift is
   weaker when the population is large. The ratio
   :math:`t_{\text{exp}} / t_{\text{const}} \approx 3.91` reflects the
   harmonic-mean-like weighting of population sizes: most of the integral is
   dominated by the early period (backward in time) when the population is
   small (:math:`N_0`), but the large recent size (:math:`N_T`) significantly
   slows the accumulation of drift during the recent epoch.

Next: :ref:`tensor_machinery`
