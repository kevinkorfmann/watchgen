.. _haploid_algorithms:

=========================
Haploid LS HMM Algorithms
=========================

   *The gear train: turning the model into answers.*

In the :ref:`previous chapter <copying_model>`, we built the gears of the
template mechanism: transition probabilities, emission probabilities, and the
:math:`O(n)` trick. Now we assemble them into a working gear train -- the three
core algorithms that turn the Li & Stephens model into actionable results:

1. **Forward algorithm** -- computes the data likelihood and forward probabilities
2. **Backward algorithm** -- computes backward probabilities for posterior decoding
3. **Viterbi algorithm** -- finds the single most likely copying path (the most
   likely gear sequence through which the mechanism ticked)

Each algorithm is derived step by step, implemented in Python, and verified.

.. note::

   **Prerequisites.** This chapter builds directly on two earlier chapters:

   - The :ref:`HMM chapter <hmms>`, where the forward, backward, and Viterbi
     algorithms were introduced in their general form for arbitrary HMMs.
     We assume you are comfortable with the concepts of forward variables,
     backward variables, scaling, and traceback.
   - The :ref:`copying model chapter <copying_model>`, where we derived the
     Li-Stephens transition and emission probabilities and the :math:`O(n)`
     trick. We will use these results directly.

   If any of these concepts are unfamiliar, reviewing those chapters first
   will make this one much easier to follow.


The Forward Algorithm
======================

We already saw the forward algorithm in the :ref:`HMM chapter <hmms>` and used
the Li-Stephens :math:`O(n)` trick in the :ref:`previous chapter <copying_model>`.
Here we implement the complete, production-quality version with scaling.

Recap of the forward recursion
---------------------------------

The forward variable :math:`\alpha_j(\ell) = P(X_1, \ldots, X_\ell, Z_\ell = j)`
satisfies:

**Initialization:**

.. math::

   \alpha_j(1) = \frac{1}{n} \cdot e_j(X_1)

**Recursion** (using the Li-Stephens structure):

.. math::

   \alpha_j(\ell) = e_j(X_\ell) \left[(1 - r_\ell) \alpha_j(\ell - 1) + \frac{r_\ell}{n_\ell} \sum_{i} \alpha_i(\ell - 1)\right]

where :math:`n_\ell` is the number of copiable entries at site :math:`\ell`.

.. admonition:: Probability Aside: What the Forward Variable Represents

   The forward variable :math:`\alpha_j(\ell)` is a **joint probability**:
   the probability of seeing the observations :math:`X_1, \ldots, X_\ell`
   *and* being in state :math:`j` at site :math:`\ell`. It is *not* a
   conditional probability -- it does not condition on the observations.
   Summing over all states gives the marginal probability of the
   observations up to site :math:`\ell`:
   :math:`P(X_1, \ldots, X_\ell) = \sum_j \alpha_j(\ell)`. At the last site,
   this sum gives the full data likelihood. This distinction matters because
   it explains why forward probabilities become vanishingly small for long
   sequences -- they are products of many probabilities less than 1.

With scaling (normalized)
----------------------------

As discussed in the :ref:`HMM chapter <hmms>`, the forward probabilities become
astronomically small for long sequences. The solution: normalize at each step.

After normalization, :math:`\sum_j \hat{\alpha}_j(\ell) = 1`, and the recursion
simplifies to:

.. math::

   \tilde{\alpha}_j(\ell) = e_j(X_\ell) \left[(1 - r_\ell) \hat{\alpha}_j(\ell-1) + \frac{r_\ell}{n_\ell}\right]

.. math::

   c_\ell = \sum_j \tilde{\alpha}_j(\ell), \qquad
   \hat{\alpha}_j(\ell) = \frac{\tilde{\alpha}_j(\ell)}{c_\ell}

The log-likelihood is recovered as:

.. math::

   \log_{10} P(X_1, \ldots, X_m) = \sum_{\ell=1}^m \log_{10} c_\ell

**Why does the** :math:`\sum_i` **disappear?** Because :math:`\sum_i \hat{\alpha}_i = 1`
after normalization. This is why the normalized version is actually simpler to
code -- one fewer thing to compute.

.. admonition:: Probability Aside: Scaling and Numerical Stability

   Without scaling, the forward probabilities at site :math:`\ell` are
   roughly :math:`O(\epsilon^\ell)` where :math:`\epsilon` is a typical
   emission probability (say 0.01 for a mismatch). After a few hundred
   sites, these values underflow to zero in floating-point arithmetic.
   Scaling rescues us by dividing by the sum at each step. The log-likelihood
   is then accumulated as a sum of log-scaling-factors, which stays in a
   numerically well-behaved range. This technique is standard in HMM
   implementations and was introduced in the :ref:`HMM chapter <hmms>`.

Let's look at the complete implementation, matching the structure of the lshmm
library:

.. code-block:: python

   import numpy as np

   def forwards_ls_hap(n, m, H, s, emission_matrix, r, norm=True):
       """Forward algorithm for the haploid Li-Stephens model.

       Parameters
       ----------
       n : int
           Number of reference haplotypes.
       m : int
           Number of sites.
       H : ndarray of shape (m, n)
           Reference panel.
       s : ndarray of shape (1, m)
           Query haplotype (wrapped in 2D array for API compatibility).
       emission_matrix : ndarray of shape (m, 2)
           Column 0 = mismatch prob, column 1 = match prob.
       r : ndarray of shape (m,)
           Per-site recombination probability.
       norm : bool
           Whether to normalize (scale) the forward probabilities.

       Returns
       -------
       F : ndarray of shape (m, n)
           Forward probabilities.
       c : ndarray of shape (m,)
           Scaling factors.
       ll : float
           Log-likelihood (base 10).
       """
       F = np.zeros((m, n))
       r_n = r / n  # Pre-compute r/n for each site (the O(n) trick constant)

       if norm:
           c = np.zeros(m)

           # Initialization: site 0
           # Prior pi_j = 1/n, times emission probability
           for i in range(n):
               if H[0, i] == s[0, 0]:
                   F[0, i] = (1 / n) * emission_matrix[0, 1]  # match
               else:
                   F[0, i] = (1 / n) * emission_matrix[0, 0]  # mismatch
               c[0] += F[0, i]  # Accumulate scaling factor

           # Normalize so forward probs sum to 1
           for i in range(n):
               F[0, i] /= c[0]

           # Forward pass: sites 1, ..., m-1
           for l in range(1, m):
               for i in range(n):
                   # Li-Stephens transition (normalized: sum = 1, so
                   # the switch term is simply r_n[l])
                   F[l, i] = F[l - 1, i] * (1 - r[l]) + r_n[l]

                   # Emission: multiply by match or mismatch probability
                   if H[l, i] == s[0, l]:
                       F[l, i] *= emission_matrix[l, 1]  # match
                   else:
                       F[l, i] *= emission_matrix[l, 0]  # mismatch

                   c[l] += F[l, i]  # Accumulate scaling factor

               # Normalize
               for i in range(n):
                   F[l, i] /= c[l]

           # Log-likelihood = sum of log scaling factors
           ll = np.sum(np.log10(c))

       else:
           c = np.ones(m)

           # Initialization: site 0 (same as above, without normalization)
           for i in range(n):
               if H[0, i] == s[0, 0]:
                   F[0, i] = (1 / n) * emission_matrix[0, 1]
               else:
                   F[0, i] = (1 / n) * emission_matrix[0, 0]

           # Forward pass: sites 1, ..., m-1
           for l in range(1, m):
               S = np.sum(F[l - 1, :])  # Must compute sum explicitly (O(n))
               for i in range(n):
                   # Unnormalized: switch term requires S * r_n[l]
                   F[l, i] = F[l - 1, i] * (1 - r[l]) + S * r_n[l]

                   if H[l, i] == s[0, l]:
                       F[l, i] *= emission_matrix[l, 1]
                   else:
                       F[l, i] *= emission_matrix[l, 0]

           # Log-likelihood from final sum
           ll = np.log10(np.sum(F[m - 1, :]))

       return F, c, ll

Let's trace through a small example to make sure we understand every step:

.. code-block:: python

   # Tiny example: 3 reference haplotypes, 4 sites
   H = np.array([
       [0, 1, 0],  # site 0
       [0, 0, 1],  # site 1
       [1, 1, 0],  # site 2
       [0, 1, 1],  # site 3
   ])
   s = np.array([[0, 0, 1, 1]])  # query haplotype
   mu = 0.1  # exaggerated for visibility
   num_alleles = np.array([2, 2, 2, 2])

   # Build emission matrix: column 0 = mismatch, column 1 = match
   e_mat = np.zeros((4, 2))
   for i in range(4):
       e_mat[i, 0] = mu        # P(mismatch) = mu
       e_mat[i, 1] = 1 - mu    # P(match) = 1 - mu

   r = np.array([0.0, 0.1, 0.1, 0.1])
   n = 3

   # Run both versions and verify they give the same log-likelihood
   F_norm, c, ll_norm = forwards_ls_hap(n, 4, H, s, e_mat, r, norm=True)
   F_raw, _, ll_raw = forwards_ls_hap(n, 4, H, s, e_mat, r, norm=False)

   print(f"Log-likelihood (normalized): {ll_norm:.4f}")
   print(f"Log-likelihood (raw):        {ll_raw:.4f}")
   print(f"Match: {np.isclose(ll_norm, ll_raw)}")

   print(f"\nNormalized forward probs (sum to 1 at each site):")
   for l in range(4):
       print(f"  Site {l}: {F_norm[l].round(4)} (sum={F_norm[l].sum():.4f})")

   print(f"\nRaw forward probs:")
   for l in range(4):
       print(f"  Site {l}: {F_raw[l].round(8)} (sum={F_raw[l].sum():.8f})")

With the forward algorithm in place, we now turn to its mirror image: the
backward algorithm.


The Backward Algorithm
========================

The backward algorithm computes :math:`\beta_j(\ell) = P(X_{\ell+1}, \ldots, X_m \mid Z_\ell = j)`:
the probability of the future data given the current state.

While the forward algorithm asks "how likely is everything up to and including
site :math:`\ell`?", the backward algorithm asks "how likely is everything
*after* site :math:`\ell`?" Together, they give the full posterior probability
at each site, using all the data.

Deriving the recursion
------------------------

**Initialization** (:math:`\ell = m`):

.. math::

   \beta_j(m) = 1 \quad \text{for all } j

There's no future data after the last site, so the probability is 1 regardless
of the state.

**Recursion** (:math:`\ell = m-1, \ldots, 1`):

.. math::

   \beta_j(\ell) = \sum_{i=1}^n A_{ji} \cdot e_i(X_{\ell+1}) \cdot \beta_i(\ell+1)

Note the index order: :math:`A_{ji}` (transition **from** :math:`j` **to** :math:`i`),
because we're asking "given state :math:`j` now, what's the probability of the
future data?"

.. admonition:: Probability Aside: Forward vs. Backward Index Order

   A common source of confusion is the index order in the backward recursion.
   In the forward recursion, we sum over the *previous* state :math:`i` with
   transition :math:`A_{ij}` (from :math:`i` to :math:`j`). In the backward
   recursion, we sum over the *next* state :math:`i` with transition
   :math:`A_{ji}` (from :math:`j` to :math:`i`). The difference is which
   direction we are moving in time:

   - Forward: "I was in state :math:`i`; what is the probability of reaching
     state :math:`j`?" -- sum over previous states.
   - Backward: "I am in state :math:`j`; what is the probability of the future
     data?" -- sum over next states.

   For the LS model, :math:`A_{ji}` has the same structure as :math:`A_{ij}`
   (since :math:`A` is symmetric: :math:`A_{ji} = A_{ij}`), so the index
   order doesn't change the formulas. But it matters for understanding what
   the backward variable represents.

Using the Li-Stephens structure:

.. math::

   \beta_j(\ell) &= \sum_i \left[(1 - r_{\ell+1})\delta_{ji} + \frac{r_{\ell+1}}{n}\right] e_i(X_{\ell+1}) \beta_i(\ell+1) \\
   &= (1 - r_{\ell+1}) \cdot e_j(X_{\ell+1}) \cdot \beta_j(\ell+1)
   + \frac{r_{\ell+1}}{n} \sum_i e_i(X_{\ell+1}) \beta_i(\ell+1)

Again, the sum :math:`\sum_i e_i \beta_i` is computed once and reused, giving
:math:`O(n)` per site.

Scaling the backward variables
---------------------------------

To maintain numerical stability, the backward variables are scaled using the
**same** scaling factors :math:`c_\ell` from the forward pass. At each step, we
divide by :math:`c_{\ell+1}`:

.. math::

   \hat{\beta}_j(\ell) = \frac{1}{c_{\ell+1}} \left[(1 - r_{\ell+1}) \cdot e_j(X_{\ell+1}) \cdot \hat{\beta}_j(\ell+1) + \frac{r_{\ell+1}}{n} \sum_i e_i(X_{\ell+1}) \hat{\beta}_i(\ell+1)\right]

**Why the same** :math:`c` **?** The posterior probability at any site is:

.. math::

   P(Z_\ell = j \mid X_1, \ldots, X_m) = \hat{\alpha}_j(\ell) \cdot \hat{\beta}_j(\ell)

This only works if the forward and backward variables use compatible scaling.
Using :math:`c_\ell` from the forward pass ensures this.

.. admonition:: Probability Aside: Why the Product Gives the Posterior

   The unscaled posterior is:

   .. math::

      P(Z_\ell = j \mid X) = \frac{\alpha_j(\ell) \beta_j(\ell)}{P(X)}

   where :math:`P(X) = \sum_j \alpha_j(\ell) \beta_j(\ell)` (this sum is
   the same for all :math:`\ell`). When we use compatible scaling, the
   product :math:`\hat{\alpha}_j \hat{\beta}_j` is already proportional to
   the posterior, and we simply need to normalize so it sums to 1. This is
   derived in detail in the :ref:`HMM chapter <hmms>`.

.. code-block:: python

   def backwards_ls_hap(n, m, H, s, emission_matrix, c, r):
       """Backward algorithm for the haploid Li-Stephens model.

       Parameters
       ----------
       n, m, H, s, emission_matrix, r : same as forwards_ls_hap.
       c : ndarray of shape (m,)
           Scaling factors from the forward pass.

       Returns
       -------
       B : ndarray of shape (m, n)
           Scaled backward probabilities.
       """
       B = np.zeros((m, n))

       # Initialization: last site -- beta = 1 for all states
       for i in range(n):
           B[m - 1, i] = 1.0

       r_n = r / n  # Pre-compute r/n for the O(n) trick

       # Backward pass: iterate from site m-2 down to 0
       for l in range(m - 2, -1, -1):
           # Pre-compute emission * backward for each state at site l+1.
           # This is the inner product that gets reused via the O(n) trick.
           tmp_B = np.zeros(n)
           tmp_B_sum = 0.0
           for i in range(n):
               if H[l + 1, i] == s[0, l + 1]:
                   emission_prob = emission_matrix[l + 1, 1]  # match
               else:
                   emission_prob = emission_matrix[l + 1, 0]  # mismatch
               tmp_B[i] = emission_prob * B[l + 1, i]  # e_i * beta_i
               tmp_B_sum += tmp_B[i]  # Accumulate the sum (O(n) trick)

           # Compute backward variable at site l
           for i in range(n):
               B[l, i] = r_n[l + 1] * tmp_B_sum      # switch: sum term
               B[l, i] += (1 - r[l + 1]) * tmp_B[i]  # stay: diagonal term
               B[l, i] /= c[l + 1]                    # scale with forward c

       return B

   # Run forward-backward on our small example
   F, c, ll = forwards_ls_hap(n, 4, H, s, e_mat, r, norm=True)
   B = backwards_ls_hap(n, 4, H, s, e_mat, c, r)

   # Posterior probabilities: alpha * beta (should sum to ~1 at each site)
   print("Posterior P(Z_l = j | all data):")
   for l in range(4):
       posterior = F[l] * B[l]
       posterior /= posterior.sum()  # Should already sum to ~1
       print(f"  Site {l}: {posterior.round(4)}")

With both forward and backward probabilities computed, we can now perform
posterior decoding.


Posterior Decoding
====================

With both forward and backward probabilities, we can compute the **posterior
probability** of being in state :math:`j` at site :math:`\ell`:

.. math::

   \gamma_j(\ell) = P(Z_\ell = j \mid X_1, \ldots, X_m)
   = \hat{\alpha}_j(\ell) \cdot \hat{\beta}_j(\ell)

This is the full posterior -- it uses the data at **all** sites, not just the
ones up to :math:`\ell`. The **posterior decoding** chooses the most likely state
independently at each site:

.. math::

   Z_\ell^* = \arg\max_j \gamma_j(\ell)

.. admonition:: Probability Aside: Posterior Decoding vs. Viterbi

   Posterior decoding and the Viterbi algorithm both produce a state sequence,
   but they optimize different objectives:

   - **Posterior decoding** maximizes the probability of each site
     *independently*: :math:`\arg\max_j P(Z_\ell = j \mid X)` at each
     :math:`\ell`. This minimizes the expected number of incorrectly
     decoded sites.
   - **Viterbi** maximizes the probability of the *entire path jointly*:
     :math:`\arg\max_{Z_1,\ldots,Z_m} P(Z_1,\ldots,Z_m \mid X)`. This
     finds the single most probable sequence.

   The difference matters when there are multiple plausible paths. Posterior
   decoding might assign site :math:`\ell` to state A and site :math:`\ell+1`
   to state B, even if the transition :math:`A \to B` has zero probability.
   Viterbi never does this, because it considers the path as a whole.

   In practice, for the LS model, the two methods usually agree except near
   recombination breakpoints, where the uncertainty is highest.

**Warning**: Posterior decoding is not the same as the Viterbi path. Posterior
decoding maximizes each :math:`Z_\ell` independently, which can sometimes
produce impossible state sequences. The Viterbi algorithm (next section) finds
the globally most likely path.

.. code-block:: python

   def posterior_decoding(F, B):
       """Compute posterior decoding from forward-backward probabilities.

       Parameters
       ----------
       F : ndarray of shape (m, n)
           Scaled forward probabilities.
       B : ndarray of shape (m, n)
           Scaled backward probabilities.

       Returns
       -------
       gamma : ndarray of shape (m, n)
           Posterior state probabilities.
       path : ndarray of shape (m,)
           Most likely state at each site (posterior decoding).
       """
       # Element-wise product of forward and backward probabilities
       gamma = F * B
       # Normalize rows (should already be close to 1, but enforce it)
       gamma /= gamma.sum(axis=1, keepdims=True)
       # Posterior decoding: pick the most probable state at each site
       path = np.argmax(gamma, axis=1)
       return gamma, path

   gamma, decoded = posterior_decoding(F, B)
   print("Posterior decoding:")
   for l in range(4):
       print(f"  Site {l}: state={decoded[l]}, "
             f"confidence={gamma[l, decoded[l]]:.3f}")

Having obtained a per-site optimal decoding, we now turn to finding the globally
optimal path.


The Viterbi Algorithm
======================

The Viterbi algorithm finds the **single most likely state sequence** -- the
most likely gear sequence through which the mechanism ticked from the first site
to the last:

.. math::

   Z^* = \arg\max_{Z_1, \ldots, Z_m} P(Z_1, \ldots, Z_m \mid X_1, \ldots, X_m)

This is the global optimum, not the per-site optimum of posterior decoding.

Think of it this way: if the Li & Stephens HMM is a watch mechanism, the
Viterbi algorithm traces the exact sequence of gears that drove the hands from
start to finish. Every transition must be mechanically valid (non-zero
transition probability), and the algorithm finds the sequence that best
explains all the observed ticks (alleles) simultaneously.

The Viterbi recursion
-----------------------

Define the **Viterbi variable**:

.. math::

   V_j(\ell) = \max_{Z_1, \ldots, Z_{\ell-1}} P(X_1, \ldots, X_\ell, Z_1, \ldots, Z_{\ell-1}, Z_\ell = j)

This is the probability of the most likely path ending in state :math:`j` at
site :math:`\ell`.

.. admonition:: Terminology: Viterbi Variable vs. Forward Variable

   The Viterbi variable :math:`V_j(\ell)` looks almost identical to the
   forward variable :math:`\alpha_j(\ell)`, with one crucial difference:
   :math:`\alpha_j(\ell)` **sums** over all paths, while :math:`V_j(\ell)`
   takes the **max** over all paths. This difference -- sum vs. max -- is
   the entire distinction between the forward algorithm (which computes
   likelihoods) and the Viterbi algorithm (which finds the best path). In
   the :ref:`HMM chapter <hmms>`, we showed that this replacement is valid
   because max distributes over products just like sum does.

**Initialization:**

.. math::

   V_j(1) = \frac{1}{n} \cdot e_j(X_1)

**Recursion:**

.. math::

   V_j(\ell) = e_j(X_\ell) \cdot \max_{i} \left[\alpha_i(\ell-1) \cdot A_{ij}\right]

The key difference from the forward algorithm: we take the **max** instead of
the **sum** over previous states.

We also store the **pointer** (traceback) array:

.. math::

   P_j(\ell) = \arg\max_i \left[V_i(\ell-1) \cdot A_{ij}\right]


Exploiting Li-Stephens structure for Viterbi
------------------------------------------------

The Li-Stephens transition structure gives us the same :math:`O(n)` speedup
for Viterbi. For each state :math:`j` at site :math:`\ell`, the predecessor
is either:

1. **Stay** (:math:`i = j`): :math:`V_j(\ell-1) \cdot [(1 - r) + r/n]`
2. **Switch** (:math:`i \neq j`): :math:`\max_i V_i(\ell-1) \cdot r/n`

For the "switch" case, we only need :math:`\max_i V_i(\ell-1)`, which is the
same for all :math:`j` and computed once in :math:`O(n)`.

So the decision at each state :math:`j` is: is it better to **stay** (with the
bonus :math:`(1 - r)`) or **switch** from the globally best previous state?

.. math::

   V_j(\ell) = e_j(X_\ell) \cdot \max\left\{V_j(\ell-1) \cdot (1 - r + r/n), \quad V^*(\ell-1) \cdot r/n\right\}

where :math:`V^*(\ell-1) = \max_i V_i(\ell-1)` is computed once.

The pointer :math:`P_j(\ell)` is:

- :math:`j` (stay) if the first term wins
- :math:`\arg\max_i V_i(\ell-1)` (switch) if the second term wins

This is :math:`O(n)` per site: one pass to find :math:`V^*`, then :math:`O(1)`
per state.

.. admonition:: Probability Aside: When Does Viterbi Switch?

   The Viterbi algorithm switches from state :math:`j` at site :math:`\ell`
   when the "switch" option beats the "stay" option:

   .. math::

      V^*(\ell-1) \cdot \frac{r}{n} > V_j(\ell-1) \cdot \left(1 - r + \frac{r}{n}\right)

   Rearranging:

   .. math::

      \frac{V^*(\ell-1)}{V_j(\ell-1)} > \frac{(1-r) + r/n}{r/n} = \frac{n(1-r) + r}{r} = \frac{n - r(n-1)}{r}

   For small :math:`r`, this ratio is approximately :math:`n/r`. So a switch
   happens when the best alternative state is roughly :math:`n/r` times
   more probable than the current state. With :math:`n = 100` and :math:`r = 0.01`,
   that's a factor of 10,000 -- switches are rare and require strong evidence,
   which is biologically appropriate (recombination breakpoints are infrequent).

.. code-block:: python

   def forwards_viterbi_hap(n, m, H, s, emission_matrix, r):
       """Viterbi algorithm for the haploid Li-Stephens model.

       Uses the Li-Stephens structure for O(n) per site.
       Includes rescaling for numerical stability.

       Parameters
       ----------
       n, m, H, s, emission_matrix, r : same as forwards_ls_hap.

       Returns
       -------
       V : ndarray of shape (n,)
           Viterbi probabilities at the last site.
       P : ndarray of shape (m, n), dtype int
           Pointer (traceback) array.
       ll : float
           Log-likelihood of the best path (base 10).
       """
       V = np.zeros(n)
       P = np.zeros((m, n), dtype=np.int64)
       r_n = r / n
       c = np.ones(m)  # Rescaling factors for numerical stability

       # Initialization: uniform prior times emission
       for i in range(n):
           if H[0, i] == s[0, 0]:
               V[i] = (1 / n) * emission_matrix[0, 1]  # match
           else:
               V[i] = (1 / n) * emission_matrix[0, 0]  # mismatch

       # Forward pass: find best path to each state at each site
       for j in range(1, m):
           # Rescale V to prevent underflow.
           # We divide by the current maximum, which keeps the
           # largest value at 1.0 and prevents all values from
           # drifting toward zero.
           argmax = np.argmax(V)
           c[j] = V[argmax]
           V /= c[j]

           for i in range(n):
               # Option 1: Stay in state i (no recombination).
               # Transition weight is (1 - r) + r/n.
               stay = V[i] * (1 - r[j] + r_n[j])

               # Option 2: Switch from the best previous state.
               # Transition weight is r/n. After rescaling,
               # the best previous state has value 1.0, so
               # switch = r_n[j] * 1.0 = r_n[j].
               switch = r_n[j]

               # Compare: is it better to stay or switch?
               V[i] = stay
               P[j, i] = i  # Default: stay in current state
               if V[i] < switch:
                   V[i] = switch
                   P[j, i] = argmax  # Switch from the best previous state

               # Emission: multiply by match or mismatch probability
               if H[j, i] == s[0, j]:
                   V[i] *= emission_matrix[j, 1]  # match
               else:
                   V[i] *= emission_matrix[j, 0]  # mismatch

       # Log-likelihood: sum of log rescaling factors + log of final max
       ll = np.sum(np.log10(c)) + np.log10(np.max(V))
       return V, P, ll


The backward traceback
------------------------

Once the forward pass is complete, we trace back through the pointer array to
recover the most likely path. This is the final step: we start at the last site,
pick the best state, and follow the pointers backward to reconstruct the
complete gear sequence.

.. code-block:: python

   def backwards_viterbi_hap(m, V_last, P):
       """Traceback to find the most likely path.

       Parameters
       ----------
       m : int
           Number of sites.
       V_last : ndarray of shape (n,)
           Viterbi probabilities at the last site.
       P : ndarray of shape (m, n)
           Pointer array from the forward pass.

       Returns
       -------
       path : ndarray of shape (m,)
           Most likely state sequence.
       """
       path = np.zeros(m, dtype=np.int64)
       # Start at the last site: pick the state with highest Viterbi prob
       path[m - 1] = np.argmax(V_last)

       # Trace backward: at each site, follow the pointer from the
       # state we chose at the next site
       for j in range(m - 2, -1, -1):
           path[j] = P[j + 1, path[j + 1]]

       return path

   # Run Viterbi on our example
   V, P, ll_vit = forwards_viterbi_hap(n, 4, H, s, e_mat, r)
   viterbi_path = backwards_viterbi_hap(4, V, P)
   print(f"Viterbi path: {viterbi_path}")
   print(f"Viterbi log-likelihood: {ll_vit:.4f}")

The traceback is :math:`O(m)` -- just follow the pointers backward from the best
final state.

To build further intuition, let us visualize the stay-vs-switch decision in a
larger example.


Understanding the Viterbi Decision
-------------------------------------

Let's visualize the stay-vs-switch decision at each site:

.. code-block:: python

   # Larger example to see the Viterbi algorithm in action
   np.random.seed(42)
   n, m = 5, 20
   H = np.random.binomial(1, 0.3, size=(m, n))

   # Mosaic query: copy h_0 for sites 0-9, h_3 for sites 10-19
   true_path = np.zeros(m, dtype=int)
   true_path[10:] = 3
   s_flat = np.array([H[l, true_path[l]] for l in range(m)])
   s = s_flat.reshape(1, -1)  # Wrap in 2D array for API compatibility

   mu = 0.05
   e_mat = np.zeros((m, 2))
   e_mat[:, 0] = mu        # mismatch probability
   e_mat[:, 1] = 1 - mu    # match probability
   r = np.full(m, 0.1)
   r[0] = 0.0  # No recombination before the first site

   V, P, ll = forwards_viterbi_hap(n, m, H, s, e_mat, r)
   vit_path = backwards_viterbi_hap(m, V, P)

   print(f"True path:    {true_path}")
   print(f"Viterbi path: {vit_path}")
   print(f"Match: {np.array_equal(true_path, vit_path)}")

Now that we have all three algorithms -- forward, backward, and Viterbi -- let
us introduce a utility for verifying the Viterbi result.


Path Log-Likelihood
=====================

Given a specific copying path, we can evaluate its log-likelihood directly.
This is useful for verifying the Viterbi algorithm (the Viterbi path should have
the highest likelihood) and for comparing paths.

.. admonition:: Probability Aside: Path Likelihood vs. Data Likelihood

   The **path log-likelihood** :math:`\log P(Z, X)` is the joint probability
   of a specific path *and* the data. The **data log-likelihood**
   :math:`\log P(X) = \log \sum_Z P(Z, X)` sums over all paths. The Viterbi
   path maximizes the path log-likelihood, not the data log-likelihood. The
   data log-likelihood is always at least as large as the path log-likelihood
   (since it sums over all paths, not just the best one). The forward
   algorithm gives the data log-likelihood; the path log-likelihood function
   below evaluates a specific path.

.. code-block:: python

   def path_loglik_hap(n, m, H, path, s, emission_matrix, r):
       """Evaluate the log-likelihood of a specific copying path.

       Parameters
       ----------
       n, m, H, s, emission_matrix, r : same as forwards_ls_hap.
       path : ndarray of shape (m,)
           The copying path to evaluate.

       Returns
       -------
       ll : float
           Log-likelihood (base 10).
       """
       r_n = r / n

       # First site: initial probability times emission
       if H[0, path[0]] == s[0, 0]:
           ll = np.log10((1 / n) * emission_matrix[0, 1])  # match
       else:
           ll = np.log10((1 / n) * emission_matrix[0, 0])  # mismatch

       old = path[0]

       for l in range(1, m):
           current = path[l]

           # Transition: stay or switch?
           if old == current:
               ll += np.log10((1 - r[l]) + r_n[l])  # Stay: (1-r) + r/n
           else:
               ll += np.log10(r_n[l])  # Switch: r/n

           # Emission: match or mismatch?
           if H[l, current] == s[0, l]:
               ll += np.log10(emission_matrix[l, 1])  # match
           else:
               ll += np.log10(emission_matrix[l, 0])  # mismatch

           old = current

       return ll

   # Verify: Viterbi path should have the highest log-likelihood
   ll_viterbi = path_loglik_hap(n, m, H, vit_path, s, e_mat, r)
   ll_true = path_loglik_hap(n, m, H, true_path, s, e_mat, r)

   print(f"Viterbi path LL: {ll_viterbi:.4f}")
   print(f"True path LL:    {ll_true:.4f}")
   print(f"Viterbi >= True: {ll_viterbi >= ll_true - 1e-10}")

   # Try a random path -- should be worse
   random_path = np.random.randint(0, n, m)
   ll_random = path_loglik_hap(n, m, H, random_path, s, e_mat, r)
   print(f"Random path LL:  {ll_random:.4f}")


Memory Optimization: From O(mn) to O(m + n)
=============================================

The naive Viterbi stores a full :math:`(m \times n)` matrix of Viterbi values.
In practice, we only need the values at the **current** site (plus the pointer
array :math:`P`). The lshmm library implements this optimization.

The key insight: the forward pass only needs :math:`V_{j}(\ell-1)` to compute
:math:`V_j(\ell)`. So we keep a single vector ``V`` of length :math:`n` and
overwrite it at each step.

The pointer array :math:`P` still requires :math:`O(mn)`, but this is stored as
integers (8 bytes each vs. 8 bytes for doubles), and in many applications the
memory savings from not storing the full Viterbi matrix are significant.

The lshmm library goes even further: the
``forwards_viterbi_hap_lower_mem_rescaling_no_pointer`` function eliminates the
pointer array entirely, storing only which states had recombination at each site.
The traceback reconstructs the path from this minimal information.

.. admonition:: Probability Aside: Why Can We Discard the Pointer Array?

   The pointer-free Viterbi works because the LS model has only two types
   of transitions: "stay" (same state) and "switch" (to the globally best
   state). At each site, we only need to know *whether* a switch happened --
   a single bit per state. During traceback, if the current state shows
   "switch," we jump to the globally best state at the previous site; if
   "stay," we remain. This binary encoding reduces memory from :math:`O(mn)`
   integers to :math:`O(mn)` bits, a factor-of-64 improvement.

With all the individual algorithms in place, let us now run them together on a
realistic-sized example.


Putting It All Together: A Complete Example
=============================================

.. code-block:: python

   # Full pipeline on a realistic-sized example
   np.random.seed(123)
   n = 20   # reference haplotypes
   m = 200  # sites

   # Simulate reference panel (biallelic, allele frequency 0.3)
   H = np.random.binomial(1, 0.3, size=(m, n))

   # Create a mosaic query with 4 segments
   true_path = np.zeros(m, dtype=int)
   true_path[0:50] = 3     # Copy from haplotype 3 for sites 0-49
   true_path[50:100] = 7   # Switch to haplotype 7 at site 50
   true_path[100:150] = 12 # Switch to haplotype 12 at site 100
   true_path[150:200] = 1  # Switch to haplotype 1 at site 150

   # Copy alleles from reference with 2% mutation rate
   s_flat = np.array([H[l, true_path[l]] for l in range(m)])
   mutation_mask = np.random.random(m) < 0.02  # ~2% of sites mutate
   s_flat[mutation_mask] = 1 - s_flat[mutation_mask]  # Flip allele
   s = s_flat.reshape(1, -1)

   print(f"Simulated {mutation_mask.sum()} mutations in the query")
   print(f"True breakpoints at sites 50, 100, 150")

   # Set up model parameters using the Li-Stephens mutation estimator
   mu_est = 1.0 / sum(1.0 / k for k in range(1, n - 1))
   mu = 0.5 * mu_est / (n + mu_est)
   print(f"Estimated mu: {mu:.6f}")

   # Build emission matrix
   e_mat = np.zeros((m, 2))
   e_mat[:, 0] = mu        # mismatch probability
   e_mat[:, 1] = 1 - mu    # match probability

   # Recombination probabilities (uniform for this example)
   r = np.full(m, 0.04)
   r[0] = 0.0

   # Forward-backward: compute posteriors
   F, c, ll = forwards_ls_hap(n, m, H, s, e_mat, r, norm=True)
   B = backwards_ls_hap(n, m, H, s, e_mat, c, r)
   gamma, posterior_path = posterior_decoding(F, B)

   # Viterbi: find the most likely gear sequence
   V, P, ll_vit = forwards_viterbi_hap(n, m, H, s, e_mat, r)
   viterbi_path = backwards_viterbi_hap(m, V, P)

   # Compare the two decodings
   posterior_accuracy = np.mean(posterior_path == true_path)
   viterbi_accuracy = np.mean(viterbi_path == true_path)

   print(f"\nResults:")
   print(f"  Forward log-likelihood: {ll:.2f}")
   print(f"  Viterbi log-likelihood: {ll_vit:.2f}")
   print(f"  Posterior decoding accuracy: {posterior_accuracy:.1%}")
   print(f"  Viterbi decoding accuracy:   {viterbi_accuracy:.1%}")

   # Find detected breakpoints (where the Viterbi path changes)
   viterbi_breaks = np.where(np.diff(viterbi_path) != 0)[0] + 1
   print(f"  True breakpoints:     [50, 100, 150]")
   print(f"  Detected breakpoints: {list(viterbi_breaks)}")


Summary
========

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Algorithm
     - What it computes
     - Complexity
   * - Forward
     - :math:`P(X_1..X_\ell, Z_\ell=j)`
     - :math:`O(mn)` time, :math:`O(mn)` space
   * - Backward
     - :math:`P(X_{\ell+1}..X_m \mid Z_\ell=j)`
     - :math:`O(mn)` time, :math:`O(mn)` space
   * - Posterior decoding
     - :math:`\arg\max_j P(Z_\ell=j \mid \text{all data})`
     - Per-site optimal (not global)
   * - Viterbi
     - :math:`\arg\max_{Z_{1..m}} P(Z_{1..m} \mid \text{all data})`
     - :math:`O(mn)` time, :math:`O(mn)` space
   * - Path log-likelihood
     - :math:`\log P(Z, X)`
     - :math:`O(m)` time

All algorithms use the Li-Stephens :math:`O(n)` trick -- the jeweled bearing
of this mechanism -- giving :math:`O(mn)` total instead of the naive
:math:`O(mn^2)`.

We have now assembled the complete haploid gear train: the forward algorithm
computes likelihoods, the backward algorithm enables posterior decoding, and the
Viterbi algorithm traces the most likely gear sequence. Together, they turn the
template mechanism of the :ref:`copying model <copying_model>` into concrete
answers about which reference haplotypes the query is copying from.

But real organisms are diploid -- they carry two copies of each chromosome. In
the next chapter, we extend this mechanism to handle two watches ticking
together.

Next: :ref:`diploid` -- extending the model to diploid genotypes.
