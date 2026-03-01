.. _gamma_smc_forward_backward:

======================================
The Forward-Backward CS-HMM
======================================

   *The mainspring: the engine that drives inference from one end of the genome to the other.*

With the gamma approximation (:ref:`gamma_smc_gamma_approximation`) providing
exact emission updates and the flow field (:ref:`gamma_smc_flow_field`)
providing approximate transition updates, we can now assemble the complete
inference algorithm. This chapter describes the forward pass, the backward
pass, and the formula for combining them into a full posterior at each
position.

In a mechanical watch, the mainspring stores energy and releases it steadily
to drive the gear train. In Gamma-SMC, the forward-backward algorithm is the
mainspring: it sweeps the accumulated evidence from one end of the genome to
the other, building up the posterior at each position.


.. admonition:: Biology Aside -- Reading the genome like a tape

   The forward-backward algorithm scans the genome from left to right (and
   then right to left), accumulating evidence about the TMRCA at each
   position. This is possible because the genome is a linear sequence of
   observations -- heterozygous or homozygous at each site -- and the
   TMRCA at adjacent sites is correlated (it changes only when
   recombination occurs). The forward pass builds a running estimate: "given
   everything I've seen so far on this chromosome, what do I think the TMRCA
   is here?" The backward pass does the same from the other end. Combining
   both gives the best possible estimate at every position, using all the
   data from the entire chromosome.

The Forward Algorithm
=======================

The forward algorithm in a continuous-state HMM (CS-HMM) tracks the
**forward density** -- the probability density of the hidden state at
position :math:`i`, given all observations up to and including position
:math:`i`:

.. math::

   \hat{\alpha}(x_i) := P(X_i = x_i \mid Y_{1:i} = y_{1:i})

This is the continuous-state analogue of the forward variable
:math:`\alpha_k(i)` in discrete HMMs (see :ref:`hmms`). Instead of a
probability vector over :math:`n` states, we have a probability density
function over the continuous TMRCA.

The recursion is:

.. math::

   \hat{\alpha}(x_i) = \frac{1}{c_i} \cdot P(y_i \mid x_i)
   \int_0^\infty P(x_i \mid x_{i-1}) \, \hat{\alpha}(x_{i-1}) \, dx_{i-1}

where :math:`c_i` is a scaling factor that ensures :math:`\hat{\alpha}` is
a normalized density. This recursion says: take the previous forward density,
propagate it through the transition kernel (the integral), then update with
the emission probability, and normalize.

The **Gamma-SMC forward pass** approximates this recursion using two steps
at each position:

**Step 1: Transition.** Apply the flow field :math:`\mathcal{F}` to advance
the gamma parameters through one transition step:

.. math::

   (\alpha, \beta) \xrightarrow{\mathcal{F}} (\alpha', \beta')

This replaces the integral
:math:`\int P(x_i \mid x_{i-1}) \hat{\alpha}(x_{i-1}) dx_{i-1}` with a
single flow field lookup.

**Step 2: Emission.** Incorporate the observation using the conjugate update:

.. math::

   (\alpha', \beta') \xrightarrow{Y_i} (\alpha' + Y_i, \; \beta' + \theta)

(where :math:`Y_i = 0` for hom, :math:`Y_i = 1` for het, and the step is
skipped for missing data).

**Initialization.** The prior on the TMRCA is the stationary distribution of
the coalescent under constant population size, which is
:math:`\text{Exp}(1) = \text{Gamma}(1, 1)`. So the forward pass begins with
:math:`(\alpha, \beta) = (1, 1)`, or equivalently
:math:`(l_\mu, l_C) = (0, 0)`.

.. admonition:: Comparison to PSMC's forward pass

   In PSMC (:ref:`psmc_hmm`), the forward pass at each position performs:

   1. A **matrix-vector multiply**: :math:`\alpha_k(i) = \sum_l p_{lk} \alpha_l(i-1)`,
      which is :math:`O(n^2)` in the number of time intervals.
   2. An **elementwise multiply**: :math:`\alpha_k(i) \leftarrow e_k(y_i) \cdot \alpha_k(i)`,
      which is :math:`O(n)`.

   Gamma-SMC replaces both operations with :math:`O(1)` operations: a bilinear
   interpolation lookup (transition) and a parameter increment (emission). The
   difference in computational cost is dramatic: PSMC with :math:`n = 64` time
   intervals requires :math:`O(64^2) = O(4096)` operations per position;
   Gamma-SMC requires :math:`O(1)`.

The forward pass proceeds along the genome, recording the gamma parameters
:math:`(\alpha_i, \beta_i)` at each user-specified output position.

.. code-block:: python

   import numpy as np

   def gamma_smc_forward(observations, theta, rho, flow_field):
       """Run the Gamma-SMC forward pass.

       Parameters
       ----------
       observations : list of int
           Observation at each position: 1 (het), 0 (hom), -1 (missing).
       theta : float
           Scaled mutation rate per position.
       rho : float
           Scaled recombination rate per position.
       flow_field : FlowField
           Precomputed flow field with a .query(l_mu, l_C) method.

       Returns
       -------
       alphas : ndarray, shape (N,)
           Forward shape parameter at each position.
       betas : ndarray, shape (N,)
           Forward rate parameter at each position.
       """
       N = len(observations)
       alphas = np.zeros(N)
       betas = np.zeros(N)

       # Initialize with the prior: Gamma(1, 1) = Exp(1)
       alpha, beta = 1.0, 1.0

       for i in range(N):
           # Step 1: Transition via flow field
           l_mu = np.log10(alpha / beta)
           l_C = np.log10(1.0 / np.sqrt(alpha))
           dl_mu, dl_C = flow_field.query(l_mu, l_C)
           l_mu += rho * dl_mu  # displacement scaled by rho
           l_C += rho * dl_C
           alpha = 10.0 ** (-2 * l_C)
           beta = alpha * 10.0 ** (-l_mu)

           # Step 2: Emission update (conjugate)
           y = observations[i]
           if y >= 0:  # not missing
               alpha += y       # +1 for het, +0 for hom
               beta += theta

           alphas[i] = alpha
           betas[i] = beta

       return alphas, betas

   # Demonstrate on a short synthetic sequence
   obs = [0]*50 + [1] + [0]*30 + [1] + [0]*19
   theta, rho = 0.001, 0.0004

   # Use a trivial flow field (zero displacement) for illustration
   class ZeroFlow:
       def query(self, l_mu, l_C): return 0.0, 0.0

   a_fwd, b_fwd = gamma_smc_forward(obs, theta, rho, ZeroFlow())
   print(f"After {len(obs)} positions ({sum(obs)} hets):")
   print(f"  Final forward: Gamma({a_fwd[-1]:.1f}, {b_fwd[-1]:.4f})")
   print(f"  Mean TMRCA = {a_fwd[-1]/b_fwd[-1]:.2f}")


The Backward Pass as Reversed Forward
========================================

In a standard discrete HMM, the backward algorithm computes
:math:`\beta_k(i) = P(Y_{i+1:N} \mid X_i = k)` using a recursion that runs
from right to left. The continuous-state analogue is:

.. math::

   \tilde{\alpha}(x_{i+1}) := P(X_{i+1} = x_{i+1} \mid Y_{i+1:N} = y_{i+1:N})

Gamma-SMC uses a key reformulation: instead of deriving a separate backward
recursion, it observes that the backward density can be obtained by **running
the forward algorithm on the reversed sequence**. That is:

- Reverse the observation sequence: :math:`y_N, y_{N-1}, \ldots, y_1`.
- Run the standard forward pass on this reversed sequence.
- At each position :math:`i+1`, the resulting forward density is the backward
  density :math:`\tilde{\alpha}(x_{i+1})`.

This is possible because the SMC transition density is symmetric in a specific
sense: reversing the sequence and re-running the forward algorithm produces
the same mathematical object as the backward algorithm.

.. admonition:: Why reuse the forward algorithm?

   The forward-on-reversed approach has two advantages. First, it avoids
   implementing a separate backward recursion, reducing code complexity and
   the chance of bugs. Second, the flow field :math:`\mathcal{F}` is the
   same in both directions (the SMC transition kernel under constant
   population size is time-reversible), so no additional precomputation is
   needed.


Combining Forward and Backward
=================================

At each output position :math:`i`, we now have:

- The forward density: :math:`\hat{\alpha}(x_i) \approx \text{Gamma}(a, b)`
- The backward density (after one transition step back from :math:`i+1` to
  :math:`i`): :math:`\approx \text{Gamma}(a', b')`

To combine these into the full posterior
:math:`P(X_i = x_i \mid Y_{1:N} = y_{1:N})`, we use the result derived in
Appendix D of the supplement:

.. math::

   P(x_i \mid y_{1:N}) \propto \hat{\alpha}(x_i) \cdot
   \frac{\int_0^\infty P(x_i \mid x_{i+1}) \tilde{\alpha}(x_{i+1}) dx_{i+1}}{P(x_i)}

The numerator involves the backward density propagated one step back through
the transition. The denominator divides out the prior
:math:`P(x_i) = \text{Exp}(1) = \text{Gamma}(1, 1)`.

Substituting the gamma approximations:

.. math::

   P(x_i \mid y_{1:N})
   &\propto x_i^{a-1} e^{-b x_i} \cdot x_i^{a'-1} e^{-b' x_i} / e^{-x_i} \\
   &= x_i^{(a + a' - 1) - 1} \, e^{-(b + b' - 1) x_i}

This is the kernel of a :math:`\text{Gamma}(a + a' - 1, b + b' - 1)`
distribution. Therefore:

.. math::

   X_i \mid Y_{1:N} \sim \text{Gamma}(a + a' - 1, \; b + b' - 1)

.. admonition:: Plain-language summary -- Combining forward and backward

   The forward pass says: "based on everything to the left on the
   chromosome, I think the TMRCA here is about X." The backward pass says:
   "based on everything to the right, I think it's about Y." The combination
   merges these two independent sources of evidence into a single, more
   precise estimate. The subtraction of 1 from both parameters avoids
   counting the prior twice (both passes started from the same prior).
   The result is the **full posterior** at each position -- the best estimate
   of the TMRCA given the entire chromosome.

.. admonition:: Where does the :math:`-1` come from?

   The :math:`-1` in both :math:`a + a' - 1` and :math:`b + b' - 1` arises
   from dividing out the prior :math:`\text{Gamma}(1, 1)`. The forward
   density :math:`\text{Gamma}(a, b)` and the backward density
   :math:`\text{Gamma}(a', b')` each include the prior once (they are
   posterior distributions that incorporate the prior). When we multiply them,
   the prior is counted twice. Dividing by the prior once gives the correct
   posterior, and dividing :math:`\text{Gamma}(1, 1)` out of a product of
   two gammas subtracts 1 from both the shape and rate parameters.

   A useful sanity check: if we have no observations at all, the forward and
   backward densities are both the prior :math:`\text{Gamma}(1, 1)`.
   Combining gives :math:`\text{Gamma}(1 + 1 - 1, 1 + 1 - 1) = \text{Gamma}(1, 1)`,
   recovering the prior. This is correct -- with no data, the posterior
   should equal the prior.


The Backward-Then-Combine Procedure
--------------------------------------

In practice, the combination works as follows:

1. Run the forward pass left-to-right. At each output position :math:`i`,
   record :math:`(a_i, b_i)`.

2. Run the forward pass right-to-left on the reversed sequence. At each
   output position :math:`i+1`, record :math:`(a''_{i+1}, b''_{i+1})`.

3. For each output position :math:`i`, apply the flow field once to
   :math:`(a''_{i+1}, b''_{i+1})` to propagate the backward density from
   :math:`i+1` back to :math:`i`, giving :math:`(a'_i, b'_i)`.

4. Combine:
   :math:`(\alpha_\text{post}, \beta_\text{post}) = (a_i + a'_i - 1, \; b_i + b'_i - 1)`.

The additional flow field step in step 3 accounts for the transition from
position :math:`i+1` to position :math:`i` -- without it, the backward
density would be aligned to position :math:`i+1` instead of :math:`i`.

.. code-block:: python

   def gamma_smc_posterior(observations, theta, rho, flow_field):
       """Compute the full Gamma-SMC posterior at each position.

       Runs forward and backward passes and combines them.

       Parameters
       ----------
       observations : list of int
           Observation at each position: 1 (het), 0 (hom), -1 (missing).
       theta, rho : float
           Scaled mutation and recombination rates.
       flow_field : FlowField
           Precomputed flow field.

       Returns
       -------
       post_alpha : ndarray
           Posterior shape at each position.
       post_beta : ndarray
           Posterior rate at each position.
       """
       # Forward pass (left to right)
       a_fwd, b_fwd = gamma_smc_forward(observations, theta, rho, flow_field)

       # Backward pass = forward pass on reversed sequence
       a_bwd_rev, b_bwd_rev = gamma_smc_forward(
           observations[::-1], theta, rho, flow_field
       )
       # Reverse the backward results to align with original positions
       a_bwd = a_bwd_rev[::-1]
       b_bwd = b_bwd_rev[::-1]

       # Combine: Gamma(a + a' - 1, b + b' - 1)
       post_alpha = a_fwd + a_bwd - 1
       post_beta = b_fwd + b_bwd - 1

       return post_alpha, post_beta

   # Demonstrate on the same synthetic sequence
   a_post, b_post = gamma_smc_posterior(obs, theta, rho, ZeroFlow())
   mean_tmrca = a_post / b_post

   # Sanity check: with no observations, posterior = prior Gamma(1,1)
   a_empty, b_empty = gamma_smc_posterior([-1]*10, theta, rho, ZeroFlow())
   print(f"No observations: Gamma({a_empty[0]:.1f}, {b_empty[0]:.1f}) "
         f"(should be ~Gamma(1, 1))")
   print(f"\nWith data ({len(obs)} positions):")
   print(f"  Position 0:  mean TMRCA = {mean_tmrca[0]:.2f}")
   print(f"  Position 51 (het): mean TMRCA = {mean_tmrca[51]:.2f}")
   print(f"  Position 99: mean TMRCA = {mean_tmrca[-1]:.2f}")


Validity of the Combination
==============================

The combination formula
:math:`\text{Gamma}(a + a' - 1, b + b' - 1)` is valid only if:

- :math:`a + a' - 1 > 0`, i.e., :math:`a + a' > 1`
- :math:`b + b' - 1 > 0`, i.e., :math:`b + b' > 1`

Since the forward and backward passes both start from the prior
:math:`\text{Gamma}(1, 1)` and only add to the shape (via heterozygous
observations) and rate (via homozygous and heterozygous observations), one
might expect :math:`a \geq 1` and :math:`b \geq 1` throughout. However, the
flow field transition step can decrease both :math:`a` and :math:`b`
(recombination introduces uncertainty, which can reduce the effective shape
parameter).

If the gamma approximation is accurate and the entropy of the posterior never
exceeds the entropy of the prior, then :math:`b \geq 1` is guaranteed. This
is enforced by the **entropy clipping** mechanism described in
:ref:`gamma_smc_segmentation_caching`. Without entropy clipping, it is
possible (though rare) for the approximation to drift into a region where
:math:`b + b' < 1`, producing an invalid combined distribution.


Summary
=========

The Gamma-SMC forward-backward algorithm:

1. **Forward pass**: sweep left-to-right, applying
   :math:`\mathcal{F}` (transition) then conjugate update (emission) at each
   position. Cost: :math:`O(N)`.

2. **Backward pass**: run the forward algorithm on the reversed sequence.
   Cost: :math:`O(N)`.

3. **Combination**: at each output position, combine forward
   :math:`\text{Gamma}(a, b)` and backward :math:`\text{Gamma}(a', b')` into
   :math:`\text{Gamma}(a + a' - 1, b + b' - 1)`. Cost: :math:`O(1)` per
   output position.

Total cost: :math:`O(N)` for the entire genome, with no matrix operations
and no time discretization. This makes Gamma-SMC orders of magnitude faster
than PSMC for the same task.

Next: :ref:`gamma_smc_segmentation_caching` -- the engineering tricks that
make the :math:`O(N)` algorithm fast in practice.
