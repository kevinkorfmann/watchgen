.. _psmc_hmm:

================================
The PSMC HMM and EM Algorithm
================================

   *The gear train: turning mathematical insight into a learning machine.*

In the previous chapters, we derived all the HMM parameters as functions of
:math:`\theta_0`, :math:`\rho_0`, and :math:`\lambda_0, \ldots, \lambda_n`.
We built the time intervals (:ref:`psmc_discretization`), computed the
transition matrix, and worked out emission probabilities -- all the individual
gears of the PSMC watch. Now we close the loop: given the observed data
(the heterozygosity sequence), use the **Expectation-Maximization (EM)
algorithm** to find the parameters that best explain the data.

Think of it this way. So far we have been a watchmaker who knows how to cut
gears of any size. But we have not yet figured out *which* sizes to cut. The
data -- the sequence of heterozygous and homozygous sites along the genome --
is the standard clock signal we are trying to match. The EM algorithm is the
process of iteratively adjusting the gears until the watch keeps time: we
try a set of gear sizes, see how well the watch matches the signal, adjust,
and repeat. Each iteration brings the gears closer to the right configuration.

This chapter has five main sections:

1. **The complete HMM specification** -- collecting everything from earlier
   chapters into one place.
2. **The forward-backward algorithm** -- the computational engine that asks
   "given these gear sizes, how well does the watch keep time, and which
   configurations are most likely?"
3. **The EM algorithm** -- the iterative adjustment loop.
4. **The full EM loop** -- putting it all together for real inference.
5. **Multiple sequences and missing data** -- handling real genomes.

If any of the HMM concepts feel unfamiliar -- forward probabilities, scaling,
posterior decoding -- revisit :ref:`the HMM prerequisite chapter <hmms>`, where
we built these algorithms from scratch. This chapter assumes you are comfortable
with the forward algorithm and the idea of hidden states; we will explain the
backward algorithm and the EM machinery in full detail here.


Step 1: The Complete HMM Specification
=========================================

Before we can learn anything, we need to state precisely what our HMM looks
like. Let's collect every piece from the previous chapters into one place.

The state space is :math:`\{0, 1, \ldots, n\}`,
where state :math:`k` means "the coalescence time at this position falls in the
interval :math:`[t_k, t_{k+1})`." In the watch metaphor, each state is a
possible gear configuration -- a hypothesis about how deep in time the two
haplotypes share their most recent common ancestor at this genomic position.

**Initial distribution:**

.. math::

   a_0(k) = \sigma_k = \frac{1}{C_\sigma}\left[\frac{\alpha_k - \alpha_{k+1}}{C_\pi \rho} + \frac{\pi_k}{2}\right]

This is the stationary distribution we derived in :ref:`psmc_discretization`.
It tells us, before looking at any data, how likely each coalescence time
interval is. The watch starts in its equilibrium state.

**Transition matrix:**

.. math::

   p_{kl} = \frac{\pi_k}{C_\sigma \sigma_k} q_{kl} + \delta_{kl}\left(1 - \frac{\pi_k}{C_\sigma \sigma_k}\right)

This is the probability of moving from interval :math:`k` at one genomic
position to interval :math:`l` at the next. It captures both the possibility of
recombination (which reshuffles the coalescence time) and no recombination
(which keeps it the same). This matrix was derived in :ref:`psmc_discretization`,
Step 4.

**Emission probabilities:**

.. math::

   e_k(0) = e^{-\theta \bar{t}_k} \quad \text{(homozygous)}

.. math::

   e_k(1) = 1 - e^{-\theta \bar{t}_k} \quad \text{(heterozygous)}

where :math:`\bar{t}_k = -\frac{1}{\rho}\ln\left(1 - \frac{\pi_k}{C_\sigma \sigma_k}\right)` is the effective coalescence time in interval :math:`k`.

The intuition: a deeper coalescence time (larger :math:`\bar{t}_k`) means the
two haplotypes have been evolving independently for longer, so there is a higher
chance of observing a heterozygous site. A shallow coalescence time means
recent common ancestry and thus more homozygous sites.

**Observation alphabet:** :math:`\{0, 1\}` (hom/het). Missing data gets
:math:`e_k(\text{missing}) = 1` for all :math:`k`.

.. admonition:: Why missing data gets emission probability 1

   Setting :math:`e_k(\text{missing}) = 1` for all states means "this
   observation is equally consistent with every possible coalescence time."
   Missing data provides zero information -- it neither favors nor disfavors
   any state. The HMM simply passes through missing positions, maintaining
   the transition dynamics but learning nothing from them. This is a standard
   HMM technique (see :ref:`hmms`) and requires no special machinery.

Now let's implement this complete specification:

.. code-block:: python

   import numpy as np

   class PSMC_HMM:
       """Complete PSMC Hidden Markov Model.

       This class bundles together all the HMM parameters (transitions,
       emissions, initial distribution) and provides methods for computing
       likelihoods and running the forward-backward algorithm.

       Think of this as the fully assembled watch: all the gears from the
       discretization chapter are now meshed together into a working mechanism.

       Parameters
       ----------
       n : int
           n+1 = number of hidden states (time intervals).
       theta : float
           Mutation rate per bin.
       rho : float
           Recombination rate per bin.
       lambdas : ndarray of shape (n + 1,)
           Relative population sizes.
       t_max : float
       alpha_param : float
       """

       def __init__(self, n, theta, rho, lambdas, t_max=15.0, alpha_param=0.1):
           self.n = n
           self.N = n + 1  # number of states (one per time interval)
           self.theta = theta
           self.rho = rho
           self.lambdas = lambdas.copy()  # store our own copy to avoid aliasing
           self.t_max = t_max
           self.alpha_param = alpha_param

           # Build all HMM parameters from the population-genetic quantities.
           # build_psmc_hmm() was defined in the discretization chapter and
           # returns the transition matrix, emission matrix, and initial dist.
           self.transitions, self.emissions, self.initial = build_psmc_hmm(
               n, t_max, theta, rho, lambdas, alpha_param=alpha_param)

       def log_likelihood(self, seq):
           """Compute log-likelihood of an observation sequence.

           Uses the scaled forward algorithm (see Step 2 below).

           Parameters
           ----------
           seq : ndarray of shape (L,), dtype=int
               Observation sequence (0 = hom, 1 = het, 2+ = missing).

           Returns
           -------
           ll : float
               Log-likelihood log P(X | theta, rho, lambdas).
           """
           _, ll = self.forward_scaled(seq)
           return ll

       def forward_scaled(self, seq):
           """Scaled forward algorithm.

           The forward algorithm computes, for each position a and state k,
           the joint probability P(X_1, ..., X_a, Z_a = k) -- "the probability
           of seeing these observations AND being in state k right now."

           We use scaling to prevent numerical underflow. At each position,
           we normalize the forward probabilities to sum to 1 and accumulate
           the log of the normalization constants. The sum of these logs
           gives us the log-likelihood. (This technique is explained in detail
           in the HMM prerequisite chapter: see :ref:`hmms`.)

           Returns
           -------
           alpha_hat : ndarray of shape (L, N)
               Scaled forward probabilities.
           log_likelihood : float
           """
           L = len(seq)            # length of observation sequence
           N = self.N              # number of hidden states
           alpha_hat = np.zeros((L, N))  # scaled forward probabilities
           log_likelihood = 0.0         # accumulator for log P(X)

           # --- Initialization (position 0) ---
           # alpha(0, k) = P(Z_0 = k) * P(X_0 | Z_0 = k)
           #             = initial[k] * emission[X_0, k]
           for k in range(N):
               obs = seq[0]          # observation at position 0
               if obs >= 2:          # missing data: no information
                   e = 1.0
               else:
                   e = self.emissions[obs, k]  # P(obs | state k)
               alpha_hat[0, k] = self.initial[k] * e

           # Scale: normalize so alpha_hat[0, :] sums to 1
           c = alpha_hat[0].sum()    # c is the scaling constant
           if c > 0:
               alpha_hat[0] /= c     # now alpha_hat[0] sums to 1
               log_likelihood += np.log(c)  # accumulate log(c)

           # --- Recursion (positions 1 through L-1) ---
           # alpha(a, k) = P(X_a | Z_a=k) * sum_j[ alpha(a-1, j) * P(k|j) ]
           # In words: "probability of arriving in state k at position a"
           # = (emission at k) * (sum over all previous states j of:
           #    being in j at a-1 times transitioning from j to k)
           for pos in range(1, L):
               for k in range(N):
                   obs = seq[pos]
                   if obs >= 2:      # missing data
                       e = 1.0
                   else:
                       e = self.emissions[obs, k]

                   # np.dot computes the sum over all previous states j:
                   # sum_j alpha_hat[pos-1, j] * transitions[j, k]
                   alpha_hat[pos, k] = e * np.dot(
                       alpha_hat[pos - 1, :], self.transitions[:, k])

               # Scale this position
               c = alpha_hat[pos].sum()
               if c > 0:
                   alpha_hat[pos] /= c
                   log_likelihood += np.log(c)

           return alpha_hat, log_likelihood

       def backward_scaled(self, seq, alpha_hat):
           """Scaled backward algorithm.

           The backward algorithm is the mirror image of the forward algorithm.
           Where the forward algorithm answers "what is the probability of the
           observations up to position a, given state k at a?", the backward
           algorithm answers "what is the probability of the observations
           AFTER position a, given state k at a?"

           Together, forward and backward give us the posterior: the probability
           of each state at each position, given ALL the observations. This is
           like looking at a position from both directions -- past and future --
           to triangulate which gear configuration is most likely.

           Parameters
           ----------
           seq : ndarray of shape (L,)
           alpha_hat : ndarray of shape (L, N)
               From forward_scaled (we use its scaling factors for consistency).

           Returns
           -------
           beta_hat : ndarray of shape (L, N)
               Scaled backward probabilities.
           """
           L = len(seq)
           N = self.N
           beta_hat = np.zeros((L, N))

           # --- Initialization: at the last position, beta = 1 ---
           # There are no future observations to account for, so the
           # backward probability is 1 for all states.
           beta_hat[L - 1, :] = 1.0

           # --- Recursion: work backwards from position L-2 to 0 ---
           # beta(a, k) = sum_l[ P(l|k) * P(X_{a+1}|l) * beta(a+1, l) ]
           # In words: "the probability of the future observations, given
           # state k at position a" = sum over all possible next states l of:
           # (transition k->l) * (emission at l) * (future from l onwards)
           for pos in range(L - 2, -1, -1):  # L-2, L-3, ..., 0
               for k in range(N):
                   total = 0.0
                   for l in range(N):         # sum over next states
                       obs = seq[pos + 1]     # observation at NEXT position
                       if obs >= 2:           # missing data
                           e = 1.0
                       else:
                           e = self.emissions[obs, l]
                       total += self.transitions[k, l] * e * beta_hat[pos + 1, l]
                   beta_hat[pos, k] = total

               # Scale to prevent underflow
               c = beta_hat[pos].sum()
               if c > 0:
                   beta_hat[pos] /= c

           return beta_hat


Step 2: The Forward-Backward Algorithm
=========================================

With the forward and backward algorithms in hand, we can now compute the
**posterior probability** of each hidden state at each position. This is the
central computation of the E-step and the key to the entire EM procedure.

Recall the watch metaphor: at each genomic position, the hidden state represents
which time interval the coalescence falls in -- which gear configuration the
watch is using at that position. The forward-backward algorithm figures out,
for each position, the probability of every possible gear configuration, given
the *entire* sequence of observations (not just the observations before or after
that position, but all of them together).

.. admonition:: Probability Aside: what forward-backward actually computes

   The forward-backward algorithm computes two fundamental quantities:

   **1. Posterior state probabilities** :math:`\gamma_k(a)`:

   .. math::

      \gamma_k(a) = P(Z_a = k \mid X_1, \ldots, X_L)

   This answers: "given everything we observed across the entire genome,
   what is the probability that the coalescence time at position :math:`a`
   falls in interval :math:`k`?"

   **2. Expected transition counts** :math:`\xi_{kl}(a)`:

   .. math::

      \xi_{kl}(a) = P(Z_a = k, Z_{a+1} = l \mid X_1, \ldots, X_L)

   This answers: "what is the probability that the coalescence time was in
   interval :math:`k` at position :math:`a` AND in interval :math:`l` at
   position :math:`a+1`?"

   Both are computed by combining the forward and backward variables. The
   forward variable :math:`f_k(a)` looks at the data from the left (positions
   1 through :math:`a`). The backward variable :math:`b_k(a)` looks at the
   data from the right (positions :math:`a+1` through :math:`L`). Together,
   they see everything.

The posterior is computed as:

.. math::

   P(Z_a = k \mid X_1, \ldots, X_L) = \frac{f_k(a) \cdot b_k(a)}{\sum_{l} f_l(a) \cdot b_l(a)}

where :math:`f_k(a)` is the forward probability and :math:`b_k(a)` is the backward
probability.

We've already implemented the forward algorithm above. The backward algorithm runs the
same recursion in reverse:

**Initialization** (:math:`a = L`):

.. math::

   b_k(L) = 1 \quad \text{for all } k

**Recursion** (:math:`a = L-1, \ldots, 1`):

.. math::

   b_k(a) = \sum_{l=0}^{n} p_{kl} \cdot e_l(X_{a+1}) \cdot b_l(a+1)

The backward initialization says: at the end of the sequence, there are no
future observations, so the backward probability is 1 regardless of state.
The recursion says: to compute the backward probability at position :math:`a`,
consider every possible state :math:`l` at the *next* position, weight it by
the transition probability, the emission at that next state, and the backward
probability from that next state onwards. Sum over all possibilities.

Together, forward and backward give us two things:

1. **Posterior state probabilities** :math:`\gamma_k(a) = P(Z_a = k \mid X)` at each position
2. **Expected transition counts** :math:`\xi_{kl}(a) = P(Z_a = k, Z_{a+1} = l \mid X)` between positions

These are the **expected sufficient statistics** -- exactly what EM needs.

.. admonition:: Probability Aside: what are "expected sufficient statistics"?

   The term "sufficient statistics" comes from classical statistics. A
   **sufficient statistic** is a summary of data that captures everything
   the data can tell you about a parameter. For example, the sample mean
   and sample variance are sufficient statistics for a normal distribution --
   once you know them, the individual data points add no further information
   about :math:`\mu` and :math:`\sigma^2`.

   For an HMM, the sufficient statistics are:

   - **How often each state is visited** (the :math:`\gamma_k` sums)
   - **How often each transition occurs** (the :math:`\xi_{kl}` sums)
   - **How often each observation is emitted from each state** (the emission counts)

   If we knew the hidden states, we could compute these directly by counting.
   But the states are hidden, so we compute the **expected** counts -- weighted
   averages over all possible state sequences, where the weights come from the
   posterior probabilities. This is why the E-step is called "Expectation": it
   computes the expected values of the sufficient statistics under the current
   parameter estimates.

   In the watch metaphor: we cannot directly observe which gears are engaged at
   each tick. But by looking at the output (het/hom) from both directions, we can
   estimate *how often* each gear was probably engaged. These expected gear-usage
   counts are the sufficient statistics.

.. admonition:: Probability Aside: Bayes' theorem in sequence form

   The posterior :math:`\gamma_k(a)` is a direct application of Bayes' theorem.
   The forward variable :math:`f_k(a)` captures
   :math:`P(X_1, \ldots, X_a, Z_a = k)` -- the joint probability of seeing the
   observations up to position :math:`a` **and** being in state :math:`k`. The
   backward variable :math:`b_k(a)` captures
   :math:`P(X_{a+1}, \ldots, X_L \mid Z_a = k)` -- the probability of the
   remaining observations **given** state :math:`k` at position :math:`a`.
   Multiplying and normalizing gives:

   .. math::

      \gamma_k(a) = \frac{f_k(a) \cdot b_k(a)}{\sum_l f_l(a) \cdot b_l(a)}
      = \frac{P(X, Z_a = k)}{P(X)}

   This is just Bayes' rule: posterior = likelihood :math:`\times` prior / evidence.
   The forward variable plays the role of "likelihood times prior" (all the
   evidence up to and including position :math:`a`), while the backward variable
   completes the picture with the remaining evidence. The denominator
   :math:`P(X)` -- the total probability of the data -- ensures the result is a
   proper probability distribution over states at each position.

Let's verify that the posteriors behave as expected:

.. code-block:: python

   # Demonstrate that gamma sums to 1 at each position (it's a distribution)
   def check_posteriors(hmm, seq, n_positions=5):
       """Verify posterior probabilities are valid distributions.

       At each position, gamma should sum to 1 because we must be in
       exactly one state. This is a basic sanity check that the
       forward-backward computation is correct.
       """
       alpha_hat, ll = hmm.forward_scaled(seq)     # forward pass
       beta_hat = hmm.backward_scaled(seq, alpha_hat)  # backward pass

       # Check a few representative positions across the sequence
       for pos in [0, len(seq)//4, len(seq)//2, 3*len(seq)//4, len(seq)-1]:
           # gamma = element-wise product of forward and backward
           gamma = alpha_hat[pos] * beta_hat[pos]
           gamma /= gamma.sum()   # normalize to get posterior
           peak_state = np.argmax(gamma)  # most probable state
           print(f"  pos={pos:6d}: sum(gamma)={gamma.sum():.6f}, "
                 f"peak state={peak_state}, "
                 f"peak prob={gamma[peak_state]:.4f}")

Now we compute the expected sufficient statistics. This is the heart of the
E-step -- the computation that extracts from the data everything the M-step
needs to update the parameters:

.. code-block:: python

   def compute_expected_counts(hmm, seq):
       """Compute expected counts for the E-step of EM.

       This function runs the forward-backward algorithm and then extracts
       three sets of expected counts:

       1. gamma_sum[k] = sum over all positions a of gamma_k(a)
          "How much total time does the HMM spend in state k?"

       2. xi_sum[k, l] = sum over all adjacent pairs (a, a+1) of xi_{kl}(a)
          "How many times does the HMM transition from state k to state l?"

       3. emission_counts[obs, k] = sum over positions where X_a = obs of gamma_k(a)
          "How many times does state k emit observation obs?"

       These are the expected sufficient statistics. Together, they
       summarize everything the data can tell us about the parameters.

       Parameters
       ----------
       hmm : PSMC_HMM
       seq : ndarray of shape (L,)

       Returns
       -------
       gamma_sum : ndarray of shape (N,)
           Sum of posterior state probabilities over all positions.
       xi_sum : ndarray of shape (N, N)
           Sum of expected transition counts.
       emission_counts : ndarray of shape (2, N)
           Expected emission counts: emission_counts[obs, k] =
           sum over positions with observation obs of gamma_k(pos).
       log_likelihood : float
       """
       L = len(seq)
       N = hmm.N

       # --- Run forward and backward passes ---
       alpha_hat, ll = hmm.forward_scaled(seq)
       beta_hat = hmm.backward_scaled(seq, alpha_hat)

       # --- Compute posterior state probabilities gamma_k(a) ---
       # and accumulate the sufficient statistics
       gamma_sum = np.zeros(N)          # total "time spent" in each state
       emission_counts = np.zeros((2, N))  # emission counts per state

       for pos in range(L):
           # gamma_k(a) = alpha_hat(a,k) * beta_hat(a,k), then normalize
           gamma = alpha_hat[pos] * beta_hat[pos]
           total = gamma.sum()
           if total > 0:
               gamma /= total           # normalize to a proper distribution

           gamma_sum += gamma            # accumulate state occupancy

           obs = seq[pos]
           if obs < 2:                   # not missing data
               emission_counts[obs] += gamma  # attribute this obs to states

       # --- Compute expected transition counts xi_{kl}(a) ---
       # xi_{kl}(a) = alpha_hat(a,k) * P(l|k) * e_l(X_{a+1}) * beta_hat(a+1,l)
       # then normalize. This tells us: "given all the data, how probable is
       # it that the HMM was in state k at position a and state l at a+1?"
       xi_sum = np.zeros((N, N))
       for pos in range(L - 1):         # for each adjacent pair
           obs_next = seq[pos + 1]      # observation at the next position
           for k in range(N):           # source state
               for l in range(N):       # destination state
                   if obs_next >= 2:    # missing data at next position
                       e = 1.0
                   else:
                       e = hmm.emissions[obs_next, l]

                   # Unnormalized xi: forward(a,k) * trans(k,l) * emit(l) * backward(a+1,l)
                   xi_sum[k, l] += (alpha_hat[pos, k] *
                                     hmm.transitions[k, l] *
                                     e * beta_hat[pos + 1, l])

       # Normalize xi so the total sums to L-1 (one transition per adjacent pair)
       total_xi = xi_sum.sum()
       if total_xi > 0:
           xi_sum /= total_xi
           xi_sum *= (L - 1)  # scale to expected number of transitions

       return gamma_sum, xi_sum, emission_counts, ll

**Recap so far.** We now have a complete forward-backward implementation that,
given an observation sequence and current HMM parameters, produces three things:
(1) the posterior probability of each state at each position, (2) the expected
number of transitions between each pair of states, and (3) the expected emission
counts. These are everything the EM algorithm needs to update the parameters.
The next section explains how.


Step 3: The EM Algorithm
==========================

We now arrive at the central algorithm of PSMC inference. The
**Expectation-Maximization (EM) algorithm** is an iterative procedure that
alternates between two steps:

1. **E-step** (Expectation): Given the current parameters, run forward-backward
   to compute the expected sufficient statistics. "Given our current gear sizes,
   figure out how well the watch keeps time and which gear configurations are
   most likely."

2. **M-step** (Maximization): Given the expected sufficient statistics, find new
   parameters that maximize the expected log-likelihood. "Given what we learned
   about gear usage, cut new gears that would make the watch keep better time."

The algorithm repeats these two steps until convergence. Each iteration is
guaranteed (by EM theory) to increase the log-likelihood of the data -- or at
least never decrease it. The watch gets more accurate with every iteration,
converging to a (local) optimum.

.. admonition:: Probability Aside: the Expectation step

   The "Expectation" in EM refers to computing the **expected value of the
   complete-data log-likelihood** under the posterior distribution of the
   hidden states. Let's unpack this:

   - The **complete data** is the observations :math:`X` together with the
     hidden states :math:`Z`. If we knew both, computing the log-likelihood
     would be straightforward -- just count transitions, emissions, etc.

   - But the hidden states are unknown. So instead of counting, we compute
     **expected counts** -- weighted averages where the weights are the
     posterior probabilities :math:`P(Z \mid X, \Theta^{\text{old}})`.

   - This expected complete-data log-likelihood is called the **Q function**:

     .. math::

        Q(\Theta \mid \Theta^{\text{old}}) = \mathbb{E}_{Z \mid X, \Theta^{\text{old}}}
        \left[\log P(X, Z \mid \Theta)\right]

   The E-step computes the expected counts (which are all we need to evaluate
   :math:`Q` for any :math:`\Theta`). The M-step then maximizes :math:`Q`
   over :math:`\Theta`.

.. admonition:: Probability Aside: the Q function and why it matters

   The Q function deserves a closer look, because understanding it makes the
   entire EM algorithm feel natural rather than mysterious.

   For an HMM with transition matrix :math:`A`, emissions :math:`e`, and
   initial distribution :math:`a_0`, the Q function decomposes into three
   independent terms:

   .. math::

      Q = \underbrace{\sum_k \hat{c}_k^{(0)} \log a_0(k)}_{\text{initial state term}}
        + \underbrace{\sum_{k,l} \hat{c}_{kl} \log p_{kl}}_{\text{transition term}}
        + \underbrace{\sum_{k,b} \hat{c}_k^{(b)} \log e_k(b)}_{\text{emission term}}

   where:

   - :math:`\hat{c}_k^{(0)}` = expected initial state counts (proportional to :math:`\gamma_k(1)`)
   - :math:`\hat{c}_{kl}` = expected transition counts (:math:`\sum_a \xi_{kl}(a)`)
   - :math:`\hat{c}_k^{(b)}` = expected emission counts (:math:`\sum_{a: X_a = b} \gamma_k(a)`)

   Each term has the form :math:`\sum (\text{expected count}) \times \log(\text{parameter})`.
   This is exactly the form of a log-likelihood where the "data" are the
   expected counts. So maximizing :math:`Q` is like fitting parameters to
   pseudo-data that summarize what we learned about the hidden states.

   In the watch metaphor: the expected counts are like a logbook recording
   how often each gear was probably engaged and how often each transition
   probably occurred. The M-step reads this logbook and asks: "what gear
   sizes would produce exactly these usage patterns?"

.. admonition:: Probability Aside: why EM works (the ELBO)

   The EM algorithm is built on a fundamental inequality from information theory.
   For any distribution :math:`q(Z)` over hidden states and any parameters
   :math:`\Theta`:

   .. math::

      \log P(X \mid \Theta) \geq \sum_Z q(Z) \log \frac{P(X, Z \mid \Theta)}{q(Z)}

   This is the **Evidence Lower Bound (ELBO)**. The E-step sets
   :math:`q(Z) = P(Z \mid X, \Theta^{\text{old}})` (the posterior under current
   parameters), which makes the bound tight. The M-step then maximizes the
   bound over :math:`\Theta`, guaranteeing that the log-likelihood cannot decrease.

   This monotonic improvement is the fundamental guarantee: **each EM iteration
   either improves the fit or leaves it unchanged**. The log-likelihood is a
   non-decreasing sequence, bounded above (since probabilities cannot exceed 1),
   so it must converge. In the watch metaphor: each adjustment of the gears
   brings the watch closer to keeping correct time, and you can never make it
   worse by adjusting.

   One important caveat: EM converges to a **local** optimum, not necessarily
   the global optimum. Different initial gear sizes might lead to different
   final configurations. In practice, PSMC is fairly robust to initialization,
   but running from multiple starting points is sometimes advisable.

**The M-step challenge.** For a standard HMM, the M-step has closed-form solutions:
the optimal transition probabilities are just the normalized expected counts
(see :ref:`hmms`). But PSMC's transition matrix :math:`p_{kl}` is a complex
function of :math:`(\theta, \rho, \lambda_0, \ldots, \lambda_n)`, not a set of
independent parameters. The gears are all interconnected -- changing one
population size :math:`\lambda_k` affects many entries of the transition matrix
simultaneously.

So the M-step requires **numerical optimization**: maximize :math:`Q`
over :math:`(\theta, \rho, t_{\max}, \lambda_0, \ldots, \lambda_n)` using a
general-purpose optimizer. PSMC uses the Hooke-Jeeves direct search method -- a
derivative-free optimizer that works well for this kind of constrained problem.
In our implementation below, we use Nelder-Mead (a similar derivative-free method
available in scipy) for clarity.

.. admonition:: Probability Aside: the Maximization step

   The M-step asks: "given the expected counts from the E-step, which
   parameter values :math:`\Theta` maximize the Q function?"

   For a standard HMM with independent parameters, the answer has a beautiful
   closed form:

   .. math::

      \hat{p}_{kl} = \frac{\hat{c}_{kl}}{\sum_m \hat{c}_{km}}

   -- just the fraction of times state :math:`k` transitioned to state :math:`l`.
   Similarly for emissions. But PSMC's parameters are *not* the individual
   entries of the transition matrix. They are the underlying population-genetic
   quantities :math:`(\theta, \rho, \lambda_0, \ldots, \lambda_n)` that
   *generate* the transition matrix through a complex chain of formulas.

   So we must evaluate :math:`Q(\Theta)` as a black-box function: for any
   candidate :math:`\Theta`, rebuild the HMM (recompute the transition matrix,
   emissions, etc.), then evaluate the Q function using the expected counts
   from the E-step. A numerical optimizer searches over :math:`\Theta` to
   find the maximum.

   This is computationally more expensive than the closed-form solution, but
   it is still efficient because the expected counts (from the E-step) are
   fixed during the M-step. We only need to rebuild the HMM and evaluate
   :math:`Q` -- we do *not* need to re-run forward-backward.

.. code-block:: python

   from scipy.optimize import minimize

   def psmc_em_step(hmm, seq, par_map=None):
       """One EM iteration for PSMC.

       This function performs one complete E-step followed by one M-step.
       The E-step runs forward-backward to get expected counts.
       The M-step numerically optimizes the Q function over the PSMC parameters.

       Parameters
       ----------
       hmm : PSMC_HMM
       seq : ndarray
       par_map : list, optional
           Parameter grouping map (from parse_pattern).
           Maps each atomic interval to a free lambda parameter.

       Returns
       -------
       new_hmm : PSMC_HMM
           HMM with updated parameters.
       log_likelihood : float
       """
       N = hmm.N
       n = hmm.n

       # ============================================================
       # E-step: compute expected sufficient statistics
       # ============================================================
       # This runs forward-backward and extracts the three sets of
       # expected counts (gamma_sum, xi_sum, emission_counts).
       gamma_sum, xi_sum, emission_counts, ll = compute_expected_counts(hmm, seq)

       # ============================================================
       # M-step: maximize Q over the population-genetic parameters
       # ============================================================

       # Pack the current parameters into a single vector for the optimizer.
       # The parameters are: theta, rho, t_max, and the free lambdas.
       if par_map is None:
           par_map = list(range(N))   # no grouping: each interval is free
       n_free = max(par_map) + 1      # number of free lambda parameters
       n_params = n_free + 3          # total: theta + rho + t_max + lambdas

       params0 = np.zeros(n_params)
       params0[0] = hmm.theta         # current mutation rate
       params0[1] = hmm.rho           # current recombination rate
       params0[2] = hmm.t_max         # current maximum time
       # Extract the free lambdas (one per parameter group)
       for k in range(n_free):
           idx = par_map.index(k)     # first interval using this parameter
           params0[3 + k] = hmm.lambdas[idx]

       def neg_Q(params):
           """Negative Q function (we minimize this to maximize Q).

           For each candidate parameter vector, we:
           1. Rebuild the HMM (recompute transitions, emissions, initial dist)
           2. Evaluate Q using the FIXED expected counts from the E-step
           """
           # Unpack parameters (use abs to enforce positivity)
           theta = abs(params[0])
           rho = abs(params[1])
           t_max = abs(params[2])
           free_lambdas = np.abs(params[3:])

           # Expand the free lambdas to full lambdas using the par_map
           full_lambdas = np.array([free_lambdas[par_map[k]] for k in range(N)])

           # Rebuild the HMM with these candidate parameters
           try:
               transitions, emissions, initial = build_psmc_hmm(
                   n, t_max, theta, rho, full_lambdas, alpha_param=hmm.alpha_param)
           except (ValueError, RuntimeWarning):
               return 1e30  # return a large value if parameters are invalid

           # Evaluate Q = sum of (expected count * log parameter)
           Q = 0.0

           # Term 1: initial state contribution
           for k in range(N):
               if initial[k] > 0:
                   Q += gamma_sum[k] * np.log(initial[k] + 1e-300) / len(seq)

           # Term 2: transition contribution
           # Each expected transition count xi_sum[k,l] is weighted by
           # log of the transition probability under the candidate parameters
           for k in range(N):
               for l in range(N):
                   if transitions[k, l] > 0 and xi_sum[k, l] > 0:
                       Q += xi_sum[k, l] * np.log(transitions[k, l] + 1e-300)

           # Term 3: emission contribution
           # Each expected emission count is weighted by log of the emission
           # probability under the candidate parameters
           for b in range(2):  # b = 0 (hom) or 1 (het)
               for k in range(N):
                   if emissions[b, k] > 0 and emission_counts[b, k] > 0:
                       Q += emission_counts[b, k] * np.log(emissions[b, k] + 1e-300)

           return -Q  # negate because scipy minimizes

       # Run the optimizer to find the parameters that maximize Q
       result = minimize(neg_Q, params0, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-6})

       # Unpack the optimized parameters
       new_params = np.abs(result.x)
       new_theta = new_params[0]
       new_rho = new_params[1]
       new_t_max = new_params[2]
       new_free_lambdas = new_params[3:]
       new_full_lambdas = np.array([new_free_lambdas[par_map[k]] for k in range(N)])

       # Build a new HMM with the optimized parameters
       new_hmm = PSMC_HMM(n, new_theta, new_rho, new_full_lambdas,
                            new_t_max, hmm.alpha_param)

       return new_hmm, ll

   # Demonstrate one EM step
   n = 10
   theta = 0.001
   rho = theta / 5
   lambdas = np.ones(n + 1)

   hmm = PSMC_HMM(n, theta, rho, lambdas)

   # Simulate a short sequence for testing
   np.random.seed(42)
   seq, _ = simulate_psmc_input(5000, theta, rho, lambda t: 1.0)

   print(f"Initial log-likelihood: {hmm.log_likelihood(seq):.2f}")
   print(f"Initial theta: {hmm.theta:.6f}")

**Recap.** One EM iteration works as follows: (1) Run forward-backward with the
current parameters to get expected counts. (2) Numerically optimize the Q
function over the population-genetic parameters, using the expected counts as
fixed inputs. (3) Rebuild the HMM with the new parameters. The log-likelihood
is guaranteed not to decrease. Now let's see the full loop.


Step 4: The Full EM Loop
==========================

The complete PSMC inference runs EM for a fixed number of iterations (typically
20--25), printing the parameters after each round. This is the outermost loop
of the PSMC algorithm -- the process of iteratively refining the gear sizes
until the watch keeps accurate time.

.. admonition:: Probability Aside: convergence -- when to stop adjusting the gears

   How do we know when the EM algorithm has converged? There are several
   criteria:

   **1. Log-likelihood plateau.** The most common criterion: stop when the
   change in log-likelihood between iterations falls below a threshold
   (e.g., :math:`|\Delta \text{LL}| < 0.01`). The log-likelihood is a
   non-decreasing sequence (guaranteed by EM theory), so when it stops
   increasing, we have found a (local) optimum.

   **2. Parameter stability.** Stop when the parameters themselves stop
   changing: :math:`\|\Theta^{(t+1)} - \Theta^{(t)}\| < \epsilon`. This is
   often more meaningful than the log-likelihood criterion because two very
   different log-likelihoods might correspond to nearly identical parameters
   on a flat part of the likelihood surface.

   **3. Fixed iteration count.** PSMC simply runs 20--25 iterations. Li and
   Durbin found empirically that this is sufficient for convergence on
   whole-genome data. This has the advantage of predictable running time.

   In practice, PSMC uses approach (3) but monitors the log-likelihood for
   debugging. If the log-likelihood ever *decreases*, something is wrong --
   either a bug in the implementation or a numerical issue in the optimizer.

.. code-block:: python

   def psmc_inference(seq, n=63, t_max=15.0, theta_rho_ratio=5.0,
                       pattern="4+25*2+4+6", n_iters=25, alpha_param=0.1):
       """Run the full PSMC inference.

       This is the top-level function that ties everything together:
       initialize parameters, then alternate E-step and M-step for
       n_iters iterations.

       Parameters
       ----------
       seq : ndarray of shape (L,), dtype=int
           Observation sequence (0/1/2+).
       n : int
           Number of atomic time intervals - 1.
       t_max : float
           Maximum coalescence time (in coalescent units).
       theta_rho_ratio : float
           Assumed ratio theta/rho for initialization.
       pattern : str
           Parameter grouping pattern (see discretization chapter).
       n_iters : int
           Number of EM iterations.
       alpha_param : float
           Spacing parameter for time intervals.

       Returns
       -------
       results : list of dict
           Parameters at each iteration.
       """
       L = len(seq)
       N = n + 1  # number of hidden states

       # Parse the grouping pattern (e.g., "4+25*2+4+6")
       # This maps each atomic interval to a free lambda parameter
       par_map, n_free, n_intervals = parse_pattern(pattern)
       assert n_intervals == N, f"Pattern gives {n_intervals} intervals, need {N}"

       # Initialize theta from the observed heterozygosity rate.
       # If the fraction of het sites is f, then theta ~ -log(1-f).
       # This comes from inverting the emission formula: P(het) = 1 - exp(-theta*t),
       # and assuming t ~ 1 (the average coalescence time under constant pop size).
       frac_het = np.mean(seq[seq < 2])  # fraction of het sites (excluding missing)
       theta = -np.log(1.0 - frac_het)
       rho = theta / theta_rho_ratio     # initialize rho from the assumed ratio

       # Initialize all lambdas to 1 (constant population size)
       # The EM algorithm will adjust them to match the data
       free_lambdas = np.ones(n_free)
       full_lambdas = np.array([free_lambdas[par_map[k]] for k in range(N)])

       # Build the initial HMM
       hmm = PSMC_HMM(n, theta, rho, full_lambdas, t_max, alpha_param)

       results = []
       print(f"PSMC inference: {L} bins, {N} intervals, {n_free} free params")
       print(f"Initial theta={theta:.6f}, rho={rho:.6f}")

       for iteration in range(n_iters):
           # --- E-step ---
           # Run forward-backward to compute expected sufficient statistics
           gamma_sum, xi_sum, emission_counts, ll = compute_expected_counts(
               hmm, seq)

           # Record the current state for later analysis
           results.append({
               'iteration': iteration,
               'log_likelihood': ll,
               'theta': hmm.theta,
               'rho': hmm.rho,
               'lambdas': hmm.lambdas.copy(),
           })

           print(f"  Iteration {iteration}: LL = {ll:.2f}, "
                 f"theta = {hmm.theta:.6f}, rho = {hmm.rho:.6f}")

           # --- M-step ---
           # Update parameters by maximizing the Q function
           hmm, _ = psmc_em_step(hmm, seq, par_map)

       return results


.. admonition:: Convergence in practice

   PSMC typically converges within 20--25 iterations. The log-likelihood should
   increase monotonically -- this is guaranteed by EM theory, as we explained
   in the Probability Aside above. If it doesn't, there's a bug in the
   implementation.

   You can verify convergence by plotting the log-likelihood across iterations.
   It should rise steeply in the first few iterations (the initial constant-population
   guess is far from the truth) and then plateau as the inferred :math:`\lambda_k`
   values settle into their final configuration.

   The Q function value (the expected complete-data log-likelihood under the
   current parameters) should also increase after the M-step. PSMC reports
   both Q values (before and after maximization) at each iteration for debugging.


Step 5: Multiple Sequences and Missing Data
=============================================

So far we have described the algorithm for a single observation sequence. In
practice, a diploid genome is split into chromosomes (or even sub-chromosomal
segments for bootstrapping). PSMC processes each sequence independently in the
E-step and sums the expected counts before the M-step.

This works because the sufficient statistics are additive: the expected counts
from two independent sequences can simply be added together. If sequence 1
tells us that state :math:`k` was visited an expected 500 times, and sequence 2
tells us 300 times, then the combined estimate is 800 times. The M-step then
uses these combined counts as if they came from a single long sequence.

In the watch metaphor: each chromosome is an independent test of the watch's
accuracy. By combining the evidence from all chromosomes, we get a more
reliable picture of which gear sizes are correct.

.. code-block:: python

   def psmc_em_multiple_seqs(hmm, sequences, par_map):
       """EM step with multiple sequences.

       Each sequence is processed independently in the E-step (separate
       forward-backward runs), and the expected counts are summed before
       the M-step. This is mathematically equivalent to treating all
       sequences as independent observations of the same underlying HMM.

       Parameters
       ----------
       hmm : PSMC_HMM
       sequences : list of ndarray
           Each element is an observation sequence for one chromosome.
       par_map : list

       Returns
       -------
       new_hmm : PSMC_HMM
       total_ll : float
       """
       N = hmm.N

       # Accumulate expected counts across all sequences
       total_gamma = np.zeros(N)
       total_xi = np.zeros((N, N))
       total_emission = np.zeros((2, N))
       total_ll = 0.0

       for seq in sequences:
           # Run forward-backward on this sequence independently
           gamma, xi, emission, ll = compute_expected_counts(hmm, seq)
           # Add this sequence's expected counts to the running totals
           total_gamma += gamma
           total_xi += xi
           total_emission += emission
           total_ll += ll

       # M-step uses accumulated counts
       # (same optimization as single sequence, just with summed counts)

       return total_ll

**Missing data** is handled elegantly: when :math:`X_a` is missing (encoded as
a value >= 2), the emission probability is set to 1 for all states:

.. math::

   e_k(\text{missing}) = 1 \quad \text{for all } k

This means missing positions contribute nothing to the likelihood -- the HMM
simply passes through them, maintaining the transition dynamics but receiving no
information from the data. The transition structure is preserved (we still model
the passage of genomic distance, which matters for recombination), but the
observation at that position provides zero evidence about the hidden state.

This is a standard HMM technique (covered in :ref:`hmms`) and requires no
special code beyond the ``if obs >= 2: e = 1.0`` checks you have already
seen in the forward, backward, and expected-counts functions above.


Step 6: Understanding What EM Learns
=======================================

After running EM to convergence, we can use the forward-backward algorithm one
final time to understand what the model has learned. The **posterior decoding**
gives us, at each position along the genome, a probability distribution over
coalescence time intervals.

This is the moment where the watch reveals its reading: at each position, the
posterior tells us how deep in time the most recent common ancestor lies. In
regions with many heterozygous sites, the posterior will peak at large :math:`k`
(deep coalescence, ancient MRCA). In regions with few heterozygous sites, it
will peak at small :math:`k` (recent MRCA). The spatial pattern of these
posteriors -- how the coalescence time waxes and wanes along the genome --
reflects the history of recombination and population size changes.

.. code-block:: python

   def posterior_decoding(hmm, seq):
       """Compute the posterior state probabilities at each position.

       This is the final step of PSMC analysis: after EM has found the
       best parameters, run forward-backward one more time to decode
       the hidden states. The result is a probability distribution over
       coalescence time intervals at each genomic position.

       Parameters
       ----------
       hmm : PSMC_HMM
       seq : ndarray

       Returns
       -------
       posterior : ndarray of shape (L, N)
           posterior[a, k] = P(Z_a = k | X_1, ..., X_L)
       map_states : ndarray of shape (L,)
           Maximum a posteriori state at each position.
       """
       L = len(seq)
       N = hmm.N

       alpha_hat, _ = hmm.forward_scaled(seq)          # forward pass
       beta_hat = hmm.backward_scaled(seq, alpha_hat)  # backward pass

       posterior = np.zeros((L, N))
       for pos in range(L):
           # Posterior = forward * backward, then normalize
           gamma = alpha_hat[pos] * beta_hat[pos]
           total = gamma.sum()
           if total > 0:
               posterior[pos] = gamma / total
           else:
               posterior[pos] = 1.0 / N  # uniform fallback (should not happen)

       # The MAP (Maximum A Posteriori) state is the most probable state
       # at each position -- the single best guess for the coalescence interval
       map_states = np.argmax(posterior, axis=1)
       return posterior, map_states

   # After running EM, decode the hidden states
   # posterior, map_states = posterior_decoding(final_hmm, seq)
   # print(f"Most common state: {np.bincount(map_states).argmax()}")

.. admonition:: What the posterior tells us

   At each position, the posterior gives a probability distribution over
   coalescence time intervals. In regions with many heterozygous sites,
   the posterior will peak at large :math:`k` (deep coalescence, ancient MRCA).
   In regions with few hets, it will peak at small :math:`k` (recent MRCA).

   The beauty of PSMC is that these local patterns, aggregated over the entire
   genome, allow inference of the global population size history :math:`N(t)`.
   The :math:`\lambda_k` parameters -- the gear sizes that EM has refined over
   many iterations -- directly encode the population size at each time interval.
   The next chapter (:ref:`psmc_decoding`) shows how to read these gear sizes
   as a population size curve and scale them to real biological units.


Chapter Summary
=================

Let's trace the complete path we have traveled in this chapter:

1. **We assembled the HMM** by collecting the initial distribution, transition
   matrix, and emission probabilities from the discretization chapter
   (:ref:`psmc_discretization`) into a single ``PSMC_HMM`` class.

2. **We implemented the forward-backward algorithm**, which computes posterior
   state probabilities by combining evidence from both directions along the
   sequence. The forward pass looks at data from the left; the backward pass
   looks from the right; together they give us the posterior at every position
   (see :ref:`hmms` for the foundational version).

3. **We extracted expected sufficient statistics** -- the expected state
   occupancies, transition counts, and emission counts -- which summarize
   everything the data tells us about the parameters.

4. **We built the EM loop**: alternate between computing expected counts
   (E-step) and finding parameters that maximize the expected log-likelihood
   (M-step). Each iteration improves the fit, guaranteed by the ELBO inequality.

5. **We handled multiple sequences and missing data**, exploiting the additivity
   of sufficient statistics and the standard HMM trick of setting missing-data
   emissions to 1.

6. **We decoded the hidden states** using the final parameters, producing a
   posterior distribution over coalescence times at every position.

In the watch metaphor: we started with a box of gears (the HMM components from
the discretization chapter), assembled them into a working mechanism (the
forward-backward algorithm), and then used the EM algorithm to iteratively
adjust every gear until the watch keeps accurate time. The next chapter
(:ref:`psmc_decoding`) shows how to read the dial -- converting the inferred
parameters into a population size history in real biological units.


Exercises
=========

.. admonition:: Exercise 1: Build and verify the HMM

   Construct the PSMC HMM for a constant population (:math:`\lambda_k = 1`).
   Verify that:
   (a) forward probabilities can be computed without numerical issues for a
   10,000-bin sequence,
   (b) the log-likelihood increases across EM iterations,
   (c) :math:`\hat{\theta}` converges to the true value on simulated data.

.. admonition:: Exercise 2: EM on simulated bottleneck data

   Simulate 100,000 bins of data under a population that has
   :math:`\lambda(t) = 1` for :math:`t < 0.5`, :math:`\lambda(t) = 0.1` for
   :math:`0.5 \leq t < 1.5`, and :math:`\lambda(t) = 1` for :math:`t > 1.5`.
   Run PSMC on this data and compare the inferred :math:`\lambda_k` to the truth.

.. admonition:: Exercise 3: The effect of sequence length

   Run PSMC on sequences of length 1000, 10000, 100000, and 1000000 (all simulated
   under the same demographic model). How does the accuracy of the inferred
   :math:`\lambda_k` improve with more data? At what length does the inference
   become reliable?

.. admonition:: Exercise 4: Watch the EM gears turn

   Run 25 EM iterations on simulated data and record the :math:`\lambda_k` values
   at each iteration. Plot them as a function of iteration number. You should see
   them start at 1.0 (the initial guess) and gradually converge to the true
   population size history. How many iterations does it take for the curves to
   stabilize? Does convergence happen uniformly across all time intervals, or do
   some intervals converge faster than others?

Next: :ref:`psmc_decoding` -- reading the population size history from the inferred parameters.


Solutions
=========

.. admonition:: Solution 1: Build and verify the HMM

   We construct the PSMC HMM for a constant population and verify three properties:
   numerical stability of forward probabilities, monotonic log-likelihood increase
   under EM, and convergence of :math:`\hat{\theta}` to the true value.

   .. code-block:: python

      import numpy as np

      # Setup
      n = 10
      theta_true = 0.001
      rho_true = theta_true / 5
      lambdas = np.ones(n + 1)

      # Build HMM with true parameters
      hmm = PSMC_HMM(n, theta_true, rho_true, lambdas)

      # Simulate a 10,000-bin sequence
      np.random.seed(42)
      seq, _ = simulate_psmc_input(10000, theta_true, rho_true, lambda t: 1.0)

      # (a) Verify forward probabilities compute without issues
      alpha_hat, ll = hmm.forward_scaled(seq)
      print(f"(a) Forward algorithm on 10,000 bins:")
      print(f"  Log-likelihood: {ll:.2f}")
      print(f"  alpha_hat min: {alpha_hat.min():.2e}")
      print(f"  alpha_hat max: {alpha_hat.max():.2e}")
      print(f"  Any NaN? {np.any(np.isnan(alpha_hat))}")
      print(f"  Any Inf? {np.any(np.isinf(alpha_hat))}")

      # (b) Verify log-likelihood increases across EM iterations
      # Start with a perturbed theta to give EM room to improve
      hmm_init = PSMC_HMM(n, theta_true * 0.5, rho_true, lambdas)
      log_likelihoods = []

      current_hmm = hmm_init
      for i in range(10):
          ll_i = current_hmm.log_likelihood(seq)
          log_likelihoods.append(ll_i)
          current_hmm, _ = psmc_em_step(current_hmm, seq)
          print(f"  Iteration {i}: LL = {ll_i:.2f}, "
                f"theta = {current_hmm.theta:.6f}")

      # Check monotonicity
      for i in range(1, len(log_likelihoods)):
          assert log_likelihoods[i] >= log_likelihoods[i-1] - 1e-6, \
              f"LL decreased at iteration {i}!"
      print(f"\n(b) Log-likelihood monotonically non-decreasing: PASSED")

      # (c) Check theta convergence
      print(f"\n(c) Theta convergence:")
      print(f"  True theta:  {theta_true:.6f}")
      print(f"  Final theta: {current_hmm.theta:.6f}")
      print(f"  Relative error: "
            f"{abs(current_hmm.theta - theta_true)/theta_true:.4f}")

   **(a)** The scaled forward algorithm should produce no NaN or Inf values, even
   for a 10,000-bin sequence. The scaling (normalizing at each position) prevents
   the underflow that would occur with raw forward probabilities, which shrink
   exponentially with sequence length.

   **(b)** The log-likelihood must be monotonically non-decreasing -- this is
   guaranteed by EM theory (the ELBO argument). Any decrease indicates a bug.

   **(c)** After ~10 iterations, :math:`\hat{\theta}` should be within a few
   percent of the true value. The estimate improves with longer sequences; with
   10,000 bins, expect a relative error of roughly 5--10%.

.. admonition:: Solution 2: EM on simulated bottleneck data

   We simulate data under a bottleneck model and run PSMC to see if it recovers
   the true population history.

   .. code-block:: python

      import numpy as np

      # Bottleneck model: lambda=1 for t<0.5, lambda=0.1 for 0.5<=t<1.5, lambda=1 for t>=1.5
      def bottleneck_lambda(t):
          if t < 0.5:
              return 1.0
          elif t < 1.5:
              return 0.1
          else:
              return 1.0

      # Simulate 100,000 bins
      np.random.seed(123)
      theta = 0.001
      rho = theta / 5
      seq, _ = simulate_psmc_input(100000, theta, rho, bottleneck_lambda)

      print(f"Observed heterozygosity: {np.mean(seq):.4f}")

      # Run PSMC with n=20 intervals for tractability
      n = 20
      results = psmc_inference(seq, n=n, t_max=15.0, n_iters=20,
                                pattern=f"{n+1}*1")

      # Compare inferred lambdas to truth
      final = results[-1]
      t = compute_time_intervals(n, t_max=15.0)
      print(f"\nInferred vs. true lambda:")
      print(f"{'Interval':>10} {'t_mid':>8} {'True':>8} {'Inferred':>10}")
      print("-" * 40)
      for k in range(n + 1):
          t_mid = (t[k] + t[k+1]) / 2.0
          lam_true = bottleneck_lambda(t_mid)
          lam_inferred = final['lambdas'][k]
          print(f"{k:>10} {t_mid:>8.3f} {lam_true:>8.2f} {lam_inferred:>10.4f}")

   The inferred :math:`\lambda_k` values should show a clear dip (values
   significantly less than 1) in the intervals corresponding to
   :math:`t \in [0.5, 1.5]`, matching the bottleneck. The depth of the
   inferred bottleneck may not exactly reach 0.1 due to the discretization
   smoothing the sharp edges, but the qualitative pattern should be clear.
   The constant-size intervals (:math:`t < 0.5` and :math:`t > 1.5`)
   should have :math:`\lambda_k \approx 1.0`.

.. admonition:: Solution 3: The effect of sequence length

   We run PSMC on sequences of increasing length to study how statistical power
   scales with data. The key insight is that PSMC accuracy improves with
   :math:`\sqrt{L}` (standard statistical scaling), and a minimum of ~50,000
   bins is needed for reliable inference.

   .. code-block:: python

      import numpy as np

      def bottleneck_lambda(t):
          if t < 0.5:
              return 1.0
          elif t < 1.5:
              return 0.1
          else:
              return 1.0

      theta = 0.001
      rho = theta / 5
      n = 10  # fewer intervals for speed

      lengths = [1000, 10000, 100000, 1000000]
      t = compute_time_intervals(n, t_max=15.0)

      for L in lengths:
          np.random.seed(42)
          seq, _ = simulate_psmc_input(L, theta, rho, bottleneck_lambda)
          results = psmc_inference(seq, n=n, t_max=15.0, n_iters=15,
                                    pattern=f"{n+1}*1")
          final_lambdas = results[-1]['lambdas']

          # Compute mean squared error against truth
          mse = 0.0
          for k in range(n + 1):
              t_mid = (t[k] + t[k+1]) / 2.0
              lam_true = bottleneck_lambda(t_mid)
              mse += (final_lambdas[k] - lam_true) ** 2
          mse /= (n + 1)

          print(f"L={L:>8}: RMSE = {np.sqrt(mse):.4f}, "
                f"final LL = {results[-1]['log_likelihood']:.2f}")

   **Expected results:**

   - **L = 1,000:** Very noisy. The inferred :math:`\lambda_k` will be far from
     the truth. With ~1 heterozygous site per 1,000 bins, there is almost no
     signal. RMSE will be high (:math:`> 0.5`).

   - **L = 10,000:** Marginal. The bottleneck may be faintly visible but with
     large errors. RMSE improves but remains substantial.

   - **L = 100,000:** Reliable. The bottleneck should be clearly detected with
     quantitative accuracy. This is the minimum recommended sequence length for
     PSMC. RMSE should be below 0.2.

   - **L = 1,000,000:** Excellent. The inferred :math:`\lambda_k` closely match
     the true values. RMSE should be well below 0.1.

   The accuracy scales approximately as :math:`1/\sqrt{L}`, consistent with
   standard statistical theory -- doubling the data reduces the error by a
   factor of :math:`\sqrt{2}`.

.. admonition:: Solution 4: Watch the EM gears turn

   We track :math:`\lambda_k` across 25 EM iterations to visualize how the
   parameters converge from the initial guess to the final estimate.

   .. code-block:: python

      import numpy as np

      def bottleneck_lambda(t):
          if t < 0.5:
              return 1.0
          elif t < 1.5:
              return 0.1
          else:
              return 1.0

      np.random.seed(42)
      n = 10
      theta = 0.001
      rho = theta / 5
      seq, _ = simulate_psmc_input(100000, theta, rho, bottleneck_lambda)
      t = compute_time_intervals(n, t_max=15.0)

      # Run PSMC and record lambdas at each iteration
      results = psmc_inference(seq, n=n, t_max=15.0, n_iters=25,
                                pattern=f"{n+1}*1")

      # Print lambda trajectories for selected intervals
      # Pick one interval in the bottleneck and one outside
      bottleneck_interval = None
      normal_interval = None
      for k in range(n + 1):
          t_mid = (t[k] + t[k+1]) / 2.0
          if bottleneck_interval is None and 0.5 < t_mid < 1.5:
              bottleneck_interval = k
          if normal_interval is None and t_mid < 0.3:
              normal_interval = k

      print(f"Tracking interval {bottleneck_interval} (in bottleneck) "
            f"and interval {normal_interval} (outside):")
      print(f"{'Iter':>5} {'lambda_bottleneck':>18} {'lambda_normal':>15} "
            f"{'LL':>12}")
      print("-" * 55)
      for r in results:
          lam_b = r['lambdas'][bottleneck_interval]
          lam_n = r['lambdas'][normal_interval]
          print(f"{r['iteration']:>5} {lam_b:>18.4f} {lam_n:>15.4f} "
                f"{r['log_likelihood']:>12.2f}")

   **Expected behavior:**

   - **All :math:`\lambda_k` start at 1.0** (the initial flat guess).

   - **The bottleneck intervals converge fastest.** Within 5--10 iterations,
     the :math:`\lambda_k` for intervals in :math:`[0.5, 1.5]` drop sharply
     toward 0.1. This is because the emission signal is strongest there -- the
     pronounced lack of heterozygosity in those time intervals provides a clear
     gradient for the M-step optimizer.

   - **The non-bottleneck intervals converge more slowly.** They remain near 1.0
     but may fluctuate slightly before settling. Intervals in the very recent or
     very ancient past converge last because they have the least statistical
     power (fewest recombination events).

   - **Convergence is not uniform.** Intervals with more expected recombination
     events (as measured by :math:`C_\sigma \sigma_k`) converge faster because
     the expected sufficient statistics are more precisely estimated there. This
     is analogous to a watch where some gears engage frequently and are quickly
     tuned, while others engage rarely and take longer to adjust.

   - **The log-likelihood plateaus** after roughly 15--20 iterations, indicating
     convergence. The rate of improvement should decrease exponentially -- large
     gains in the first few iterations, then diminishing returns.
