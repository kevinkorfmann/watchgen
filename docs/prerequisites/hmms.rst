.. _hmms:

======================
Hidden Markov Models
======================

   *The states are hidden, but the mechanism is not.*

Think of a mechanical watch. You can see the hands sweep across the dial -- the
hours, the minutes, the seconds -- but the gears and springs that drive those
hands are concealed behind the case. You observe the *output* of the mechanism,
not the mechanism itself. A Hidden Markov Model is a mathematical formalization
of exactly this situation: a system whose internal workings are invisible, but
whose observable behavior carries clues about what is happening inside.

In population genetics, the analogy is almost literal. When we sequence genomes,
we see the mutations -- the positions where alleles differ between individuals.
These are the hands on the dial. What we cannot see directly are the genealogical
relationships: which branches of the ancestral tree each genomic position "sits
on," and where recombination events reshuffled the ancestry. These hidden
genealogical states are the gears behind the watch face.

This chapter develops the mathematical machinery of HMMs from the ground up,
building toward the specific form used in ARG inference tools like SINGER. If you
have not yet read the chapters on :ref:`coalescent_theory` and :ref:`args`, now
is a good time -- the HMM framework here is the bridge that connects those
theoretical concepts to practical inference algorithms.


Why HMMs for ARG Inference?
============================

Before diving into formalism, it is worth understanding *why* HMMs are the right
tool for this problem.

In the :ref:`args` chapter, we saw that an ARG encodes a sequence of marginal
trees along the genome, with recombination events causing transitions from one
tree to the next. In the :ref:`smc` chapter, we saw that the SMC approximation
makes this sequence Markov: the tree at position :math:`\ell` depends only on the
tree at position :math:`\ell - 1`, not on the entire history.

This is exactly the structure an HMM is designed to handle:

- We have a sequence of **hidden states** (the marginal tree, or more
  specifically, the branch of the tree to which a lineage belongs) that evolves
  along the genome.
- At each position, the hidden state **emits** an observation (the allele we
  see -- whether there is a mutation or not).
- The hidden states form a **Markov chain**: the state at position :math:`\ell`
  depends only on the state at position :math:`\ell - 1`.

The goal of ARG inference is to go from the observations (the genome data) back
to the hidden states (the genealogical relationships). HMMs give us principled,
efficient algorithms for doing exactly this.


A Warm-Up Example: Weather and Umbrellas
==========================================

Before tackling genetics, let us build intuition with a simpler example.

Suppose you are locked in a windowless office. Each day, a colleague visits you,
and you notice whether they are carrying an umbrella. You cannot see the weather
outside (it is *hidden*), but you can observe the umbrella (the *emission*). Over
time, you want to figure out what the weather has been doing.

The weather follows a simple pattern:

- If today is sunny, there is a 90% chance tomorrow is also sunny (weather tends
  to persist).
- If today is rainy, there is an 80% chance tomorrow is also rainy.
- When it is sunny, your colleague carries an umbrella only 10% of the time.
- When it is rainy, your colleague carries an umbrella 80% of the time.

This is a two-state HMM:

- **Hidden states**: Sunny, Rainy
- **Observations**: Umbrella, No umbrella
- **Transition probabilities**: The chance of weather changing from one day to
  the next
- **Emission probabilities**: The chance of seeing an umbrella given the weather

Now, if you observe the sequence (Umbrella, Umbrella, No umbrella, Umbrella),
what was the weather? This is the fundamental HMM inference question. The forward
algorithm, which we develop below, gives us a principled way to answer it.

In genetics, replace "sunny/rainy" with "which branch of the genealogical tree
the lineage sits on" and "umbrella/no umbrella" with "derived allele / ancestral
allele." The logic is identical; only the vocabulary changes.


.. admonition:: Probability Aside -- Conditional Probability

   Several probability concepts appear throughout this chapter. Let us establish
   them now.

   **Conditional probability** is the probability of an event given that another
   event has occurred. We write :math:`P(A \mid B)` and read it as "the
   probability of A given B." For example, :math:`P(\text{umbrella} \mid
   \text{rain}) = 0.8` means that if it is raining, there is an 80% chance of
   seeing an umbrella.

   Formally:

   .. math::

      P(A \mid B) = \frac{P(A, B)}{P(B)}

   where :math:`P(A, B)` is the **joint probability** -- the probability that
   both A and B occur together. Rearranging gives the **product rule**:

   .. math::

      P(A, B) = P(B) \cdot P(A \mid B)

   This generalizes to the **chain rule of probability**. For three events:

   .. math::

      P(A, B, C) = P(A) \cdot P(B \mid A) \cdot P(C \mid A, B)

   And for a whole sequence:

   .. math::

      P(X_1, X_2, \ldots, X_L) = P(X_1) \cdot P(X_2 \mid X_1) \cdot
      P(X_3 \mid X_1, X_2) \cdots P(X_L \mid X_1, \ldots, X_{L-1})

   Finally, **marginalizing** (or "summing out") a variable means computing the
   probability of an event by summing over all possible values of another
   variable:

   .. math::

      P(A) = \sum_{b} P(A, B = b)

   If you know the joint probability of A and B for every possible value of B,
   you can recover the probability of A alone by summing over all B values. This
   is sometimes called the **law of total probability**, and it appears
   repeatedly in the forward algorithm.


The Core Idea
==============

A **Hidden Markov Model (HMM)** is a probabilistic model with two layers:

1. A **hidden state sequence** :math:`(Z_1, Z_2, \ldots, Z_L)` that evolves
   according to a Markov chain -- each state depends only on the previous one.

2. An **observed sequence** :math:`(X_1, X_2, \ldots, X_L)` where each
   observation depends only on the hidden state at that position.

You observe :math:`X`, but you want to infer :math:`Z`. This is exactly the
situation in ARG inference:

- The **hidden states** are the branches of the genealogical tree at each
  genomic position
- The **observations** are the alleles (mutations) at each position
- The **transitions** between states represent recombination events

.. admonition:: The Markov Property -- Intuition

   The word "Markov" refers to a specific kind of forgetfulness. A process is
   Markov if its future behavior depends only on where it is *right now*, not on
   how it got there. In the weather example, tomorrow's weather depends only on
   today's weather -- it does not matter whether today's sun came after a week of
   rain or a month of clear skies.

   Formally, if :math:`Z_1, Z_2, \ldots` is a Markov chain, then:

   .. math::

      P(Z_{\ell+1} \mid Z_\ell, Z_{\ell-1}, \ldots, Z_1) = P(Z_{\ell+1} \mid Z_\ell)

   This is a drastic simplification. Without it, to predict the state at
   position :math:`\ell + 1`, we would need to consider the entire history of
   states -- an exponentially growing space. With the Markov property, we only
   need the current state.

   For ARGs, the :ref:`smc` approximation is precisely what gives us this
   property: it ensures that the marginal tree at genomic position :math:`\ell`
   depends only on the tree at position :math:`\ell - 1`, making the sequence of
   trees a Markov chain (see :ref:`smc`).


Formal Definition
==================

An HMM is defined by four components:

- **State space** :math:`\mathcal{S} = \{s_1, \ldots, s_K\}` -- the :math:`K`
  possible hidden states

- **Initial distribution** :math:`\pi_i = P(Z_1 = s_i)` -- probability of
  starting in state :math:`s_i`

- **Transition matrix** :math:`A_{ij} = P(Z_\ell = s_j \mid Z_{\ell-1} = s_i)` --
  probability of moving from state :math:`s_i` to :math:`s_j`

- **Emission probabilities** :math:`e_j(x) = P(X_\ell = x \mid Z_\ell = s_j)` --
  probability of observing :math:`x` when in state :math:`s_j`

Let us unpack the last two, since they carry all the modeling power.

**Transition probabilities** describe how the hidden state changes from one
position to the next. In the weather example, the transition from Sunny to Rainy
has probability 0.10. In genetics, a transition corresponds to a recombination
event: the lineage jumps from one branch of the genealogical tree to another. A
high transition probability between two states means recombination frequently
causes the lineage to switch between those branches. A low transition probability
(or a high probability of staying in the same state) means the same tree branch
tends to persist across adjacent genomic positions.

**Emission probabilities** describe what you *observe* given the hidden state.
The hidden state "emits" an observation, like a machine dropping a colored ball
into a bin. In the weather model, the rainy state emits "umbrella" with
probability 0.8. In genetics, the emission probability encodes the chance of
seeing a particular allele given that the lineage sits on a specific branch of
the tree. Under the infinite-sites mutation model (see :ref:`coalescent_theory`),
this depends on the branch length measured in coalescent time units -- longer
branches accumulate more mutations, so a mutation is more likely to be observed
on a long branch than a short one.

.. admonition:: Matrix Notation -- What Is a Transition Matrix?

   A **matrix** is a rectangular array of numbers. The transition matrix
   :math:`A` has :math:`K` rows and :math:`K` columns, where :math:`K` is the
   number of hidden states. The entry :math:`A_{ij}` in row :math:`i` and column
   :math:`j` gives the probability of transitioning from state :math:`s_i` to
   state :math:`s_j`.

   Each row of :math:`A` must sum to 1, because from any state, the process must
   go *somewhere*:

   .. math::

      \sum_{j=1}^{K} A_{ij} = 1 \quad \text{for all } i

   When we later write expressions like :math:`\sum_i \alpha_i \cdot A_{ij}`,
   this is computing a **weighted sum** over column :math:`j` of the matrix,
   where the weights are the :math:`\alpha_i` values. In matrix notation, this
   corresponds to a matrix-vector multiplication. The important thing is that it
   combines information from *all* possible previous states to compute the
   probability of being in state :math:`j` next.

Let's implement a basic HMM:

.. code-block:: python

   import numpy as np

   class HMM:
       """A Hidden Markov Model.

       Parameters
       ----------
       initial : ndarray of shape (K,)
           Initial state distribution pi.
       transition : ndarray of shape (K, K)
           Transition matrix A[i, j] = P(Z_l = j | Z_{l-1} = i).
       emission : callable
           emission(state, observation) returns P(X = obs | Z = state).
       """
       def __init__(self, initial, transition, emission):
           self.initial = initial           # pi: starting probabilities
           self.transition = transition     # A: K x K transition matrix
           self.emission = emission         # function(state, obs) -> probability
           self.K = len(initial)            # number of hidden states

   # Example: a simple 2-state HMM (the weather model)
   # States: 0 = Sunny, 1 = Rainy
   # Observations: 0 = No umbrella, 1 = Umbrella
   hmm = HMM(
       initial=np.array([0.5, 0.5]),         # equal chance of starting sunny or rainy
       transition=np.array([[0.95, 0.05],     # Sunny -> Sunny: 0.95, Sunny -> Rainy: 0.05
                            [0.10, 0.90]]),    # Rainy -> Sunny: 0.10, Rainy -> Rainy: 0.90
       # The lambda function takes a state index s and observation x,
       # and returns the emission probability P(X=x | Z=s).
       # For state 0 (Sunny): P(no umbrella)=0.9, P(umbrella)=0.1
       # For state 1 (Rainy): P(no umbrella)=0.2, P(umbrella)=0.8
       emission=lambda s, x: [0.9, 0.1][x] if s == 0 else [0.2, 0.8][x]
   )


The Forward Algorithm
======================

We now arrive at the first major computational tool: the **forward algorithm**.
This algorithm answers the question: "Given a sequence of observations, how
likely is each hidden state at each position?"

Think of it this way. Behind the watch face, many different gear configurations
could produce the hand positions you observe. The forward algorithm systematically
counts up the probability of *every* possible gear configuration that is
consistent with what you see, working from left to right along the sequence. At
each position, it combines two sources of information: what the current
observation tells us about the hidden state (emission), and what the previous
position's state probabilities tell us about where we could have come from
(transition).

The forward algorithm computes :math:`P(X_1, \ldots, X_\ell, Z_\ell = s_j)`
-- the joint probability of the observations up to position :math:`\ell` and
being in state :math:`s_j`.

.. admonition:: Probability Aside -- Joint Probability

   The quantity :math:`P(X_1, \ldots, X_\ell, Z_\ell = s_j)` is a **joint
   probability**. It answers the question: "What is the probability that the
   observations are :math:`X_1, \ldots, X_\ell` *and simultaneously* the
   hidden state at position :math:`\ell` is :math:`s_j`?"

   This is not the same as asking "what is the probability that the state is
   :math:`s_j` given the observations?" (which would be a conditional
   probability). The joint probability is smaller, because it also accounts for
   how likely the observations themselves are. We will see later that by
   normalizing, we can convert between the two.

Define the **forward variable**:

.. math::

   \alpha_j(\ell) = P(X_1, \ldots, X_\ell, Z_\ell = s_j)

This is the joint probability of having observed the data :math:`X_1, \ldots, X_\ell`
*and* being in state :math:`s_j` at position :math:`\ell`.

**Initialization** (:math:`\ell = 1`):

.. math::

   \alpha_j(1) = P(X_1, Z_1 = s_j) = P(Z_1 = s_j) \cdot P(X_1 \mid Z_1 = s_j) = \pi_j \cdot e_j(X_1)

.. admonition:: Probability Aside -- The Chain Rule at Work

   The initialization step uses the **product rule** (a special case of the chain
   rule). We want the joint probability :math:`P(X_1, Z_1 = s_j)`, which is the
   probability of two things happening together: the state is :math:`s_j` and the
   observation is :math:`X_1`.

   By the product rule: :math:`P(A, B) = P(A) \cdot P(B \mid A)`. Here,
   :math:`A` is ":math:`Z_1 = s_j`" and :math:`B` is ":math:`X_1`." So:

   .. math::

      P(X_1, Z_1 = s_j) = P(Z_1 = s_j) \cdot P(X_1 \mid Z_1 = s_j)
      = \pi_j \cdot e_j(X_1)

   The first factor is "how likely is this starting state?" and the second is
   "given this state, how likely is the observation?" Multiplying them gives the
   joint probability of both.


**Recursion** (:math:`\ell = 2, \ldots, L`):

To derive the recursion, we use the law of total probability to sum over all
possible previous states, and then apply the two key HMM independence
assumptions:

.. math::

   \alpha_j(\ell) &= P(X_1, \ldots, X_\ell, Z_\ell = s_j) \\
   &= \sum_{i=1}^{K} P(X_1, \ldots, X_\ell, Z_{\ell-1} = s_i, Z_\ell = s_j) \\
   &= \sum_{i=1}^{K} P(X_1, \ldots, X_{\ell-1}, Z_{\ell-1} = s_i) \cdot
   P(Z_\ell = s_j \mid Z_{\ell-1} = s_i) \cdot P(X_\ell \mid Z_\ell = s_j)

Let us walk through each step carefully:

- **Line 1 to Line 2**: We introduced the previous state :math:`Z_{\ell-1}` and
  summed over all its possible values. This is marginalizing -- we do not know
  what the previous state was, so we consider all possibilities. (See the
  Probability Aside on marginalizing above.)

- **Line 2 to Line 3**: We factored the joint probability using the chain rule
  and the two HMM independence assumptions (detailed below). The key insight is
  that the big joint probability :math:`P(X_1, \ldots, X_\ell, Z_{\ell-1} = s_i,
  Z_\ell = s_j)` splits into three manageable pieces.

The third line relies on two HMM properties:

1. **Markov property**: :math:`P(Z_\ell = s_j \mid Z_{\ell-1} = s_i, X_1, \ldots, X_{\ell-1}) = P(Z_\ell = s_j \mid Z_{\ell-1} = s_i) = A_{ij}`.
   The next state depends only on the current state, not on older states or observations.

2. **Conditional independence of emissions**: :math:`P(X_\ell \mid Z_\ell = s_j, Z_{\ell-1}, X_1, \ldots, X_{\ell-1}) = P(X_\ell \mid Z_\ell = s_j) = e_j(X_\ell)`.
   Each observation depends only on the hidden state at that position.

Substituting our definitions:

.. math::

   \alpha_j(\ell) = e_j(X_\ell) \sum_{i=1}^{K} \alpha_i(\ell-1) \cdot A_{ij}

Read this equation aloud: "The forward probability of being in state :math:`j`
at position :math:`\ell` equals the emission probability of the current
observation in state :math:`j`, multiplied by the sum over all previous states
:math:`i` of (the forward probability of being in state :math:`i` at the
previous position, times the transition probability from :math:`i` to
:math:`j`)."

The sum :math:`\sum_i \alpha_i(\ell-1) \cdot A_{ij}` aggregates contributions
from every possible predecessor state, weighted by how likely each predecessor
is and how likely it is to transition to state :math:`j`. The emission term
:math:`e_j(X_\ell)` then adjusts for how well state :math:`j` explains the
current observation.

Note the computational cost: for each of the :math:`K` states at position :math:`\ell`,
we sum over :math:`K` states at position :math:`\ell - 1`. This is :math:`O(K^2)` per
position and :math:`O(LK^2)` total.

**Likelihood** (marginalizing over final state):

.. math::

   P(X_1, \ldots, X_L) = \sum_{j=1}^{K} \alpha_j(L)

This final step sums over all possible hidden states at the last position. We do
not know (or care about) which state the system ended in -- we want the total
probability of the observations regardless of the final state. This is another
application of marginalization.

.. code-block:: python

   def forward_algorithm(hmm, observations):
       """Run the forward algorithm.

       Parameters
       ----------
       hmm : HMM
       observations : list of length L

       Returns
       -------
       alpha : ndarray of shape (L, K)
           Forward probabilities.
       """
       L = len(observations)
       K = hmm.K
       # alpha[ell, j] will hold the forward probability alpha_j(ell)
       alpha = np.zeros((L, K))

       # Initialization: alpha_j(1) = pi_j * e_j(X_1)
       for j in range(K):
           alpha[0, j] = hmm.initial[j] * hmm.emission(j, observations[0])

       # Recursion: process each position left-to-right
       for ell in range(1, L):
           for j in range(K):
               # hmm.transition[:, j] is the j-th column of A, i.e., A[i,j] for all i.
               # alpha[ell-1, :] is the row of forward probs at the previous position.
               # Multiplying element-wise and summing gives sum_i alpha_i * A_{ij}.
               # np.sum(...) computes this sum over all K previous states.
               alpha[ell, j] = hmm.emission(j, observations[ell]) * \
                   np.sum(alpha[ell - 1, :] * hmm.transition[:, j])

       return alpha

   # Test with a simple observation sequence
   # 0 = No umbrella, 1 = Umbrella
   obs = [0, 0, 1, 1, 0, 1, 1, 1, 0, 0]
   alpha = forward_algorithm(hmm, obs)

   # The total likelihood is the sum over all states at the last position.
   # alpha[-1, :] grabs the last row of the alpha table (i.e., position L).
   likelihood = np.sum(alpha[-1, :])
   print(f"Log-likelihood: {np.log(likelihood):.4f}")

   # Normalizing the last row gives P(state | all observations):
   # this converts joint probabilities to conditional probabilities.
   print(f"Forward probs at last position: {alpha[-1] / alpha[-1].sum()}")


Scaling for Numerical Stability
================================

There is a practical problem with the forward algorithm as written above. Each
forward variable :math:`\alpha_j(\ell)` is a *joint* probability of an
increasingly long sequence of observations. As the sequence grows, this joint
probability shrinks -- it is a product of many numbers less than 1. For a genome
with millions of positions, these numbers become astronomically small, far
smaller than the smallest number a computer can represent (roughly :math:`10^{-308}`
for standard 64-bit floating-point numbers).

When a number becomes too small to represent, the computer rounds it to zero.
This is called **numerical underflow**. Once a forward variable becomes zero, it
stays zero -- all subsequent computations that depend on it produce zero as well.
The entire calculation collapses.

.. admonition:: Why Underflow Happens -- A Concrete Example

   Suppose you have 1,000 positions and at each position the forward variables
   are multiplied by emission probabilities around 0.5. The cumulative product
   is roughly :math:`0.5^{1000} \approx 10^{-301}`. This is barely
   representable. With 10,000 positions, you get :math:`0.5^{10000} \approx
   10^{-3010}`, which is hopelessly beyond the range of floating-point numbers.
   And real genomes have millions of positions.

The solution: **normalize** at each step. Instead of letting the forward
variables shrink to zero, we divide them by their sum at each position, keeping
them in a numerically comfortable range. We record the normalization constants
separately and use them to recover the total likelihood.

The idea is to compute **scaled forward variables** :math:`\hat{\alpha}_j(\ell)`
that sum to 1 at each position, and keep track of the normalizing constants
:math:`c_\ell` separately.

At each step :math:`\ell`, we first compute the unnormalized values:

.. math::

   \tilde{\alpha}_j(\ell) = e_j(X_\ell) \sum_{i=1}^{K} \hat{\alpha}_i(\ell-1) \cdot A_{ij}

Then we normalize by their sum:

.. math::

   c_\ell = \sum_{j=1}^{K} \tilde{\alpha}_j(\ell), \qquad
   \hat{\alpha}_j(\ell) = \frac{\tilde{\alpha}_j(\ell)}{c_\ell}

The **scaled forward variables** :math:`\hat{\alpha}_j(\ell)` represent the
**conditional state distribution**:

.. math::

   \hat{\alpha}_j(\ell) = P(Z_\ell = s_j \mid X_1, \ldots, X_\ell)

Note the difference from the unscaled version: :math:`\alpha_j(\ell)` is a joint
probability (of both the observations and the state), while
:math:`\hat{\alpha}_j(\ell)` is a conditional probability (of the state given the
observations). The scaled version directly answers the question we usually care
about: "given what I have observed so far, how likely is each state?"

**Why does this work?** The unscaled forward variables can be recovered as:

.. math::

   \alpha_j(\ell) = \hat{\alpha}_j(\ell) \prod_{m=1}^{\ell} c_m

To see this, note that at each step we divided by :math:`c_\ell`, so to recover
the original values we multiply back all the :math:`c_m`. The product of all
scaling factors gives the total data likelihood:

.. math::

   P(X_1, \ldots, X_L) = \sum_{j} \alpha_j(L) = \left(\prod_{m=1}^{L} c_m\right) \underbrace{\sum_j \hat{\alpha}_j(L)}_{= 1} = \prod_{m=1}^{L} c_m

Taking logarithms:

.. math::

   \log P(X_1, \ldots, X_L) = \sum_{\ell=1}^{L} \log c_\ell

Each :math:`c_\ell` is a sum of a few numbers (manageable), and the log-likelihood
is a sum of logs (no underflow). This is numerically stable even for sequences of
millions of positions.

.. admonition:: What are the scaling factors, intuitively?

   Each :math:`c_\ell = P(X_\ell \mid X_1, \ldots, X_{\ell-1})` is the
   **predictive probability** of the observation at position :math:`\ell` given all
   previous observations. It answers: "Given everything we have seen so far, how
   surprising is the current observation?" The total likelihood is the product of
   these predictive probabilities -- this is just the chain rule:
   :math:`P(X_1, \ldots, X_L) = P(X_1) \cdot P(X_2 | X_1) \cdot P(X_3 | X_1, X_2) \cdots`

.. code-block:: python

   def forward_scaled(hmm, observations):
       """Forward algorithm with scaling for numerical stability.

       Returns
       -------
       alpha_hat : ndarray of shape (L, K)
           Scaled forward probabilities (conditional state distributions).
           alpha_hat[ell, j] = P(Z_ell = j | X_1, ..., X_ell)
       log_likelihood : float
           log P(X_1, ..., X_L), computed as sum of log(c_ell).
       """
       L = len(observations)
       K = hmm.K
       alpha_hat = np.zeros((L, K))
       log_likelihood = 0.0

       # Initialization: same as before, then normalize
       for j in range(K):
           alpha_hat[0, j] = hmm.initial[j] * hmm.emission(j, observations[0])
       c = alpha_hat[0].sum()        # c_1: normalizing constant for position 1
       alpha_hat[0] /= c             # now alpha_hat[0] sums to 1
       log_likelihood += np.log(c)   # accumulate log-likelihood

       # Recursion: compute, normalize, accumulate
       for ell in range(1, L):
           for j in range(K):
               alpha_hat[ell, j] = hmm.emission(j, observations[ell]) * \
                   np.sum(alpha_hat[ell - 1, :] * hmm.transition[:, j])
           c = alpha_hat[ell].sum()        # c_ell: normalizing constant
           alpha_hat[ell] /= c             # normalize to sum to 1
           log_likelihood += np.log(c)     # accumulate log(c_ell)

       return alpha_hat, log_likelihood

   alpha_hat, ll = forward_scaled(hmm, obs)
   print(f"Log-likelihood (scaled): {ll:.4f}")
   # alpha_hat[-1] is already a proper probability distribution over states:
   print(f"P(state | observations) at last position: {alpha_hat[-1]}")


Stochastic Traceback (Sampling)
================================

The forward algorithm tells us the probability of each hidden state at each
position. But for ARG inference, we need more: we need to produce complete
sequences of hidden states that are consistent with the observations. SINGER
doesn't just find the single most likely state sequence -- it **samples** from
the posterior distribution of state sequences. This is done via **stochastic
traceback** (also called forward-filtering backward-sampling).

The watchmaking analogy: if the forward algorithm is like cataloging every
possible gear configuration that could produce the observed hand positions, then
stochastic traceback is like **rewinding the mechanism**. Starting from the final
position (where we know the state probabilities), we trace backwards through the
sequence, randomly selecting states according to their posterior probability.
Each run of the traceback produces a different plausible gear configuration --
a different possible history that is consistent with the observations.

Why sample rather than just pick the most likely sequence? Because in Bayesian
inference, a single "best" answer can be misleading. By generating many samples,
SINGER captures the *uncertainty* in the genealogical reconstruction. Some parts
of the genome may have a clear signal (all samples agree), while others are
ambiguous (samples disagree). This uncertainty is itself informative.

After running the forward algorithm, we sample states from right to left:

1. Sample :math:`Z_L` from :math:`P(Z_L \mid X_1, \ldots, X_L) \propto \alpha_{Z_L}(L)`

   (At the last position, the scaled forward variables already give us the
   conditional distribution over states.)

2. For :math:`\ell = L-1, \ldots, 1`, sample :math:`Z_\ell` from:

.. math::

   P(Z_\ell = s_i \mid Z_{\ell+1} = s_j, X_1, \ldots, X_\ell)
   \propto \hat{\alpha}_i(\ell) \cdot A_{ij}

This formula has an elegant interpretation: the probability of state :math:`i` at
position :math:`\ell` is proportional to two factors:

- :math:`\hat{\alpha}_i(\ell)`: how likely state :math:`i` is given the
  observations up to position :math:`\ell` (the "forward" information)
- :math:`A_{ij}`: how likely state :math:`i` is to transition to the state
  :math:`j` that we already sampled at position :math:`\ell + 1` (the
  "backward" constraint)

The product of these two factors balances the evidence from the left (what the
data says) with the constraint from the right (what the next state requires).

.. code-block:: python

   def stochastic_traceback(hmm, alpha_hat):
       """Sample a state sequence from the posterior using forward probs.

       This implements "forward-filtering backward-sampling": the forward
       algorithm filters information left-to-right, and this function
       samples a path right-to-left.

       Parameters
       ----------
       hmm : HMM
       alpha_hat : ndarray of shape (L, K)
           Scaled forward probabilities from forward_scaled().

       Returns
       -------
       states : ndarray of shape (L,)
           Sampled state sequence.
       """
       L, K = alpha_hat.shape
       states = np.zeros(L, dtype=int)  # will hold the sampled state at each position

       # Step 1: Sample the last state from the conditional distribution.
       # alpha_hat[-1] is already normalized (sums to 1) from the scaled forward pass.
       probs = alpha_hat[-1] / alpha_hat[-1].sum()
       # np.random.choice(K, p=probs) draws one integer from {0, ..., K-1}
       # with probability given by 'probs'.
       states[-1] = np.random.choice(K, p=probs)

       # Step 2: Traceback from right to left.
       # range(L-2, -1, -1) counts down: L-2, L-3, ..., 1, 0
       for ell in range(L - 2, -1, -1):
           j = states[ell + 1]  # the state we already sampled at position ell+1
           # P(Z_ell = i | Z_{ell+1} = j, data) is proportional to
           # alpha_hat[ell, i] * A[i, j]
           # alpha_hat[ell] is a length-K array of scaled forward probs.
           # hmm.transition[:, j] is column j of A, i.e., A[i, j] for all i.
           # Element-wise multiplication gives the unnormalized probabilities.
           probs = alpha_hat[ell] * hmm.transition[:, j]
           probs /= probs.sum()  # normalize so probabilities sum to 1
           states[ell] = np.random.choice(K, p=probs)

       return states

   np.random.seed(42)
   alpha_hat, _ = forward_scaled(hmm, obs)
   # Sample 5 state sequences from the posterior.
   # Each sample is a different plausible hidden state sequence.
   for i in range(5):
       states = stochastic_traceback(hmm, alpha_hat)
       print(f"Sample {i+1}: {states}")


The Li-Stephens Trick: Linear-Time Transitions
================================================

We have now seen the full HMM inference pipeline: the forward algorithm computes
state probabilities, scaling prevents underflow, and stochastic traceback
generates samples. There is one remaining problem: **speed**.

A standard HMM forward step is :math:`O(K^2)` because every state can transition
to every other state -- the sum :math:`\sum_i \alpha_i \cdot A_{ij}` has
:math:`K` terms and must be computed for each of :math:`K` states. For SINGER,
:math:`K` is the number of branches in the marginal tree (:math:`2n - 1` for
:math:`n` samples), so this is expensive. With :math:`n = 1000` samples, we
would have :math:`K \approx 2000` states and :math:`K^2 = 4{,}000{,}000`
operations per genomic position. Multiply by millions of positions and the
computation becomes intractable.

The **Li-Stephens model** (Li and Stephens, 2003) exploits a special structure in
the transition matrix that reduces the per-position cost from :math:`O(K^2)` to
:math:`O(K)`. This is the single most important computational trick in HMM-based
ARG inference.

The Li-Stephens Transition Structure
--------------------------------------

The transition probability has the form:

.. math::

   A_{ij} = (1 - r_i)\delta_{ij} + r_i \cdot \frac{q_j}{\sum_k q_k}

Let us unpack every symbol:

.. admonition:: What Is the Kronecker Delta?

   The symbol :math:`\delta_{ij}` is the **Kronecker delta**. It is the simplest
   possible function of two indices:

   .. math::

      \delta_{ij} = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}

   It acts as a mathematical "filter": in any sum involving :math:`\delta_{ij}`,
   only the term where :math:`i = j` survives. For example:

   .. math::

      \sum_{i=1}^{K} f(i) \cdot \delta_{ij} = f(j)

   All terms where :math:`i \neq j` are multiplied by zero and vanish. This
   property is what makes the Li-Stephens trick work.

The transition formula says: with probability :math:`1 - r_i`, **stay in the same
state** (no recombination). With probability :math:`r_i`, **"recombine"** and
jump to any state :math:`j` with probability proportional to :math:`q_j`.

In the genetics context:

- :math:`r_i` is the probability that a recombination event occurs while the
  lineage sits on branch :math:`i`. This depends on the branch length in
  coalescent time units (see :ref:`coalescent_theory`) and the recombination rate.
- :math:`q_j` is the probability that, after recombination, the lineage re-joins
  branch :math:`j`. This is proportional to the coalescence probability on
  branch :math:`j`.
- :math:`\delta_{ij}` ensures that "staying" means remaining on the *same*
  branch.

The key structural property is that the "jump" part of the transition is
**separable**: the probability of jumping *to* state :math:`j` (which is
:math:`q_j / \sum_k q_k`) does not depend on which state :math:`i` we are
jumping *from*. This separability is what enables the :math:`O(K)` trick.

**A concrete small example.** Consider :math:`K = 3` states with :math:`r_i = 0.1`
for all :math:`i` and :math:`q = (0.5, 0.3, 0.2)`. Then :math:`\sum_k q_k = 1.0`
and the transition matrix is:

.. math::

   A = \begin{pmatrix}
   0.9 + 0.1 \cdot 0.5 & 0.1 \cdot 0.3 & 0.1 \cdot 0.2 \\
   0.1 \cdot 0.5 & 0.9 + 0.1 \cdot 0.3 & 0.1 \cdot 0.2 \\
   0.1 \cdot 0.5 & 0.1 \cdot 0.3 & 0.9 + 0.1 \cdot 0.2
   \end{pmatrix}
   = \begin{pmatrix}
   0.95 & 0.03 & 0.02 \\
   0.05 & 0.93 & 0.02 \\
   0.05 & 0.03 & 0.92
   \end{pmatrix}

Notice the structure: each row has a large diagonal entry (staying) and small
off-diagonal entries (jumping), and the off-diagonal entries in each column are
identical (because :math:`q_j / \sum_k q_k` does not depend on the row).

**Proof that rows sum to 1.** For any row :math:`i`:

.. math::

   \sum_j A_{ij} &= \sum_j \left[(1 - r_i)\delta_{ij} + r_i \frac{q_j}{\sum_k q_k}\right] \\
   &= (1 - r_i) \underbrace{\sum_j \delta_{ij}}_{= 1} + r_i \underbrace{\frac{\sum_j q_j}{\sum_k q_k}}_{= 1} \\
   &= (1 - r_i) + r_i = 1 \quad \checkmark


The :math:`O(K)` Forward Step
-------------------------------

Now we substitute the Li-Stephens transition structure into the forward recursion
and see how the sum simplifies. This is the algebraic heart of the trick, so we
will go step by step.

Start with the standard forward recursion:

.. math::

   \alpha_j(\ell) = e_j(X_\ell) \sum_{i=1}^{K} \alpha_i(\ell-1) \cdot A_{ij}

Substitute the Li-Stephens form of :math:`A_{ij}`:

.. math::

   \alpha_j(\ell) = e_j(X_\ell) \sum_i \alpha_i(\ell-1) \left[(1 - r_i)\delta_{ij} + r_i \frac{q_j}{\sum_k q_k}\right]

Now distribute the sum over the two terms inside the brackets:

.. math::

   \alpha_j(\ell) = e_j(X_\ell) \left[\sum_i \alpha_i(\ell-1)(1 - r_i)\delta_{ij} + \sum_i \alpha_i(\ell-1) r_i \frac{q_j}{\sum_k q_k}\right]

**Simplify the first sum.** The Kronecker delta :math:`\delta_{ij}` kills every
term except :math:`i = j`:

.. math::

   \sum_i \alpha_i(\ell-1)(1 - r_i)\delta_{ij} = \alpha_j(\ell-1)(1 - r_j)

Only the "stay in state :math:`j`" term survives.

**Simplify the second sum.** The factor :math:`q_j / \sum_k q_k` does not depend
on :math:`i`, so it can be pulled out of the sum:

.. math::

   \sum_i \alpha_i(\ell-1) r_i \frac{q_j}{\sum_k q_k} = \frac{q_j}{\sum_k q_k} \sum_i r_i \alpha_i(\ell-1)

This is the crucial step: the sum :math:`\sum_i r_i \alpha_i(\ell-1)` does not
depend on :math:`j` at all. It is the same number regardless of which target
state we are computing.

Putting it together:

.. math::

   \alpha_j(\ell) = e_j(X_\ell) \left[\alpha_j(\ell-1)(1 - r_j) + \frac{q_j}{\sum_k q_k} \underbrace{\sum_i r_i \alpha_i(\ell-1)}_{R}\right]

The key: the sum :math:`R = \sum_i r_i \alpha_i(\ell-1)` is computed **once**
in :math:`O(K)` time and reused for all :math:`j`. Each :math:`\alpha_j` then
requires only :math:`O(1)` work (one multiplication for the "stay" term, one
for the "jump" term, one addition, and one multiplication by the emission), giving
:math:`O(K)` total per position instead of :math:`O(K^2)`.

.. admonition:: Why the Sum Factorizes -- The Key Insight

   The reason this trick works is that the Li-Stephens transition separates into
   two parts: a "stay" part that depends on both source and target, and a "jump"
   part where the source and target contributions *multiply independently*.

   In the "jump" component, :math:`r_i` (from the source state) and
   :math:`q_j / \sum_k q_k` (from the target state) do not interact -- they
   are multiplied, not entangled. This means the double sum
   :math:`\sum_i \sum_j` can be split into :math:`(\sum_i \cdots)(\sum_j \cdots)`
   or equivalently, the inner sum :math:`\sum_i r_i \alpha_i` can be precomputed
   once.

   If the transition probability had a more complicated dependence on *both*
   :math:`i` and :math:`j`, this factorization would not be possible, and we
   would be stuck with :math:`O(K^2)`.

**A small worked example.** Suppose :math:`K = 3`, :math:`r = (0.1, 0.1, 0.1)`,
:math:`q = (0.5, 0.3, 0.2)`, and the forward variables at the previous position
are :math:`\alpha(\ell-1) = (0.4, 0.35, 0.25)`. Emissions at the current
position are :math:`e = (0.8, 0.6, 0.9)`.

Step 1: Compute the recombination sum once.

.. math::

   R = \sum_i r_i \alpha_i(\ell-1) = 0.1 \times 0.4 + 0.1 \times 0.35 + 0.1 \times 0.25 = 0.1

Step 2: Compute each :math:`\alpha_j(\ell)`.

.. math::

   \alpha_1(\ell) &= 0.8 \times [0.4 \times 0.9 + 0.5 \times 0.1] = 0.8 \times [0.36 + 0.05] = 0.328 \\
   \alpha_2(\ell) &= 0.6 \times [0.35 \times 0.9 + 0.3 \times 0.1] = 0.6 \times [0.315 + 0.03] = 0.207 \\
   \alpha_3(\ell) &= 0.9 \times [0.25 \times 0.9 + 0.2 \times 0.1] = 0.9 \times [0.225 + 0.02] = 0.2205

Each :math:`\alpha_j` used only its own previous value (for the "stay" part) and
the shared :math:`R` (for the "jump" part). No double sum over states was needed.

.. code-block:: python

   def forward_li_stephens(initial, r, q, emissions, observations):
       """Forward algorithm with Li-Stephens transition structure.

       The Li-Stephens transition matrix has the form:
           A[i,j] = (1 - r[i]) * delta(i,j) + r[i] * q[j] / sum(q)
       This enables an O(K) forward step instead of O(K^2).

       Parameters
       ----------
       initial : ndarray of shape (K,)
           Initial state distribution.
       r : ndarray of shape (K,)
           Per-state recombination probabilities. r[i] is the probability
           of a recombination event on branch i.
       q : ndarray of shape (K,)
           Re-joining weights. q[j] / sum(q) is the probability of
           re-joining branch j after recombination.
       emissions : ndarray of shape (L, K)
           Pre-computed emission probabilities: emissions[ell, j] = P(X_ell | Z_ell = j).
       observations : ignored (emissions pre-computed)

       Returns
       -------
       alpha : ndarray of shape (L, K)
           Forward probabilities at each position.
       """
       L, K = emissions.shape
       alpha = np.zeros((L, K))

       # Initialization: alpha_j(1) = pi_j * e_j(X_1)
       alpha[0] = initial * emissions[0]

       # Pre-compute the sum of q for normalization (constant across positions)
       q_sum = q.sum()

       for ell in range(1, L):
           # O(K) step: compute the recombination sum R = sum_i r[i] * alpha[i]
           # np.sum(r * alpha[ell-1]) multiplies r and alpha element-wise,
           # then sums all K elements. This is done ONCE per position.
           recomb_sum = np.sum(r * alpha[ell - 1])

           for j in range(K):
               # "Stay" term: probability of no recombination on branch j,
               # times the previous forward probability on the same branch j.
               stay = (1 - r[j]) * alpha[ell - 1, j]

               # "Jump" term: probability of recombining and re-joining branch j.
               # Uses the pre-computed recomb_sum (shared across all j).
               switch = (q[j] / q_sum) * recomb_sum

               # Multiply by emission probability for state j at position ell.
               alpha[ell, j] = emissions[ell, j] * (stay + switch)

       return alpha

   # Test: 10 states, 100 positions
   K, L = 10, 100
   r = np.full(K, 0.05)                          # uniform recombination rate
   q = np.random.dirichlet(np.ones(K))            # random re-joining weights
   emissions = np.random.uniform(0.1, 0.9, size=(L, K))  # random emissions

   alpha = forward_li_stephens(
       initial=np.ones(K) / K,    # uniform initial distribution
       r=r, q=q, emissions=emissions, observations=None
   )
   print(f"Forward probs shape: {alpha.shape}")
   print(f"Sum at last position: {alpha[-1].sum():.6e}")

.. admonition:: Why this matters for SINGER

   In SINGER's branch sampling HMM, the hidden states are branches in the
   marginal tree. The transition probability has exactly this Li-Stephens
   structure: with probability :math:`1 - r_i`, stay on the same branch (no
   recombination); with probability :math:`r_i`, recombine and re-join a
   branch with probability proportional to the branch's coalescence probability.
   The coalescence probabilities play the role of the :math:`q_j` weights.
   Branch lengths are measured in coalescent time units (see
   :ref:`coalescent_theory`), and the recombination probabilities :math:`r_i`
   depend on both the recombination rate and the branch length.

   Without the Li-Stephens trick, SINGER's forward algorithm would be
   :math:`O(K^2)` per position -- infeasible for large sample sizes. With it,
   the cost is :math:`O(K)`, making genome-scale inference practical.


Summary
=======

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Algorithm
     - What it gives you
   * - Forward algorithm
     - :math:`\alpha_j(\ell) = P(X_1, \ldots, X_\ell, Z_\ell = j)`
   * - Scaled forward
     - Numerically stable version + log-likelihood
   * - Stochastic traceback
     - Posterior samples of state sequences
   * - Li-Stephens trick
     - :math:`O(K)` transitions instead of :math:`O(K^2)`

These are the **gear train** of SINGER. The forward algorithm is the mechanism
that combines observations and transitions into state probabilities, position by
position. Scaling keeps the gears turning without jamming (numerical underflow).
Stochastic traceback runs the mechanism in reverse, producing complete hidden
state sequences. And the Li-Stephens structure is the elegant engineering that
makes it all fast enough to work on real genomes.

Together, the forward algorithm and stochastic traceback form the core of how
SINGER samples branches and coalescence times along the genome. The Li-Stephens
transition structure is what makes this sampling computationally feasible for
large datasets.

Next: :ref:`smc` -- the Markov approximation that makes ARG inference possible.
