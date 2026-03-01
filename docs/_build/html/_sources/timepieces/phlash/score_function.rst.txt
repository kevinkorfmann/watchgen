.. _phlash_score_function:

================================
The Score Function Algorithm
================================

   *The gear train: transmitting energy from the mainspring to the hands with minimal loss.*

In a mechanical watch, the **gear train** connects the mainspring (which
stores energy) to the hands (which display the time). Every tooth must mesh
precisely: too much friction and the watch runs slow; too loose and the hands
jump. The gear train does not generate energy or display time -- it
*transmits* one to the other with maximum efficiency.

PHLASH's score function algorithm plays exactly this role. The composite
likelihood (the mainspring) stores the information in the data. SVGD (the
hands) needs gradients to update the demographic history. The score function
algorithm transmits the likelihood's information to SVGD by computing the
gradient :math:`\nabla_\eta \ell(\eta)` in :math:`O(LM^2)` time -- 30 to 90
times faster than the naive approach of running reverse-mode automatic
differentiation (autodiff) through the forward algorithm.


.. admonition:: Biology Aside -- Why efficient computation matters for population genomics

   Whole-genome sequencing datasets now routinely contain thousands of
   individuals, each with billions of base pairs. The coalescent HMM must
   process this data to extract information about population size history.
   With naive gradient computation, fitting a model to even a handful of
   genomes could take days. The score function algorithm makes this practical
   by computing gradients at essentially the same cost as evaluating the
   likelihood itself. This is what enables PHLASH to jointly use data from
   multiple individuals -- extracting more information about demographic
   history than single-genome methods like PSMC.

Why Gradients Matter
=====================

SVGD updates each particle (candidate demographic history) using the gradient
of the log-posterior:

.. math::

   \nabla_{\boldsymbol{h}} \log p(\boldsymbol{h} \mid \text{data})
   = \nabla_{\boldsymbol{h}} \ell_{\text{comp}}(\boldsymbol{h})
   + \nabla_{\boldsymbol{h}} \log p(\boldsymbol{h})

The prior gradient is cheap (it is a linear function of
:math:`\boldsymbol{h}` for the Gaussian smoothness prior). The expensive part
is the **likelihood gradient** :math:`\nabla_{\boldsymbol{h}}
\ell_{\text{comp}}`. And the dominant cost within the likelihood gradient is
the coalescent HMM component, because the forward algorithm processes
:math:`L` genomic positions with an :math:`M \times M` transition matrix.


The Cost of Naive Autodiff
===========================

The forward algorithm for a single HMM with :math:`L` observations and
:math:`M` hidden states costs :math:`O(LM^2)` in the forward pass (matrix-
vector multiplication at each position). Reverse-mode automatic
differentiation (backpropagation) through this computation has the same
asymptotic complexity, :math:`O(LM^2)`, but with a much larger constant
factor:

- **Memory**: autodiff must store all :math:`L` intermediate forward vectors
  (or recompute them via checkpointing), requiring :math:`O(LM)` memory.

- **Overhead**: each operation in the forward algorithm (multiply, add, log,
  exp) is wrapped in a tape-recording layer that tracks the computation graph.
  This overhead factor is typically 5--20x.

- **Vectorization**: autodiff engines like JAX can JIT-compile the backward
  pass, but the backward pass through a sequential scan (the forward algorithm
  is inherently sequential along the genome) is difficult to parallelize.

In practice, computing the gradient via autodiff takes 30--90x longer than
computing the likelihood alone. For SVGD, which needs gradients at every
iteration for every particle, this overhead is the bottleneck.


The Score Function Idea
========================

The **score function** of a statistical model is the gradient of the
log-likelihood with respect to the parameters:

.. math::

   s(\eta) = \nabla_\eta \log p(\text{data} \mid \eta)

PHLASH exploits the structure of the HMM to compute this gradient
*analytically* within the forward-backward framework, rather than relying on
generic autodiff. The key insight is that the log-likelihood of an HMM can
be differentiated in closed form using the **posterior marginals** -- the
same quantities that the forward-backward algorithm already computes.


Deriving the Score for the Coalescent HMM
==========================================

The coalescent HMM log-likelihood for a single pair is:

.. math::

   \ell(\eta) = \log p(\mathbf{x} \mid \eta)
   = \log \sum_{\mathbf{z}} p(\mathbf{x}, \mathbf{z} \mid \eta)

where :math:`\mathbf{x} = (x_1, \ldots, x_L)` is the observation sequence
and :math:`\mathbf{z} = (z_1, \ldots, z_L)` is the hidden state sequence.

The gradient with respect to the parameters :math:`\eta` can be written using
the **Fisher identity** (also known as the score function identity):

.. math::

   \nabla_\eta \ell(\eta)
   = \sum_{\mathbf{z}} p(\mathbf{z} \mid \mathbf{x}, \eta) \,
   \nabla_\eta \log p(\mathbf{x}, \mathbf{z} \mid \eta)

This says: the gradient of the log-likelihood equals the **posterior
expectation** of the gradient of the complete-data log-likelihood. The
complete-data log-likelihood decomposes into a sum over positions:

.. math::

   \log p(\mathbf{x}, \mathbf{z} \mid \eta)
   = \log a_0(z_1)
   + \sum_{\ell=2}^{L} \log p_{z_{\ell-1}, z_\ell}
   + \sum_{\ell=1}^{L} \log e_{z_\ell}(x_\ell)

Taking the gradient and the posterior expectation gives:

.. math::

   \nabla_\eta \ell(\eta)
   = \sum_k \gamma_1(k) \, \nabla_\eta \log a_0(k)
   + \sum_{\ell=2}^{L} \sum_{k,l} \xi_\ell(k,l) \,
   \nabla_\eta \log p_{kl}
   + \sum_{\ell=1}^{L} \sum_k \gamma_\ell(k) \,
   \nabla_\eta \log e_k(x_\ell)

where:

- :math:`\gamma_\ell(k) = p(z_\ell = k \mid \mathbf{x}, \eta)` is the
  **posterior marginal** at position :math:`\ell` -- the probability of being
  in state :math:`k` at position :math:`\ell`, given the data.

- :math:`\xi_\ell(k,l) = p(z_{\ell-1} = k, z_\ell = l \mid \mathbf{x}, \eta)`
  is the **posterior pairwise marginal** -- the probability of transitioning
  from state :math:`k` to state :math:`l` between positions :math:`\ell-1`
  and :math:`\ell`.

.. admonition:: Plain-language summary -- What :math:`\gamma` and :math:`\xi` tell us

   :math:`\gamma_\ell(k)` answers the question: *at genomic position*
   :math:`\ell`, *what is the probability that the TMRCA falls in time
   interval* :math:`k`? This is the same posterior decoding used in PSMC.
   :math:`\xi_\ell(k, l)` goes further: it tells us the probability that the
   TMRCA changed from interval :math:`k` to interval :math:`l` between two
   adjacent positions -- i.e., the probability that recombination shifted the
   genealogy. Together, these two quantities summarize everything the data
   says about the hidden genealogy at each position. The score function
   algorithm shows that these are the only quantities needed to compute the
   gradient -- no additional backward passes through the computation graph
   are required.

Both :math:`\gamma` and :math:`\xi` are computed by the standard forward-
backward algorithm, which PHLASH already runs to evaluate the likelihood.

.. code-block:: python

   import numpy as np

   def hmm_score_function(observations, transition, emission, initial):
       """Compute the HMM log-likelihood gradient via the Fisher identity.

       This implements the score function algorithm: run forward-backward
       to get posterior marginals, then use them to weight the parameter
       derivatives of the complete-data log-likelihood.

       Parameters
       ----------
       observations : ndarray, shape (L,)
           Integer observation sequence.
       transition : ndarray, shape (M, M)
           Transition probability matrix p_{kl}.
       emission : ndarray, shape (M, n_obs)
           Emission probabilities e_k(x).
       initial : ndarray, shape (M,)
           Initial state distribution.

       Returns
       -------
       log_likelihood : float
           The log-likelihood of the observations.
       gamma : ndarray, shape (L, M)
           Posterior state marginals at each position.
       xi_sum : ndarray, shape (M, M)
           Summed posterior pairwise marginals (transition counts).
       """
       L = len(observations)
       M = len(initial)

       # Forward pass (scaled)
       alpha = np.zeros((L, M))
       scale = np.zeros(L)
       alpha[0] = initial * emission[:, observations[0]]
       scale[0] = alpha[0].sum()
       alpha[0] /= scale[0]

       for t in range(1, L):
           alpha[t] = (alpha[t-1] @ transition) * emission[:, observations[t]]
           scale[t] = alpha[t].sum()
           alpha[t] /= scale[t]

       log_likelihood = np.sum(np.log(scale))

       # Backward pass (scaled)
       beta = np.zeros((L, M))
       beta[-1] = 1.0
       for t in range(L-2, -1, -1):
           beta[t] = transition @ (emission[:, observations[t+1]] * beta[t+1])
           beta[t] /= scale[t+1]

       # Posterior marginals gamma_t(k) = alpha_t(k) * beta_t(k)
       gamma = alpha * beta
       gamma /= gamma.sum(axis=1, keepdims=True)

       # Summed pairwise marginals: xi_sum(k, l) = sum_t xi_t(k, l)
       xi_sum = np.zeros((M, M))
       for t in range(L-1):
           xi_t = (alpha[t, :, None] * transition
                   * emission[None, :, observations[t+1]] * beta[t+1, None, :])
           xi_t /= xi_t.sum()
           xi_sum += xi_t

       return log_likelihood, gamma, xi_sum

   # Demonstrate with a small HMM (2 states, binary observations)
   M = 2
   transition = np.array([[0.99, 0.01],
                           [0.02, 0.98]])
   emission = np.array([[0.999, 0.001],   # state 0: mostly hom
                         [0.95,  0.05]])   # state 1: some hets
   initial = np.array([0.5, 0.5])

   # Synthetic observation sequence
   np.random.seed(42)
   obs = np.zeros(200, dtype=int)
   obs[50] = obs[120] = obs[180] = 1  # three het sites

   ll, gamma, xi_sum = hmm_score_function(obs, transition, emission, initial)
   print(f"Log-likelihood: {ll:.2f}")
   print(f"Posterior at het site (pos 50): "
         f"state 0 = {gamma[50,0]:.3f}, state 1 = {gamma[50,1]:.3f}")
   print(f"Transition counts (sum of xi):")
   print(f"  0->0: {xi_sum[0,0]:.1f}, 0->1: {xi_sum[0,1]:.2f}")
   print(f"  1->0: {xi_sum[1,0]:.2f}, 1->1: {xi_sum[1,1]:.1f}")


Complexity
===========

The score function algorithm has three stages:

1. **Forward pass** (:math:`O(LM^2)`): compute forward probabilities and the
   log-likelihood.

2. **Backward pass** (:math:`O(LM^2)`): compute backward probabilities and
   the posterior marginals :math:`\gamma_\ell(k)` and :math:`\xi_\ell(k,l)`.

3. **Gradient accumulation** (:math:`O(LM^2 + M^2 R)`): multiply the
   posterior statistics by the parameter derivatives of the transition matrix
   and emission probabilities, and sum over positions. Here :math:`R` is the
   number of parameters (the :math:`M` population size values
   :math:`\eta_k`).

The total cost is :math:`O(LM^2)` -- the same asymptotic complexity as the
forward algorithm itself. The constant factor overhead is small: essentially
one forward pass plus one backward pass plus a matrix-vector product for each
parameter. In practice, this is **30--90x faster** than reverse-mode autodiff
through the forward algorithm, because:

- No computation graph is recorded.
- No memory is needed beyond the forward and backward vectors.
- The backward pass is just another matrix-vector scan (the same shape as the
  forward pass), not a generic reverse-mode traversal.

.. admonition:: Comparison to EM

   If you have read :ref:`psmc_hmm`, you may notice that the posterior
   statistics :math:`\gamma` and :math:`\xi` are exactly the quantities
   computed in the E-step of the EM algorithm. In PSMC, these statistics are
   used to form a Q-function that is maximized in closed form (the M-step).
   In PHLASH, the same statistics are used to compute the gradient directly,
   which is then fed to SVGD. The E-step is shared; the difference is what
   happens next. EM maximizes; SVGD explores.


The SFS Gradient
=================

The gradient of the SFS log-likelihood is simpler. Each expected SFS entry
:math:`\xi_k(\eta)` is a known function of the demographic parameters
(involving expected coalescent branch lengths, which have closed-form
derivatives for piecewise-constant :math:`\eta`). The SFS gradient is:

.. math::

   \nabla_\eta \ell_{\text{SFS}}(\eta)
   = \sum_{k=1}^{n-1} \left(
   \frac{D_k}{\xi_k(\eta)} - 1
   \right) \nabla_\eta \xi_k(\eta)

The cost is :math:`O(n \cdot M)` where :math:`n` is the sample size (number
of haploid chromosomes) and :math:`M` is the number of time intervals. This
is negligible compared to the HMM gradient.


Total Gradient
===============

The complete gradient of the composite log-posterior is:

.. math::

   \nabla_{\boldsymbol{h}} \log p(\boldsymbol{h} \mid \text{data})
   = \nabla_{\boldsymbol{h}} \ell_{\text{SFS}}
   + \sum_{p=1}^{P} \nabla_{\boldsymbol{h}} \ell_{\text{HMM}}^{(p)}
   + \nabla_{\boldsymbol{h}} \log p(\boldsymbol{h})

Each HMM gradient is computed independently (the pairs are independent given
:math:`\eta`), making the computation embarrassingly parallel across pairs.
On a GPU, all pairs can be processed simultaneously.

.. code-block:: python

   def total_gradient(h, observed_sfs, expected_sfs, hmm_scores,
                      sigma_prior=1.0):
       """Compute the total gradient of the log-posterior.

       Combines the SFS gradient, summed HMM score functions, and the
       smoothness prior gradient.

       Parameters
       ----------
       h : ndarray, shape (M,)
           Log population sizes.
       observed_sfs : ndarray
           Observed SFS.
       expected_sfs : ndarray
           Expected SFS under current h.
       hmm_scores : list of ndarray, each shape (M,)
           Score function (gradient) from each pairwise HMM.
       sigma_prior : float
           Smoothness prior scale.

       Returns
       -------
       grad : ndarray, shape (M,)
           Gradient of the composite log-posterior.
       """
       M = len(h)

       # SFS gradient: sum_k (D_k / xi_k - 1) * d(xi_k)/d(h)
       # Simplified: for constant-size model, d(xi_k)/d(h) ~ xi_k
       xi = np.maximum(expected_sfs, 1e-300)
       grad_sfs_weights = observed_sfs / xi - 1  # per-frequency weights
       # In practice, d(xi_k)/d(h_j) depends on the branch length derivatives
       grad_sfs = np.zeros(M)  # placeholder for full implementation

       # HMM gradient: sum over pairs
       grad_hmm = np.sum(hmm_scores, axis=0)

       # Prior gradient: d/dh [-0.5 * sum((h_j - h_{j-1})^2) / sigma^2]
       grad_prior = np.zeros(M)
       for j in range(1, M):
           grad_prior[j] += (h[j-1] - h[j]) / sigma_prior**2
           grad_prior[j-1] += (h[j] - h[j-1]) / sigma_prior**2

       return grad_sfs + grad_hmm + grad_prior

   # Demonstrate
   M = 32
   h = np.zeros(M)
   h[10:20] = -0.5  # mild bottleneck
   # Simulate HMM score functions from 5 pairs
   rng = np.random.default_rng(42)
   hmm_scores = [rng.normal(0, 0.1, size=M) for _ in range(5)]
   xi_expected = expected_sfs_constant(20, 100.0)
   grad = total_gradient(h, D_observed, xi_expected, hmm_scores)
   print(f"Gradient norm: {np.linalg.norm(grad):.4f}")
   print(f"Gradient at bottleneck (interval 15): {grad[15]:.4f}")
   print(f"Gradient at constant (interval 5):    {grad[5]:.4f}")


What Comes Next
================

With the gradient in hand, we have everything needed for posterior sampling.
The :ref:`next chapter <phlash_svgd>` introduces Stein Variational Gradient
Descent -- the algorithm that uses these gradients to push a set of particles
toward the posterior distribution over demographic histories. SVGD is the
winding mechanism: it converts gradient energy into the organized motion of
particles converging on the posterior.
