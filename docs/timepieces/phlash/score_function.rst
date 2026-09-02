.. _phlash_score_function:

====================================
The Linear-Memory Score Construction
====================================

The score is the gradient of a log likelihood, not merely the posterior state
probabilities returned by an ordinary forward--backward pass.  Fisher's identity
expresses an HMM score as a conditional expectation of complete-data derivative
terms.  For a transition parameter :math:`\phi`, define

.. math::

   \psi(i,j)=\frac{\partial\log T_{ij}}{\partial\phi}.

Then the transition contribution is the posterior expectation of
:math:`\sum_\ell\psi(X_{\ell-1},X_\ell)` given the observations.

PHLASH uses a linear-memory Baum--Welch recursion to carry that additive
expectation forward.  A generic version for one feature is implemented as
``fisher_transition_score`` and checked against enumeration of all latent paths.
It needs no array whose length is the chromosome length.

Why PHLASH is faster than the generic recursion
===============================================

A generic transition matrix costs :math:`O(M^2)` per forward step.  Coalescent
HMM transition matrices have structured subdiagonal, diagonal, and upper-triangle
terms.  PHLASH parameterizes those terms as ``b``, ``d``, ``u`` and ``v``;
``phlash.hmm.matvec_smc`` evaluates :math:`x^\mathsf{T}T` in :math:`O(M)` time.
The mini ``structured_smc_matvec`` is a direct NumPy transcription and is tested
against explicit dense multiplication.

Combining the linear-memory recursion with that transition structure yields the
paper's full score complexity of :math:`O(LM^2)` time and :math:`O(M^2)` storage,
independent of sequence length :math:`L`.  The supplement also describes one
feature recursion as :math:`O(M)` memory; all :math:`O(M)` structured coordinates
together require :math:`O(M^2)`.  This corrects the former chapter's contradictory
claim of constant memory and its substitution of a full forward--backward table.

Parallel chunks
===============

The software divides long data into chunks.  Each selected chunk has a warm-up
prefix (default overlap 500 bins) used to approach the filtering distribution;
only the following portion contributes to the likelihood.  A random minibatch of
chunks supplies a stochastic sequence score, scaled by :math:`N/S`.  The
independence between chunks is an approximation based on HMM forgetting, not an
exact factorization.
