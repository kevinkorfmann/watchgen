.. _smcpp_hmm:

===========================
The Continuous-Time HMM
===========================

   *Pairwise transitions, conditioned-frequency-spectrum emissions.*

Hidden states and initial probabilities
=======================================

Let the hidden intervals be

.. math::

   I_m=[t_m,t_{m+1}),\qquad t_0=0,\quad t_M=\infty.

The state :math:`Z_\ell=m` means that the TMRCA of the distinguished pair at
locus :math:`\ell` lies in :math:`I_m`. For constant relative size
:math:`\lambda`, the marginal state probabilities are

.. math::

   \pi_m=e^{-t_m/\lambda}-e^{-t_{m+1}/\lambda}.

.. literalinclude:: ../../../watchgen/mini_smcpp.py
   :language: python
   :pyobject: pair_interval_probabilities

The transition model
====================

SMC++ does not use the original SMC transition density presented for PSMC.
It uses the continuous-time conditioned Markov chain of Hobolth and Jensen,
derived from the exact two-locus coalescent. This model is slightly more
accurate than the usual SMC' construction because it integrates over any
number of recombinations and back-coalescences between adjacent loci.

The core process has three states. While both loci remain linked, a
recombination at rate :math:`\rho` creates a floating lineage. That lineage
back-coalesces at rate :math:`\alpha(t)` or the second marginal genealogy
coalesces at the same rate. The latter event is absorbing. On an interval with
constant :math:`\alpha`, the generator is

.. math::

   G=
   \begin{pmatrix}
   -\rho&\rho&0\\
   \alpha&-2\alpha&\alpha\\
   0&0&0
   \end{pmatrix},

and the exact kernel across a distance :math:`\Delta` is
:math:`e^{\Delta G}`.

.. literalinclude:: ../../../watchgen/mini_smcpp.py
   :language: python
   :pyobject: two_locus_kernel

The original ``src/transition.cpp`` evaluates this exponential analytically,
composes it across demographic intervals, and integrates the result into
hidden TMRCA intervals. Like the source, our constant-demography specialization
represents each source interval by its conditional mean coalescence time.

.. literalinclude:: ../../../watchgen/mini_smcpp.py
   :language: python
   :pyobject: constant_csc_transition_matrix

The function has no sample-count argument. That omission is deliberate: the
extra samples affect the emissions, while transitions describe the
distinguished pair at two neighboring loci.

Emission tables
===============

For state :math:`m`, the emission table is obtained in two steps. First average
the conditioned branch lengths over :math:`T\in I_m`. Then apply the mutation
transform to obtain probabilities for every :math:`(a,b)` observation.

.. literalinclude:: ../../../watchgen/mini_smcpp.py
   :language: python
   :pyobject: emission_probabilities

This table includes the monomorphic ancestral observation :math:`(0,0)`. The
fixed-derived category :math:`(2,n)` receives no finite branch length because
the ancestral state above the sample MRCA is not part of the genealogy.

Forward likelihood
==================

With transition matrix :math:`P`, emission table :math:`E`, and initial
distribution :math:`\pi`, the forward recursion is

.. math::

   \alpha_1(m)=\pi_mE_m(X_1),

.. math::

   \alpha_{\ell+1}(m)
   =E_m(X_{\ell+1})\sum_j\alpha_\ell(j)P_{jm}.

Scaling after every site avoids numerical underflow; the log of each scale
factor contributes to the log likelihood.

.. literalinclude:: ../../../watchgen/mini_smcpp.py
   :language: python
   :pyobject: forward_log_likelihood

Locus skipping
==============

Whole genomes contain long runs of identical monomorphic observations. If
:math:`B` is the diagonal emission matrix for that symbol and :math:`W=PB`, a
run of :math:`d` sites can be traversed with :math:`W^d` instead of :math:`d`
individual forward steps. Production SMC++ obtains powers from an
eigendecomposition and also accumulates the posterior transition and emission
counts needed by EM. This changes computational cost from depending directly
on sequence length to depending mainly on polymorphic sites and hidden-state
dimension.

Thinning and model misspecification
===================================

Conditioning on the pair TMRCA does not make adjacent full-sample genealogies
independent. Branch lengths in the undistinguished part remain correlated, so
emitting the complete CSFS at every site violates the HMM's conditional
independence assumption. This is a failure of conditional independence, not a
minor numerical approximation. The original paper therefore uses **thinning**: the
full CSFS is emitted only periodically, while intervening observations retain
the distinguished-pair component. The paper used a heuristic proportional to
sample size; current software exposes a configurable ``--thinning`` option and
documents a different default heuristic. Neither value is a universal law.

Optimization and regularization
===============================

Production SMC++ fits a parameterized population-size curve with EM and
automatic differentiation. It regularizes the curve to control oscillation
and supports piecewise or spline representations. The paper's spline model and
the current command-line default are not identical: current documentation says
that recent releases default to a piecewise representation, whereas cubic
splines were used in the paper. Any parity comparison must therefore record
the software version and command-line options.

Composite likelihood
====================

For data sets :math:`D_1,\ldots,D_r` made with different distinguished pairs,
SMC++ sums their HMM log likelihoods:

.. math::

   \ell_C(\eta)=\sum_{i=1}^r\log P(D_i\mid\eta).

.. literalinclude:: ../../../watchgen/mini_smcpp.py
   :language: python
   :pyobject: composite_log_likelihood

When those data sets reuse samples or genomic regions, the terms are dependent;
the result remains useful for estimation but is not an ordinary independent-
replicate likelihood, and naive likelihood-based uncertainty calculations do
not apply.
