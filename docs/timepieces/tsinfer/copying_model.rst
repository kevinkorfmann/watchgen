The copying model
=================

Dense reference recurrence
--------------------------

For teaching, let :math:`Z_j\in\{1,\ldots,K\}` identify the reference copied at
site :math:`j`. If :math:`r_j` is the recombination probability between sites,
the Li--Stephens-like transition is

.. math::

   P(Z_j=k\mid Z_{j-1}=h)=
   \begin{cases}
   1-r_j+r_j/K, & k=h,\\
   r_j/K, & k\ne h.
   \end{cases}

With mismatch probability :math:`\mu_j`, a biallelic emission is
:math:`1-\mu_j` for a match and :math:`\mu_j` for a mismatch. For :math:`A`
alleles, stable 0.4.1 distributes mismatch mass as :math:`\mu_j/(A-1)` across
the other alleles. A missing query allele has a neutral emission.

The Viterbi recurrence maximizes over the previous copying source. The executable
reference uses log probabilities and a dense :math:`O(mK^2)` loop so that the
equation is transparent. Production tsinfer compresses likelihoods on marginal
trees; it does not materialize this rectangular matrix.

Probability transforms in stable 0.4.1
--------------------------------------

If :math:`d_j` is the genetic distance in Morgans, 0.4.1 used Haldane's mapping
function

.. math::

   r_j = \frac{1-\exp(-2d_j)}{2}.

The result approaches 0.5, not 1. It is not divided by the reference-panel size.
The previous chapter's :math:`1-\exp(-d_j/K)` formula was therefore not software
parity.

For mismatch ratio :math:`X`, median adjacent genetic distance
:math:`\widetilde d`, and :math:`A` alleles, stable 0.4.1 used the constant

.. math::

   \mu = \frac{1-\exp(-A X\widetilde d)}{A}

across sites. When no rate and ratio were supplied, that release used
:math:`r=10^{-2}` and :math:`\mu=10^{-20}` to recover near-no-mismatch 0.1-era
behaviour. Defaults are versioned software decisions, not universal properties of
the method.

.. literalinclude:: ../../../watchgen/mini_tsinfer.py
   :language: python
   :pyobject: compute_recombination_probs

.. literalinclude:: ../../../watchgen/mini_tsinfer.py
   :language: python
   :pyobject: viterbi_ls
