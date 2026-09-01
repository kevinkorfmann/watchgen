.. _smcpp_splits:

=================
Population Splits
=================

   *Condition the joint frequency spectrum under a clean split.*

Scope of the model
==================

The published SMC++ extension considers a two-population **clean split**.
Looking backward, the populations remain separate until time
:math:`t_S`; before that ancestral time they occupy a common population. The
model assumes no post-split gene flow. It therefore estimates marginal size
histories and a divergence time under isolation, not a general migration or
admixture history.

Two distinguished-pair configurations are required:

``together``
   Both distinguished haplotypes are sampled from the same population.

``apart``
   One distinguished haplotype is sampled from each population.

The remaining haplotypes contribute a joint allele-count observation across
the two populations.

The joint conditioned spectrum
==============================

The **joint conditioned sample-frequency spectrum** (JCSFS) is the central
emission object for the clean-split extension.

For populations 1 and 2, define

.. math::

   \operatorname{JCSFS}(\tau,t_S)
   \in\mathbb{R}^{(a_1+1)\times(n_1+1)\times(a_2+1)\times(n_2+1)},

where :math:`a_1+a_2=2` counts distinguished haplotypes and
:math:`n_1,n_2` count undistinguished haplotypes. The tensor records expected
branch lengths by descendant counts in both populations, conditional on the
distinguished-pair TMRCA :math:`\tau` and split time :math:`t_S`.

This is the multi-population analogue of the one-population CSFS. Production
SMC++ computes it using the same coalescent/Moran duality used by ``momi``. It
does not solve two independent lineage-count ODEs and then convolve expected
counts; such a reduction would discard the descendant configuration that
determines joint allele frequencies.

Support of an apart-pair TMRCA
==============================

For an apart pair, the two distinguished ancestors cannot coalesce while they
reside in separate populations. Consequently,

.. math::

   P(T>t)=
   \begin{cases}
   1,&0\le t\le t_S,\\
   \exp\!\left[-\int_{t_S}^{t}\alpha_A(u)\,du\right],&t>t_S,
   \end{cases}

where :math:`\alpha_A` is the pairwise coalescence rate in the ancestral
population. This support constraint is one source of information about the
split time.

.. literalinclude:: ../../../watchgen/mini_smcpp.py
   :language: python
   :pyobject: cross_population_survival

The survival function alone is not the split likelihood. The JCSFS also links
the distinguished pair to derived counts among all sampled haplotypes, and the
HMM transition process supplies linkage information along the genome.

How the command-line workflow fits a split
==========================================

The current software first fits each population marginally with ``estimate``.
Users then create two-population SMC files, normally in both population orders,
and run ``split`` with the marginal model files plus the joint data. The split
stage refines the marginal histories jointly with the clean-split parameter.
It does not start by estimating four arbitrary histories from scratch, and it
does not infer ongoing migration.

Interpretation
==============

An estimated split time is conditional on the clean-split model, mutation
rate, masks, distinguished-pair construction, and regularization settings.
Population structure or gene flow after divergence can violate that model and
shift the fitted time. The output should therefore be described as a clean-
split estimate rather than an unconditional date of population separation.

What this mini implementation omits
===================================

The small Python module demonstrates the apart-pair support constraint but does
not implement the full JCSFS tensor or split optimizer. Calling the former a
complete SMC++ split implementation would be misleading. For inference, use
the original software; for understanding, the one-population partition engine
shows why descendant labels, rather than lineage-count expectations, are the
essential state of the frequency-spectrum calculation.
