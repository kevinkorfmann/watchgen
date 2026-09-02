.. _clues_demo:

=========================
Official CLUES2 CLI Check
=========================

The chapter audit used official repository commit ``b20dc5d``.  A reduced-grid
run of its supplied ancient-haplotype example was executed as follows (``PATH``
stands for the cloned repository):

.. code-block:: console

   $ python PATH/inference.py \
       --N 30000 --popFreq 0.98 \
       --ancientHaps PATH/examples/example_haplotypes.csv \
       --out ancient --tCutoff 536 --df 80 \
       --timeBins 89 179 --noAlleleTraj

The resulting row was:

.. code-block:: text

   logLR   -log10(p-value)   SelectionMLE1   SelectionMLE2   SelectionMLE3
   60.0290 25.13             0.09166         0.09588        -0.00915

This is an executable smoke/parity fixture, not a biological result: the grid
was deliberately reduced from the example's production setting.  A real
analysis should increase ``--df`` until estimates converge and should use
population size, cutoff, epochs, polarization, and ancient likelihoods justified
for that dataset.

The teaching smoke test prints the same source-pinned transition fixture:

.. code-block:: console

   $ python -m watchgen.mini_clues
   middle transition row:
   [0. ... 0.00420115 0.99303689 0.00276196 ... 0.]

Verification also checks the backward mean and variance, Beta-quantile grid,
coalescent and genotype emissions, importance-ratio direction, HMM recursion
against brute-force path enumeration, and the factor of two in the reported
likelihood-ratio statistic.
