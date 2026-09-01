.. _singer_demo:

========================
Executable mechanisms
========================

Run the scalar demonstration with

.. code-block:: bash

   python -m watchgen.mini_singer

It prints a representative joining time, a branch joining probability, and a
tree-height acceptance ratio.  The accompanying tests compare the equations
against direct integrations, row-normalized interval probabilities, a dense
forward update, and explicit rescaling calculations.

The generated figures visualize these mechanisms only.  An msprime tree
sequence is simulated truth, not SINGER output; plotting its marginal trees
does not validate ARG inference.  For a real analysis, run the authors'
``singer_master`` on phased VCF data, retain multiple thinned ARG samples,
convert them with the provided tskit utility, and examine the convergence
traces recommended by the software authors.
