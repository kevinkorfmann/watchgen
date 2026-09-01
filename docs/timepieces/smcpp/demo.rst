.. _smcpp_demo:

============================
Demo: SMC++ Building Blocks
============================

   *Inspect the CSFS emissions and continuous-time transitions directly.*

The module demo constructs four hidden TMRCA intervals for a constant
population, calculates interval-averaged CSFS emissions for two
undistinguished haplotypes, builds the conditioned two-locus transition matrix,
and evaluates a short observation sequence.

.. literalinclude:: ../../../watchgen/mini_smcpp.py
   :language: python
   :pyobject: demo

Run it from the repository root with ``python3 -m watchgen.mini_smcpp``. The
reported initial probabilities, every transition row, and every emission table
must sum to one. The example is deliberately small because the transparent
partition-state CSFS calculation grows exponentially; it verifies the
statistical gears rather than reproducing the production program's scale or
optimizer.
