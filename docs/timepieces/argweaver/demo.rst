.. _argweaver_demo:

====================================
Demo: Source-Matched Teaching Checks
====================================

   *Checking the bounded kernels without pretending to run the production sampler.*

.. figure:: /_static/figures/fig_demo_argweaver.png
   :width: 100%
   :align: center

   **ARGweaver teaching checks.** Panel A compares exact Jukes--Cantor branch
   probabilities with their low-rate approximations. Panel B shows the logarithmic
   time grid. Panel C shows re-coalescence mass under the correct :math:`2N_e`
   hazard. Panel D compares simulated coalescent-prior TMRCA values with their
   analytic mean.

This figure validates individual teaching kernels. It is not an ARGweaver MCMC run
and contains no posterior trace. The production executable validation is performed
separately against the original ARGweaver source. The plotting script is
``figures/fig_demo_argweaver.py``.
