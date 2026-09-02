.. _tsdate_demo:

===============================================
Demo: Checking tsdate Kernels on Simulated Data
===============================================

   *Checking mini-tsdate kernels on one simulated genealogy.*

.. figure:: /_static/figures/fig_demo_tsdate.png
   :width: 100%
   :align: center

   **Mini-tsdate checks on msprime-simulated data.** Panel A compares simulated
   node ages with exact conditional-coalescent prior means; this is a prior
   calibration check, not a dating result. Panel B shows optional gamma moment
   matches to the conditional-coalescent prior. Panel C shows Poisson edge
   likelihoods. Panel D compares expected and observed mutations per edge.

This figure exercises source-matched kernels in ``mini_tsdate`` on a single
tree simulated with ``msprime``. It does not claim to replace the complete
production pipeline. The script is at ``figures/fig_demo_tsdate.py``.
