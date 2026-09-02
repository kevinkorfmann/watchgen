.. _phlash_demo:

==========================
Run the Verified Miniature
==========================

From the repository root, run:

.. code-block:: console

   uv run python -m watchgen.mini_phlash

The deterministic output reports a four-interval geometric grid, the
corresponding coalescence-time masses, their sum, and the normalized AFS log
score.  The focused parity tests additionally compare the structured transition
product with a dense matrix, the forward likelihood with latent-path enumeration,
and the Fisher recursion with both enumeration and a finite-difference score.

To run the checks:

.. code-block:: console

   uv run python -m pytest tests/test_mini_phlash.py tests/test_timepieces_phlash.py

These checks demonstrate the selected identities only.  They do not analyze a
genome, reproduce the PHLASH benchmarks, or validate posterior calibration.  For
scientific inference, install and use the official ``phlash`` package and report
its version, input filtering, mutation-rate scaling, held-out data, and fitting
options.
