.. _slim_demo:

============================
Demo and Verification
============================

Run a versioned recipe with the original executable:

.. code-block:: console

   $ slim -v
   SLiM version 5.2, built <build date>
   $ slim docs/timepieces/slim/scripts/selected.slim
   selected recipe finished

The exact build line may include platform and Git information.  Verification
for this chapter used the tagged 5.2 source build, ran SLiM's own diagnostic
suite, and executed all three recipes above.  The focused Python tests check
mutation-object identity, default fitness, relative parent weights, breakpoint
parameterization and coordinates, WF tick order, and deterministic seeded
simulation.

The Python smoke test is deliberately smaller in scope:

.. code-block:: console

   $ python -m watchgen.mini_slim
   individuals: 20
   final mean fitness: 1.000

Agreement on these mechanisms does not establish parity for callbacks, nonWF
models, spatial simulation, or tree-sequence tables.  Those require direct
tests with SLiM 5.2 and the relevant downstream libraries.
