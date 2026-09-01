Auditable example
=================

The corrected miniature is intentionally a set of mechanisms rather than a fake
end-to-end inference engine. Run its deterministic example with:

.. code-block:: bash

   python -m watchgen.mini_tsinfer

The test suite is more informative than the printout. It checks that equal carrier
patterns can create multiple focal sites, intervening older polymorphism splits an
ancestor, both zero ancestors are present, Haldane and mismatch transforms match
stable 0.4.1, dense Viterbi agrees with exhaustive path enumeration, edge coordinates
use zero and sequence length, and path compression requires a repeated multi-edge
run.

For actual inference, install a released tsinfer version, follow that version's
official input documentation, run the production pipeline, and inspect the emitted
provenance. The current development documentation uses VCF-Zarr inputs and TOML
configuration; older ``SampleData`` examples are historical and should not be
silently presented as the latest interface.

Reproducibility checklist
-------------------------

#. Record tsinfer, tskit, and input-library versions.
#. Record phasing, ancestral-state, site, and sample filters.
#. Record recombination maps, mismatch settings, and path-compression choices.
#. Distinguish proxy node order from calibrated dates; use an explicit dating
   method if times in generations are required.
#. Verify genotype representation and tree-sequence validity with production APIs.
#. Preserve provenance and logs with the inferred ``.trees`` artifact.
