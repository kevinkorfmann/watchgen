.. _tsinfer_timepiece:

=====================
Timepiece VI: tsinfer
=====================

*Scalable tree-sequence inference from phased genetic variation*

tsinfer infers a tree sequence by first constructing putative ancestral
haplotypes, then matching younger ancestors against older ones, and finally
matching the observed samples. Its output is a compact genealogical hypothesis,
not a posterior distribution of ARGs. :cite:`tsinfer`

The important qualification is that the values used to order inferred ancestors
are a **time proxy**, not generations. In the paper release, larger derived-allele
counts normally implied older ordering ranks. Frequency is noisy under drift,
selection, recurrent mutation, ancestral-state error, and genotype error, so the
ordering must not be interpreted as a molecular clock.

.. admonition:: Scope of this chapter

   The executable module reproduces small mechanisms from the paper-era 0.1.4
   reference implementation and probability transforms from stable release 0.4.1.
   It is **not a replacement** for production tsinfer. Production code stores and
   matches partial ancestors on tree sequences, supports missing and multiallelic
   data in version-dependent ways, compresses shared paths, and delegates general
   table operations to tskit.

The audited ground truth was the primary paper, official tag 0.1.4
(``efbafff``), stable package 0.4.1, and official development source
``9242074`` (30 June 2026). The current development interface uses VCF-Zarr and
TOML pipeline configuration and is under active development; consult the docs for
the version you install rather than copying an old command line from this book.

.. code-block:: text

   phased variants
         |
         v
   choose inference sites --> generate partial ancestors
                                  |
                                  v
                          match older to younger
                                  |
                                  v
                          match observed samples
                                  |
                                  v
                     place remaining mutations by parsimony
                                  |
                                  v
                   post-process and simplify with tskit

.. admonition:: Primary sources

   - Kelleher et al. (2019), `doi:10.1038/s41588-019-0483-y
     <https://doi.org/10.1038/s41588-019-0483-y>`_.
   - `Official tsinfer documentation <https://tskit.dev/tsinfer/docs/latest/>`_.
   - `Official source repository <https://github.com/tskit-dev/tsinfer>`_.

.. toctree::
   :maxdepth: 2

   overview
   ancestor_generation
   copying_model
   ancestor_matching
   sample_matching
   demo
