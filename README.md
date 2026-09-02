# watchgen

[![Unit tests](https://img.shields.io/badge/tests-1849%20passed-brightgreen)](https://github.com/kevinkorfmann/watchgen/actions) [![Codex verification](https://img.shields.io/badge/Codex%20verification-100%25-brightgreen)](CODEX_VERIFICATION.md) [![Mini parity](https://img.shields.io/badge/mini%20parity-18%2F18-brightgreen)](CODEX_VERIFICATION.md#mini-implementation-parity-batch) [![CI](https://github.com/kevinkorfmann/watchgen/actions/workflows/tests.yml/badge.svg)](https://github.com/kevinkorfmann/watchgen/actions/workflows/tests.yml) [![Read the Docs](https://img.shields.io/readthedocs/watchgen)](https://watchgen.readthedocs.io) [![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

**The Watchmaker's Guide to Population Genetics** — a build-it-yourself book on the algorithms behind modern population genetics. Every concept is derived from first principles, every method reimplemented from scratch in Python. No black boxes.

**Read online:** https://watchgen.readthedocs.io &nbsp;|&nbsp; **Download PDF:** https://watchgen.readthedocs.io/_/downloads/en/latest/pdf/

**Codex verification: 100% — all 26 prerequisite and Timepiece chapters and all
18 mini implementations passed the final clean-clone gate.** The gate reported
**1,849 tests passed with 7 expected optional/private skips**, all 36 published
figure scripts passed, strict HTML/dummy/nitpicky and 733-page PDF builds passed,
and all 37 bibliography DOIs verified. See the
[verification ledger](CODEX_VERIFICATION.md) for chapter-level evidence.

**Mini implementation parity: 100% — 18 of 18 mini implementations reviewed.**
The [mini parity batch](CODEX_VERIFICATION.md#mini-implementation-parity-batch)
tracks source, upstream-fixture, and independent-oracle coverage separately from
general chapter verification.

> **Version 0.0.2:** This is a living draft under technical audit. When citing the book, please include version 0.0.2, the project URL, and the access date.

---

## What this is

Population genetics has powerful algorithms — but inaccessible ones. Most live inside papers and codebases that assume years of specialised training. This book is an attempt to change that: explicit derivations, step-by-step implementations, and unit tests for every algorithm covered.

The companion Python package `watchgen` provides 18 minimal, self-contained reimplementations — small enough to read in one sitting, complete enough to run on toy examples, tested enough to trust. Think of them as movements built on the workbench: not for production, but for understanding.

---

## Contents

**Prerequisites (8 chapters)**

Coalescent theory, ARGs, HMMs, SMC, diffusion approximation, ODEs, MCMC, probabilistic inference — everything you need before tackling a Timepiece.

**Timepieces (18 algorithms)**

| Category | Algorithms |
|---|---|
| Simulators | msprime, SLiM, discoal |
| Demographic inference | PSMC, SMC++, Gamma-SMC, PHLASH |
| SFS-based inference | moments, dadi, momi2 |
| Genealogy & ARG inference | Li & Stephens HMM, ARGweaver, tsinfer, SINGER, Threads, Relate |
| Dating & selection | tsdate, CLUES |

---

## The `watchgen` package

```python
pip install watchgen  # or: git clone + pip install -e .
```

```python
from watchgen import mini_psmc, mini_msprime, mini_tsinfer  # etc.
```

18 teaching implementations across 19 Python modules, ~17,500 lines of code,
and 1,849 passing public tests in the final repository gate. The package uses
the Python standard library, NumPy, SciPy, msprime, and tskit.

---

## Building the book locally

**HTML:**

```bash
pip install sphinx sphinx-book-theme sphinx-copybutton sphinx-design sphinxcontrib-bibtex
python -m sphinx docs docs/_build/html -b html
open docs/_build/html/index.html
```

**PDF** (requires XeLaTeX / MacTeX / TeX Live):

```bash
python -m sphinx docs docs/_build/latex -b latex
cd docs/_build/latex && xelatex watchmakers-guide.tex
```

---

## Contributing

This is version 0.0.2 — a technically audited living draft. No chapter has been reviewed by a domain expert yet. Contributions that cross-check derivations, correct mistakes, improve explanations, or add chapters are very welcome.
Open an issue or pull request on [GitHub](https://github.com/kevinkorfmann/watchgen).
