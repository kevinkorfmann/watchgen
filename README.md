# watchgen

[![Unit tests](https://img.shields.io/badge/tests-2672%20passed-brightgreen)](https://github.com/kevinkorfmann/watchgen/actions) [![CI](https://github.com/kevinkorfmann/watchgen/actions/workflows/tests.yml/badge.svg)](https://github.com/kevinkorfmann/watchgen/actions/workflows/tests.yml) [![Read the Docs](https://img.shields.io/readthedocs/watchgen)](https://watchgen.readthedocs.io) [![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

**The Watchmaker's Guide to Population Genetics** — a build-it-yourself book on the algorithms behind modern population genetics. Every concept is derived from first principles, every method reimplemented from scratch in Python. No black boxes.

**Read online:** https://watchgen.readthedocs.io &nbsp;|&nbsp; **Download PDF:** https://watchgen.readthedocs.io/_/downloads/en/latest/pdf/

> **Note:** A citable version number will be assigned in the coming days. Until then, please cite by URL and access date.

---

## What this is

Population genetics has powerful algorithms — but inaccessible ones. Most live inside papers and codebases that assume years of specialised training. This book is an attempt to change that: explicit derivations, step-by-step implementations, and unit tests for every algorithm covered.

The companion Python package `watchgen` provides 19 minimal, self-contained reimplementations — small enough to read in one sitting, complete enough to run on toy examples, tested enough to trust. Think of them as movements built on the workbench: not for production, but for understanding.

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

19 modules, ~17,500 lines of code, 2,672 unit tests. Each module depends only on NumPy and SciPy.

---

## Building the book locally

**HTML:**

```bash
pip install sphinx sphinx-rtd-theme sphinx-copybutton sphinx-design sphinxcontrib-bibtex
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

This is version 0.1 — an unverified draft. No chapter has been reviewed by a domain expert yet. Contributions that cross-check derivations, correct mistakes, improve explanations, or add chapters are very welcome. Substantial contributors will be invited as co-authors.

Open an issue or pull request on [GitHub](https://github.com/kevinkorfmann/watchgen).

---

*If you find this useful, consider [supporting with PayPal](https://www.paypal.com/donate/?hosted_button_id=VTASTXN2KAFJQ).*
