# watchgen

**The Watchmaker's Guide to Population Genetics** — a build-it-yourself book that teaches the algorithms behind modern population genetics from first principles. Every concept is derived, every method implemented from scratch in Python. No black boxes.

## Philosophy

Like a watchmaker who understands every gear: you don't just *run* the tools, you learn how to *build* them. Math, code, and verification in one place.

## Building the book

**HTML (recommended for reading):**

```bash
cd docs
pip install -r requirements.txt
make html
# open _build/html/index.html
```

**PDF:**

```bash
python -m venv .venv && .venv/bin/pip install -r docs/requirements.txt
.venv/bin/python scripts/build_pdf.py
```

Requires a LaTeX distribution (e.g. MacTeX, TeX Live). Output: `docs/_build/latex/watchmakers-guide.pdf`.

## Structure

- **Philosophy** — why we build from scratch
- **The Workbench** — prerequisites (coalescent theory, HMMs, SMC, …)
- **Timepieces** — full algorithms (ARGweaver, PSMC, msprime, tsinfer, tsdate, momi2, moments, threads, lshmm, singer) with math and code

See [docs/index.rst](docs/index.rst) for the full table of contents.
