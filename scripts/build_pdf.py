#!/usr/bin/env python3
"""
Build the Watchmaker's Guide documentation as a single PDF (book format).
Author: Kevin Korfmann (configured in docs/conf.py).

Prerequisites:
  - Python 3 venv with docs deps:  python3 -m venv .venv && .venv/bin/pip install -r docs/requirements.txt
  - A LaTeX distribution (e.g. MacTeX, TeX Live)

Usage (from repo root):
  .venv/bin/python scripts/build_pdf.py
"""

import os
import shutil
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(REPO_ROOT, "docs")
LATEX_DIR = os.path.join(DOCS_DIR, "_build", "latex")
MAIN_TEX = "watchmakers-guide.tex"


def run(cmd, cwd=None, check=True):
    print(f"  $ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=cwd or os.getcwd())
    if check and r.returncode != 0:
        sys.exit(r.returncode)
    return r.returncode


def main():
    os.chdir(REPO_ROOT)

    if not shutil.which("pdflatex"):
        print("Error: pdflatex not found. Install a LaTeX distribution "
              "(MacTeX, TeX Live).", file=sys.stderr)
        sys.exit(1)

    print("[1/2] Generating LaTeX from Sphinx sources...")
    run(
        [sys.executable, "-m", "sphinx", "-b", "latex", ".", "_build/latex"],
        cwd=DOCS_DIR,
    )

    tex_path = os.path.join(LATEX_DIR, MAIN_TEX)
    if not os.path.isfile(tex_path):
        print(f"Error: {MAIN_TEX} not found in {LATEX_DIR}", file=sys.stderr)
        sys.exit(1)

    print("[2/2] Compiling PDF (3 passes for TOC & cross-refs)...")
    for i in range(1, 4):
        print(f"  pdflatex pass {i}/3 ...")
        run(["pdflatex", "-interaction=nonstopmode", MAIN_TEX],
            cwd=LATEX_DIR, check=False)

    pdf_name = MAIN_TEX.replace(".tex", ".pdf")
    pdf_path = os.path.join(LATEX_DIR, pdf_name)
    if os.path.isfile(pdf_path):
        size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
        print(f"\nDone! {pdf_path}  ({size_mb:.1f} MB)")
    else:
        print(f"Error: PDF not found at {pdf_path}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
