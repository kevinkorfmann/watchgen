"""Documentation and optional official-CLI checks for the CLUES chapter."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
CHAPTER = ROOT / "docs" / "timepieces" / "clues"


def chapter_text() -> str:
    return "\n".join(path.read_text() for path in sorted(CHAPTER.glob("*.rst")))


def test_chapter_distinguishes_clues_from_clues2():
    text = chapter_text()
    assert "original CLUES" in text
    assert "CLUES2" in text
    assert "approximate full likelihood" in text


def test_chapter_states_population_size_conversion():
    text = chapter_text()
    assert "haploid" in text
    assert "divides it by two" in text
    assert "x(1 - x) / N_haploid" in text


def test_chapter_states_source_style_emission_boundary():
    text = chapter_text()
    assert "frequency-independent event-rate" in text
    assert "likelihood ratios" in text


def test_chapter_states_importance_weight_direction():
    text = chapter_text()
    assert "ell_s(G_m) - ell_0(G_m)" in text
    assert "log-mean-exp" in text


def test_chapter_does_not_claim_unqualified_full_likelihood():
    text = chapter_text().lower()
    assert "is a full-likelihood method" not in text


def test_official_clues2_cli_example(tmp_path):
    source = os.environ.get("CLUES2_DIR")
    python = os.environ.get("CLUES2_PYTHON")
    if not source or not python:
        pytest.skip("set CLUES2_DIR and CLUES2_PYTHON for official CLI parity")
    source = Path(source)
    output = tmp_path / "ancient"
    command = [
        python, str(source / "inference.py"), "--N", "30000",
        "--popFreq", "0.98", "--ancientHaps",
        str(source / "examples" / "example_haplotypes.csv"),
        "--out", str(output), "--tCutoff", "536", "--df", "80",
        "--timeBins", "89", "179", "--noAlleleTraj",
    ]
    subprocess.run(command, cwd=source, check=True, capture_output=True, text=True)
    fields = Path(f"{output}_inference.txt").read_text().splitlines()[1].split()
    assert float(fields[0]) == pytest.approx(60.0290, abs=5e-4)
    assert [float(fields[i]) for i in (4, 7, 10)] == pytest.approx(
        [0.09166, 0.09588, -0.00915], abs=5e-5
    )
