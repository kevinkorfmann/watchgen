"""Documentation and executable-recipe regression tests for SLiM."""

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
CHAPTER = ROOT / "docs" / "timepieces" / "slim"
SCRIPTS = CHAPTER / "scripts"


def chapter_text() -> str:
    return "\n".join(path.read_text() for path in sorted(CHAPTER.glob("*.rst")))


def test_chapter_has_explicit_version_and_scope_boundary():
    text = chapter_text()
    assert "SLiM 5.2" in text
    assert "teaching model" in text
    assert "not a reimplementation" in text


def test_chapter_distinguishes_wf_and_nonwf_fitness():
    text = chapter_text()
    assert "relative reproductive success" in text
    assert "absolute survival" in text


def test_chapter_corrects_identity_and_recombination_claims():
    text = chapter_text()
    assert "same mutation object" in text
    assert "-log(1 - p)" in text
    assert "L - 1" in text


def test_chapter_does_not_claim_complete_pedigree_tree_recording():
    text = chapter_text().lower()
    assert "complete pedigree" not in text
    assert "retained ancestry" in text


def test_all_literalinclude_scripts_exist():
    for rst in CHAPTER.glob("*.rst"):
        for relative in re.findall(r"\.\. literalinclude::\s+(.+)", rst.read_text()):
            assert (rst.parent / relative.strip()).resolve().is_file()


def slim_binary() -> str:
    binary = os.environ.get("SLIM_BIN")
    if not binary:
        pytest.skip("set SLIM_BIN to a SLiM 5.2 executable for recipe parity")
    version = subprocess.run(
        [binary, "-v"], check=True, text=True, capture_output=True
    ).stdout
    if "SLiM version 5.2" not in version:
        pytest.skip(f"recipe parity requires SLiM 5.2, found: {version.strip()}")
    return binary


@pytest.mark.parametrize("script", sorted(SCRIPTS.glob("*.slim")), ids=lambda p: p.name)
def test_versioned_recipes_execute_in_slim_5_2(script):
    result = subprocess.run(
        [slim_binary(), str(script)], check=True, text=True, capture_output=True
    )
    assert "ERROR" not in result.stderr
    assert "WARNING" not in result.stderr
    assert "finished" in result.stdout.lower()
