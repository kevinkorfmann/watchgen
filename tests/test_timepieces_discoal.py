"""Documentation/source parity gates for the discoal Timepiece."""

from pathlib import Path
import re

import numpy as np

from watchgen import mini_discoal


ROOT = Path(__file__).parents[1]
DOCS = ROOT / "docs" / "timepieces" / "discoal"
FILES = sorted(DOCS.glob("*.rst"))
TEXT = "\n".join(path.read_text() for path in FILES)


def test_all_chapter_files_are_present():
    assert {path.name for path in FILES} == {
        "allele_trajectory.rst",
        "demo.rst",
        "index.rst",
        "msprime_comparison.rst",
        "overview.rst",
        "structured_coalescent.rst",
        "sweep_types.rst",
    }


def test_primary_sources_and_audited_upstream_commit_are_named():
    for citation in [
        ":cite:`discoal`",
        ":cite:`coop_griffiths_2004`",
        ":cite:`braverman_1995`",
        ":cite:`msprime2`",
    ]:
        assert citation in TEXT
    assert "7d0955f4107053c135d2086790b0426457147a8e" in TEXT
    assert "82971bf" in TEXT


def test_chapter_no_longer_claims_the_mini_is_the_full_program():
    forbidden = [
        "faithful translation of the C code",
        "complete, self-contained Python implementation of discoal",
        "array of haplotypes, :math:`O(nS)` memory",
        "r/(r+s)",
        "Maynard Smith & Haigh approximation",
    ]
    for phrase in forbidden:
        assert phrase not in TEXT
    assert "single-locus" in TEXT
    assert "not a chromosome-scale ARG simulator" in TEXT


def test_scaling_conventions_are_explicit():
    assert "4N" in TEXT
    assert "2N" in TEXT
    assert "``ploidy=1``" in TEXT
    assert "whole-locus" in TEXT


def test_production_feature_descriptions_match_source_surface():
    for flag in ["``-wd``", "``-ws``", "``-wn``", "``-f``", "``-uA``", "``-c``"]:
        assert flag in TEXT
    for capability in ["gene conversion", "recurrent", "admixture", "ancient samples"]:
        assert capability in TEXT.lower()
    assert "population 0" in TEXT


def test_module_functions_referenced_by_docs_exist():
    names = set(re.findall(r"``mini_discoal\.([A-Za-z_]\w*)``", TEXT))
    assert names
    missing = sorted(
        name for name in names if name != "py" and not hasattr(mini_discoal, name)
    )
    assert missing == []


def test_literalincludes_resolve():
    for path in FILES:
        for relative in re.findall(r"\.\. literalinclude::\s+(\S+)", path.read_text()):
            assert (path.parent / relative).resolve().exists(), (path, relative)


def test_deterministic_values_shown_in_chapter_match_executable_module():
    alpha = 200.0
    values = mini_discoal.discoal_deterministic_frequency(
        np.array([0.0, 0.02, 0.04]), alpha
    )
    for value in values:
        assert f"{value:.6f}" in TEXT


def test_no_stale_removed_api_names_remain():
    for name in [
        "discoal_core",
        "minimal_discoal",
        "expected_independent_origins",
        "sweep_duration_table",
        "hard_sweep_genealogy",
        "soft_sweep_standing_variation",
    ]:
        assert name not in TEXT


def test_rst_headings_have_matching_underlines():
    for path in FILES:
        lines = path.read_text().splitlines()
        for i in range(len(lines) - 1):
            if lines[i + 1] and set(lines[i + 1]) <= set("=-~^"):
                assert len(lines[i + 1]) >= len(lines[i]), (path, i + 1)
