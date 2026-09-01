"""Documentation/source parity gates for the Relate Timepiece."""

from pathlib import Path
import re

from watchgen import mini_relate


ROOT = Path(__file__).parents[1]
DOCS = ROOT / "docs" / "timepieces" / "relate"
FILES = sorted(DOCS.glob("*.rst"))
TEXT = "\n".join(path.read_text() for path in FILES)


def test_all_relate_pages_are_present():
    assert {path.name for path in FILES} == {
        "asymmetric_painting.rst",
        "branch_lengths.rst",
        "demo.rst",
        "index.rst",
        "overview.rst",
        "population_size.rst",
        "tree_building.rst",
    }


def test_primary_paper_and_audited_source_are_named():
    assert ":cite:`relate`" in TEXT
    assert "https://github.com/MyersGroup/relate" in TEXT
    assert "b54ede259cbb0be095bc9c9a8bd18cdaf7e88b74" in TEXT
    for source in [
        "fast_painting.cpp",
        "tree_builder.cpp",
        "anc_builder.cpp",
        "branch_length_estimator.cpp",
    ]:
        assert f"``{source}``" in TEXT


def test_directional_emission_is_documented_without_invented_weights():
    assert "target-derived/reference-ancestral" in TEXT
    assert "The other three allele pairs" in mini_relate.modified_emission.__doc__
    assert "``w_d`` and ``w_a``" in TEXT
    assert "user-chosen mismatch weight" in TEXT


def test_tree_builder_rule_and_production_fallback_are_explicit():
    for phrase in [
        "mutual row minima",
        "cardinality-weighted mean",
        "smallest symmetrized score",
        "0.2",
    ]:
        assert phrase in TEXT
    assert "smallest value in either direction" in TEXT


def test_mutation_mapping_thresholds_match_supplement():
    assert "70%" in TEXT
    assert "0.03" in TEXT
    assert "smallest exact collection of branches" in TEXT
    assert "mutation count is divided" in TEXT


def test_branch_length_target_and_moves_are_not_overclaimed():
    for phrase in [
        "Poisson",
        "0.9",
        "0.8",
        "max(10N,1000)",
        "at least 20 proposals",
        "fixed-event-order",
    ]:
        assert phrase in TEXT.replace("\\", "")
    assert "does not associate branches" in TEXT


def test_population_rate_equation_and_iteration_are_described():
    assert "events divided by exposure" in TEXT
    assert "five cycles" in TEXT
    assert "not the panmictic population-wide MLE" in TEXT
    assert "does not silently" in TEXT


def test_former_whole_program_parity_claims_are_absent():
    forbidden = [
        "complete genealogy estimation engine from scratch",
        "five gears",
        "two independent phases",
        "w_d=1.0",
        "w_a=0.5",
        "em_population_size",
        "m_step",
    ]
    for phrase in forbidden:
        assert phrase not in TEXT
    assert "not a reimplementation" in TEXT
    assert "not labelled as Relate executable output" in TEXT


def test_documented_mini_functions_exist():
    names = set(re.findall(r"``mini_relate\.([A-Za-z_]\w*)``", TEXT))
    assert names
    assert all(hasattr(mini_relate, name) for name in names)


def test_literalincludes_resolve_and_pyobjects_exist():
    for path in FILES:
        contents = path.read_text()
        includes = re.findall(r"\.\. literalinclude::\s+(\S+)", contents)
        assert all((path.parent / item).resolve().exists() for item in includes)
        for pyobject in re.findall(r":pyobject:\s+(\w+)", contents):
            assert hasattr(mini_relate, pyobject)


def test_rst_heading_underlines_are_long_enough():
    for path in FILES:
        lines = path.read_text().splitlines()
        for index in range(len(lines) - 1):
            underline = lines[index + 1]
            if underline and set(underline) <= set("=-~^"):
                assert len(underline) >= len(lines[index]), (path, index + 1)
