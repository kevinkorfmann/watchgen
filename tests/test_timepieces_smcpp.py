"""Documentation-regression tests for the reviewed SMC++ timepiece."""

from pathlib import Path

ROOT = Path(__file__).parents[1]
CHAPTER = ROOT / "docs" / "timepieces" / "smcpp"


def _chapter_text():
    return "\n".join(path.read_text() for path in sorted(CHAPTER.glob("*.rst")))


def test_chapter_uses_distinguished_pair_not_distinguished_lineage():
    text = _chapter_text().lower()
    assert "distinguished pair" in text
    assert "one lineage is singled out" not in text
    assert "n - 1 lineages as a demographic background" not in text


def test_chapter_identifies_csfs_as_emission_model():
    text = _chapter_text().lower()
    assert "conditioned sample-frequency spectrum" in text
    assert "independently accumulates mutations" not in text
    assert "diploid genotype follows a binomial" not in text


def test_chapter_does_not_claim_extra_samples_change_pair_transitions():
    text = _chapter_text().lower()
    assert "extra samples affect the emissions" in text
    assert "rate is h(t)" not in text
    assert "modified coalescence rate" not in text


def test_chapter_names_original_transition_model_and_thinning_caveat():
    text = _chapter_text().lower()
    assert "hobolth" in text
    assert "conditioned markov chain" in text
    assert "thinning" in text
    assert "conditional-independence" in text or "conditional independence" in text


def test_chapter_uses_canonical_literalincludes():
    text = _chapter_text()
    assert text.count("../../watchgen/mini_smcpp.py") >= 5
    assert "from smcpp_ode_helpers import" not in text


def test_split_chapter_limits_claim_to_clean_split_model():
    text = (CHAPTER / "population_splits.rst").read_text().lower()
    assert "clean split" in text
    assert "no post-split gene flow" in text
    assert "joint conditioned sample-frequency spectrum" in text
    assert "solve_split_ode" not in text
