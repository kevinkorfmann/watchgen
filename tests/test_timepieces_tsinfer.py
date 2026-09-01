"""Regression checks for claims that previously drifted from tsinfer."""

from pathlib import Path

ROOT = Path(__file__).parents[1]
CHAPTER = ROOT / "docs" / "timepieces" / "tsinfer"


def chapter_text():
    return "\n".join(path.read_text() for path in sorted(CHAPTER.glob("*.rst")))


def test_chapter_marks_frequency_values_as_ordering_proxies():
    text = chapter_text().lower()
    assert "not generations" in text
    assert "time proxy" in text


def test_chapter_states_two_roots_and_multi_focal_ancestors():
    text = chapter_text().lower()
    assert "virtual root" in text
    assert "ultimate ancestor" in text
    assert "multiple focal sites" in text


def test_chapter_does_not_claim_the_teaching_code_is_production_tsinfer():
    text = chapter_text().lower()
    assert "not a replacement" in text
    assert "production tsinfer" in text


def test_chapter_records_audited_versions():
    text = chapter_text()
    assert "0.1.4" in text
    assert "0.4.1" in text
    assert "9242074" in text
