from pathlib import Path

from watchgen.mini_phlash import demo

CHAPTER = Path(__file__).parents[1] / "docs" / "timepieces" / "phlash"


def test_demo_is_deterministic_and_normalized(capsys):
    demo()
    output = capsys.readouterr().out
    assert "mass sum: 1.000000000000" in output
    assert "AFS log score:" in output


def test_every_toctree_page_exists():
    index = (CHAPTER / "index.rst").read_text()
    for page in (
        "overview",
        "composite_likelihood",
        "random_discretization",
        "score_function",
        "svgd_inference",
        "demo",
    ):
        assert page in index
        assert (CHAPTER / f"{page}.rst").exists()


def test_chapter_does_not_present_placeholders_as_parity():
    text = "\n".join(path.read_text() for path in CHAPTER.glob("*.rst"))
    banned = (
        "biases cancel when averaged",
        "eliminating the systematic error",
        "placeholder likelihood",
        "simulates a noisy gradient",
        "O(LM^2) gradient computation that is 30--90x faster",
    )
    for phrase in banned:
        assert phrase not in text


def test_chapter_records_version_and_scope():
    text = (CHAPTER / "overview.rst").read_text()
    assert "1.0.6" in text
    assert "96a6e3f" in text
    assert "not a replacement" in text
