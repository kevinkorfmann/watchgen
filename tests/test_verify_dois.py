from pathlib import Path

import pytest

from scripts.verify_dois import load_entries


def test_canonical_bibliography_provides_complete_doi_inventory():
    entries = load_entries()
    assert len(entries) == 37
    assert len({key for key, _, _ in entries}) == len(entries)
    assert all(doi.startswith("10.") for _, doi, _ in entries)
    assert all(author for _, _, author in entries)


def test_loader_rejects_empty_inventory(tmp_path):
    bibliography = tmp_path / "empty.bib"
    bibliography.write_text("% no citations\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no DOI entries"):
        load_entries(bibliography)


def test_loader_rejects_citation_without_doi(tmp_path):
    bibliography = tmp_path / "missing-doi.bib"
    bibliography.write_text(
        "@article{example, author={Example, Alice}, title={A paper}}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="example"):
        load_entries(bibliography)


def test_loader_supports_multiline_entries(tmp_path):
    bibliography = tmp_path / "multiline.bib"
    bibliography.write_text(
        """@article{example,
author = {Example, Alice and Other, Bob},
doi = \"10.1000/example\",
title = {An {Example} Paper}
}
""",
        encoding="utf-8",
    )
    assert load_entries(Path(bibliography)) == [
        ("example", "10.1000/example", "Example")
    ]
