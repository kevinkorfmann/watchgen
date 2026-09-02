#!/usr/bin/env python3
"""Verify every DOI recorded in the canonical project bibliography.

The maintained source of citation metadata is ``docs/references.bib``. This
script derives each citation key, DOI, and expected first-author surname from
that file, then checks the DOI through Crossref and doi.org content
negotiation. It never rewrites the bibliography that it is auditing.
"""

import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BIBLIOGRAPHY_FILE = Path("docs/references.bib")


def fetch_crossref(doi):
    """Fetch metadata from Crossref for a DOI."""
    url = f"https://api.crossref.org/works/{doi}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "WatchgenBot/1.0 (mailto:watchgen@example.com)",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            return data["message"]
    except urllib.error.HTTPError as error:
        print(f"  HTTP {error.code} for DOI {doi}")
        return None
    except (
        json.JSONDecodeError,
        KeyError,
        TimeoutError,
        UnicodeDecodeError,
        urllib.error.URLError,
    ) as error:
        print(f"  Error fetching {doi}: {error}")
        return None


def fetch_bibtex(doi):
    """Fetch BibTeX from doi.org content negotiation."""
    url = f"https://doi.org/{doi}"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/x-bibtex; charset=utf-8",
            "User-Agent": "WatchgenBot/1.0 (mailto:watchgen@example.com)",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8")
    except (TimeoutError, UnicodeDecodeError, urllib.error.URLError) as error:
        print(f"  Error fetching BibTeX for {doi}: {error}")
        return None


def verify_author(metadata, expected_surname):
    """Return whether Crossref reports the expected first-author surname."""
    authors = metadata.get("author", [])
    if not authors:
        return False
    first_author = authors[0].get("family", "")
    return first_author.casefold() == expected_surname.casefold()


def _bibtex_entries(text):
    """Yield ``(key, body)`` pairs using balanced entry braces."""
    entry_start = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,", re.IGNORECASE)
    position = 0
    while match := entry_start.search(text, position):
        depth = 1
        cursor = match.end()
        while cursor < len(text) and depth:
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth:
            raise ValueError(f"unterminated BibTeX entry {match.group(1)!r}")
        yield match.group(1), text[match.end() : cursor - 1]
        position = cursor


def _field(body, name):
    pattern = re.compile(
        rf"\b{re.escape(name)}\s*=\s*(?:\{{([^{{}}]+)\}}|\"([^\"]+)\")",
        re.IGNORECASE,
    )
    match = pattern.search(body)
    if match is None:
        return None
    return next(value.strip() for value in match.groups() if value is not None)


def _first_author_surname(author_field):
    first_author = re.split(
        r"\s+and\s+", author_field, maxsplit=1, flags=re.IGNORECASE
    )[0]
    if "," in first_author:
        surname = first_author.split(",", 1)[0]
    else:
        surname = first_author.rsplit(maxsplit=1)[-1]
    return surname.strip().strip("{}")


def load_entries(path=BIBLIOGRAPHY_FILE):
    """Load DOI audit tuples from the maintained BibTeX bibliography."""
    text = Path(path).read_text(encoding="utf-8")
    entries = []
    missing = []
    for key, body in _bibtex_entries(text):
        doi = _field(body, "doi")
        author = _field(body, "author")
        if doi is None or author is None:
            missing.append(key)
            continue
        entries.append((key, doi, _first_author_surname(author)))

    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"bibliography entries missing DOI or author: {joined}")
    if not entries:
        raise ValueError("bibliography contains no DOI entries")
    return entries


def _bibtex_first_author(bibtex):
    author = _field(bibtex, "author")
    return None if author is None else _first_author_surname(author)


def main():
    """Verify the canonical bibliography; return a process exit status."""
    try:
        entries = load_entries()
    except (OSError, ValueError) as error:
        print(f"DOI inventory error: {error}", file=sys.stderr)
        return 2

    print(f"Found {len(entries)} bibliography DOIs to verify\n")
    verified = []
    failed = []

    for key, doi, expected_author in entries:
        print(f"[{key}] Querying DOI: {doi}")
        metadata = fetch_crossref(doi)
        bibtex = fetch_bibtex(doi)
        if bibtex is None:
            print("  FAILED: Could not fetch BibTeX")
            failed.append((key, doi, "BibTeX fetch failed"))
            continue

        if metadata is None:
            actual = _bibtex_first_author(bibtex)
            if actual is None or expected_author.casefold() != actual.casefold():
                print(
                    f"  FAILED: Expected first author {expected_author!r}, got {actual!r}"
                )
                failed.append((key, doi, "author mismatch in BibTeX"))
                continue
            print(f"  VERIFIED via doi.org: {actual}")
        else:
            if not verify_author(metadata, expected_author):
                authors = metadata.get("author", [])
                actual = authors[0].get("family", "???") if authors else "no authors"
                print(
                    f"  FAILED: Expected first author {expected_author!r}, got {actual!r}"
                )
                failed.append((key, doi, "Crossref author mismatch"))
                continue
            title = metadata.get("title", ["???"])[0]
            print(f"  VERIFIED: {expected_author} - {title[:70]}...")

        verified.append((key, doi))
        time.sleep(0.1)

    print(f"\n{'=' * 60}")
    print(f"Results: {len(verified)} verified, {len(failed)} failed")
    print("=" * 60)
    if failed:
        print("\nFailed DOIs:")
        for key, doi, reason in failed:
            print(f"  [{key}] {doi} - {reason}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
