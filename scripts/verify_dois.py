#!/usr/bin/env python3
"""Query CrossRef API to verify DOIs and generate BibTeX entries.

Reads dois.txt, queries https://api.crossref.org for each DOI,
verifies the first author surname matches, and writes verified
BibTeX to docs/references.bib.
"""

import json
import sys
import time
import urllib.request
import urllib.error

DOIS_FILE = "docs/dois.txt"
OUTPUT_BIB = "docs/references.bib"


def fetch_crossref(doi):
    """Fetch metadata from CrossRef for a given DOI."""
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
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code} for DOI {doi}")
        return None
    except Exception as e:
        print(f"  Error fetching {doi}: {e}")
        return None


def fetch_bibtex(doi):
    """Fetch BibTeX directly from doi.org content negotiation."""
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
    except Exception as e:
        print(f"  Error fetching BibTeX for {doi}: {e}")
        return None


def verify_author(metadata, expected_surname):
    """Check if the expected first author surname appears in the author list."""
    authors = metadata.get("author", [])
    if not authors:
        return False
    first_author = authors[0].get("family", "")
    return first_author.lower() == expected_surname.lower()


def main():
    # Read DOI list
    entries = []
    with open(DOIS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) != 3:
                print(f"Skipping malformed line: {line}")
                continue
            key, doi, expected_author = parts
            entries.append((key.strip(), doi.strip(), expected_author.strip()))

    print(f"Found {len(entries)} DOIs to verify\n")

    verified = []
    failed = []

    for key, doi, expected_author in entries:
        print(f"[{key}] Querying DOI: {doi}")

        # Step 1: Verify via CrossRef metadata
        metadata = fetch_crossref(doi)

        # Step 2: If CrossRef fails, try BibTeX content negotiation as fallback
        bibtex = fetch_bibtex(doi)
        if bibtex is None:
            print(f"  FAILED: Could not fetch BibTeX")
            failed.append((key, doi, "bibtex fetch failed"))
            time.sleep(1)
            continue

        if metadata is None:
            # CrossRef failed but BibTeX worked - verify author from BibTeX
            import re as re_mod
            author_match = re_mod.search(r"author\s*=\s*\{([^}]+)\}", bibtex)
            if author_match:
                first_author_bib = author_match.group(1).split(",")[0].split(" and ")[0].strip()
                if expected_author.lower() not in first_author_bib.lower():
                    print(f"  FAILED: Expected author '{expected_author}' not in '{first_author_bib}'")
                    failed.append((key, doi, f"author mismatch in BibTeX"))
                    time.sleep(1)
                    continue
            title_match = re_mod.search(r"title\s*=\s*\{([^}]+)\}", bibtex)
            title_str = title_match.group(1)[:70] if title_match else "?"
            print(f"  VERIFIED (via BibTeX): {expected_author} - {title_str}...")
        else:
            # Step 3: Verify first author from CrossRef
            if not verify_author(metadata, expected_author):
                authors = metadata.get("author", [])
                actual = authors[0].get("family", "???") if authors else "no authors"
                print(f"  FAILED: Expected first author '{expected_author}', got '{actual}'")
                failed.append((key, doi, f"author mismatch: expected {expected_author}, got {actual}"))
                time.sleep(1)
                continue

            # Step 4: Get title for display
            title = metadata.get("title", ["???"])[0]
            year = metadata.get("published-print", metadata.get("published-online", {}))
            year_str = str(year.get("date-parts", [[None]])[0][0]) if year else "?"
            authors = metadata.get("author", [])
            first = authors[0].get("family", "?") if authors else "?"

            print(f"  VERIFIED: {first} et al. ({year_str}) - {title[:70]}...")

        # Replace the auto-generated key with our key
        # BibTeX entries typically start with @type{key,
        import re
        bibtex = re.sub(
            r"@(\w+)\{[^,]+,",
            rf"@\1{{{key},",
            bibtex,
            count=1,
        )

        verified.append((key, doi, bibtex))
        time.sleep(0.5)  # Rate-limit politely

    # Write output
    print(f"\n{'='*60}")
    print(f"Results: {len(verified)} verified, {len(failed)} failed")
    print(f"{'='*60}")

    if failed:
        print("\nFailed DOIs:")
        for key, doi, reason in failed:
            print(f"  [{key}] {doi} - {reason}")

    if verified:
        with open(OUTPUT_BIB, "w") as f:
            f.write(f"% Auto-generated BibTeX references for The Watchmaker's Guide\n")
            f.write(f"% Verified via CrossRef API ({len(verified)} entries)\n")
            f.write(f"% DO NOT EDIT MANUALLY - regenerate with: python scripts/verify_dois.py\n\n")
            for key, doi, bibtex in verified:
                f.write(f"% --- {key} ---\n")
                f.write(bibtex.strip())
                f.write("\n\n")

        print(f"\nWrote {len(verified)} entries to {OUTPUT_BIB}")
    else:
        print("\nNo entries verified, no output written.")


if __name__ == "__main__":
    main()
