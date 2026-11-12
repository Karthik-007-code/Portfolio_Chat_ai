#!/usr/bin/env python3
"""
scripts/ingest_knowledge.py

Standalone CLI script that gathers publicly available information about
Karthik and saves it as supplementary JSON files inside knowledge/.

Usage
-----
    # From the repo root, with the venv active:
    uv run python scripts/ingest_knowledge.py

    # Ingest only GitHub:
    uv run python scripts/ingest_knowledge.py --source github

    # Ingest only portfolio:
    uv run python scripts/ingest_knowledge.py --source portfolio

    # Dry-run (fetch data but do NOT write files):
    uv run python scripts/ingest_knowledge.py --dry-run

Output files
------------
    knowledge/github.scraped.json
    knowledge/portfolio.scraped.json

These files are SEPARATE from the hand-maintained JSON files.
knowledge_service.py merges them at startup without overwriting
any hand-maintained data.

Run this script:
    - Before the first deployment (to populate scraped data).
    - Whenever your GitHub profile or public repos change significantly.
    - On a schedule if you want the assistant to stay current.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# ------------------------------------------------------------------ #
# Path bootstrap
# Allow running this script directly with `python scripts/ingest_knowledge.py`
# even if the package is not installed (adds src/ to sys.path).
# ------------------------------------------------------------------ #
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from portfolio_chat_ai.services.scraping_service import ingest_github, ingest_portfolio

# ------------------------------------------------------------------ #
# Configuration — edit these if your username or URL changes
# ------------------------------------------------------------------ #
GITHUB_USERNAME = "Karthik-007-code"
PORTFOLIO_URL = "https://karthik-007-code.github.io/karthik-007-code/"


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def _write_json(path: Path, data: dict, dry_run: bool) -> None:
    """Write data to a JSON file, or just print it if dry_run is True."""
    pretty = json.dumps(data, indent=2, ensure_ascii=False)
    if dry_run:
        print(f"\n--- DRY RUN: would write to {path} ---")
        print(pretty[:500] + ("..." if len(pretty) > 500 else ""))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(pretty, encoding="utf-8")
        print(f"  Saved → {path}")


def run_github_ingestion(knowledge_dir: Path, dry_run: bool) -> bool:
    """Fetch GitHub data and write knowledge/github.scraped.json."""
    print(f"\n[1/2] Ingesting GitHub public profile ({GITHUB_USERNAME})...")
    try:
        data = ingest_github(GITHUB_USERNAME, knowledge_dir / "github.scraped.json")
    except Exception as exc:
        logging.error("GitHub ingestion failed: %s", exc)
        return False

    if not data.get("profile"):
        print("  WARNING: GitHub profile returned empty. Skipping write.")
        return False

    repo_count = len(data.get("repositories", []))
    print(f"  Profile fetched. Original repos found: {repo_count}")
    _write_json(knowledge_dir / "github.scraped.json", data, dry_run)
    return True


def run_portfolio_ingestion(knowledge_dir: Path, dry_run: bool) -> bool:
    """Fetch portfolio metadata and write knowledge/portfolio.scraped.json."""
    print(f"\n[2/2] Ingesting portfolio website ({PORTFOLIO_URL})...")
    print(
        "  Note: This site is a React SPA — only static metadata "
        "(title, description, og: tags) can be extracted."
    )
    try:
        data = ingest_portfolio(PORTFOLIO_URL, knowledge_dir / "portfolio.scraped.json")
    except Exception as exc:
        logging.error("Portfolio ingestion failed: %s", exc)
        return False

    meta = data.get("metadata", {})
    extracted_keys = [k for k in meta if k not in ("url", "note")]
    print(f"  Metadata keys extracted: {extracted_keys if extracted_keys else '(none)'}")
    _write_json(knowledge_dir / "portfolio.scraped.json", data, dry_run)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest public information into the knowledge base.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        choices=["github", "portfolio", "all"],
        default="all",
        help="Which source to ingest (default: all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch data but do NOT write any files.",
    )
    parser.add_argument(
        "--knowledge-dir",
        type=Path,
        default=_REPO_ROOT / "knowledge",
        help="Path to the knowledge directory (default: knowledge/).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging.",
    )

    args = parser.parse_args()
    _setup_logging(args.verbose)

    print("=" * 55)
    print("  Ask Karthik — Knowledge Ingestion Script")
    print("=" * 55)
    if args.dry_run:
        print("  MODE: DRY RUN (no files will be written)")
    print(f"  Knowledge dir: {args.knowledge_dir}")
    print(f"  Source(s)    : {args.source}")

    success_count = 0
    fail_count = 0

    if args.source in ("github", "all"):
        ok = run_github_ingestion(args.knowledge_dir, args.dry_run)
        if ok:
            success_count += 1
        else:
            fail_count += 1

    if args.source in ("portfolio", "all"):
        ok = run_portfolio_ingestion(args.knowledge_dir, args.dry_run)
        if ok:
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "=" * 55)
    print(f"  Done. Success: {success_count}  Failed: {fail_count}")
    if not args.dry_run and success_count > 0:
        print("  Restart the FastAPI server to load the updated knowledge.")
    print("=" * 55)

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

