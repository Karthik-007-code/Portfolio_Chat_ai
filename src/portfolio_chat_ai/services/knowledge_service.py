"""
services/knowledge_service.py

Loads all JSON knowledge files from the knowledge/ directory and formats
them into a structured dictionary ready for prompt injection.

Design decisions
----------------
- Files are loaded ONCE at application startup via the FastAPI lifespan
  handler in main.py and stored in app.state.knowledge.  This avoids
  repeated disk I/O on every chat request.

- Missing files produce a warning rather than a crash.  This allows the
  application to start even if some JSON files are not yet populated.

- Hand-maintained files (e.g. profile.json) and scraped supplement files
  (e.g. github.scraped.json) are merged, with hand-maintained data always
  taking precedence.

- The public API of this module is intentionally simple:
      knowledge = load_all_knowledge(knowledge_dir)
  Returns a plain dict[str, Any] consumed by core/prompts.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Ordered list of hand-maintained knowledge files.
# The order determines the section order in the system prompt.
KNOWLEDGE_FILES: list[str] = [
    "profile.json",
    "education.json",
    "skills.json",
    "experience.json",
    "projects.json",
    "achievements.json",
    "certifications.json",
]


def _load_json(path: Path) -> dict[str, Any] | None:
    """
    Read and parse a single JSON file.

    Returns None (and logs a warning) if the file is missing or malformed
    rather than raising an exception.  The application can still function
    with partial knowledge.
    """
    if not path.exists():
        logger.warning("Knowledge file not found, skipping: %s", path)
        return None

    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
            logger.debug("Loaded knowledge file: %s", path.name)
            return data
    except json.JSONDecodeError as exc:
        logger.error(
            "Failed to parse knowledge file %s: %s", path.name, exc
        )
        return None


def _load_scraped_supplements(knowledge_dir: Path) -> dict[str, Any]:
    """
    Load all *.scraped.json files from the knowledge directory.

    These are produced by the scraping_service and contain publicly gathered
    information.  They are merged into the knowledge dict under a
    'scraped' top-level key so they remain clearly separated from
    hand-maintained data.
    """
    scraped: dict[str, Any] = {}
    for path in sorted(knowledge_dir.glob("*.scraped.json")):
        source_name = path.name.replace(".scraped.json", "")
        data = _load_json(path)
        if data is not None:
            scraped[source_name] = data
            logger.info("Loaded scraped supplement: %s", path.name)
    return scraped


def load_all_knowledge(knowledge_dir: Path) -> dict[str, Any]:
    """
    Load all knowledge files and return a merged dictionary.

    Parameters
    ----------
    knowledge_dir : Path
        Path to the directory containing JSON knowledge files.

    Returns
    -------
    dict[str, Any]
        A dictionary with one key per knowledge section.  Missing files
        result in that key being absent from the dict (never None).

    Example
    -------
        knowledge = load_all_knowledge(Path("knowledge"))
        # knowledge["profile"]["name"] == "Ankipalli Karthik"
        # knowledge["skills"]["programming_languages"] == ["Python", ...]
    """
    if not knowledge_dir.exists():
        logger.error(
            "Knowledge directory does not exist: %s — "
            "the assistant will have no personal information.",
            knowledge_dir,
        )
        return {}

    knowledge: dict[str, Any] = {}

    for filename in KNOWLEDGE_FILES:
        section_name = filename.replace(".json", "")
        data = _load_json(knowledge_dir / filename)
        if data is not None:
            knowledge[section_name] = data

    # Merge scraped supplements under a dedicated key
    scraped = _load_scraped_supplements(knowledge_dir)
    if scraped:
        knowledge["scraped"] = scraped

    loaded = list(knowledge.keys())
    logger.info(
        "Knowledge loaded successfully. Sections: %s",
        ", ".join(loaded) if loaded else "(none)",
    )
    return knowledge


def format_skills(skills: dict[str, Any]) -> str:
    """
    Format the skills dictionary into a human-readable string for
    use in the system prompt.
    """
    lines: list[str] = []
    for category, items in skills.items():
        if isinstance(items, list) and items:
            lines.append(f"  {category.replace('_', ' ').title()}: {', '.join(items)}")
    return "\n".join(lines)


def format_projects(projects_data: dict[str, Any]) -> str:
    """
    Format the projects list into a human-readable string for the prompt.
    """
    projects = projects_data.get("projects", [])
    lines: list[str] = []
    for proj in projects:
        name = proj.get("name", "Unnamed Project")
        desc = proj.get("description", "")
        tech = ", ".join(proj.get("technologies", []))
        status = proj.get("status", "")
        url = proj.get("url", "")
        highlights = proj.get("highlights", [])

        lines.append(f"  - {name} [{status}]")
        if desc:
            lines.append(f"    Description: {desc}")
        if tech:
            lines.append(f"    Technologies: {tech}")
        if highlights:
            for h in highlights:
                lines.append(f"    • {h}")
        if url:
            lines.append(f"    GitHub: {url}")
    return "\n".join(lines)


def format_education(education_data: dict[str, Any]) -> str:
    """Format education entries into a readable string."""
    entries = education_data.get("education", [])
    lines: list[str] = []
    for edu in entries:
        degree = edu.get("degree", "")
        field = edu.get("field", "")
        specialization = edu.get("specialization", "")
        institution = edu.get("institution", "")
        duration = edu.get("duration", "")
        status = edu.get("status", "")
        highlights = edu.get("highlights", [])

        # Use specialization if available, otherwise fall back to field
        subject = specialization if specialization else field
        lines.append(f"  - {degree} in {subject}")
        lines.append(f"    Institution: {institution} | {duration} ({status})")
        if highlights:
            for h in highlights:
                lines.append(f"    • {h}")
    return "\n".join(lines)


def format_experience(experience_data: dict[str, Any]) -> str:
    """Format work experience entries into a readable string."""
    entries = experience_data.get("experience", [])
    lines: list[str] = []
    for exp in entries:
        title = exp.get("title", "")
        company = exp.get("company", "")
        duration = exp.get("duration", "")
        period = exp.get("period", "")
        summary = exp.get("summary", "")
        responsibilities = exp.get("responsibilities", [])
        technologies = exp.get("technologies", [])

        lines.append(f"  - {title} at {company} ({duration}, {period})")
        if summary:
            lines.append(f"    {summary}")
        for r in responsibilities:
            lines.append(f"    • {r}")
        if technologies:
            lines.append(f"    Technologies: {', '.join(technologies)}")
    return "\n".join(lines)

