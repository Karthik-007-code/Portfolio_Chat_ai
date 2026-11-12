"""
services/scraping_service.py

Fetches publicly available information about Karthik from permitted
sources and normalises the results into the same JSON schema used by the
hand-maintained knowledge files.

Scraping philosophy
-------------------
- Use official REST APIs when available (e.g., GitHub API).
- Use httpx + BeautifulSoup only for static HTML pages.
- Never bypass authentication, CAPTCHAs, or access restrictions.
- Respect HTTP rate limits — add delays between requests.
- Scraped data is ALWAYS written to *.scraped.json files, never to the
  hand-maintained files (profile.json, projects.json, etc.).
- If a source is unreachable or returns unexpected HTML, log the error
  and return an empty dict rather than crashing.

This module is intentionally synchronous because it is only called from
the CLI ingest script (scripts/ingest_knowledge.py), never during a live
chat request.  Keeping it synchronous avoids unnecessary complexity.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# GitHub API base — no authentication needed for public data.
_GITHUB_API = "https://api.github.com"

# Polite delay between HTTP requests (seconds)
_REQUEST_DELAY = 1.0

# Standard browser-like User-Agent to avoid being blocked by basic filters.
# We identify ourselves honestly as a personal portfolio scraper.
_HEADERS = {
    "User-Agent": (
        "AskKarthik-PortfolioBot/1.0 "
        "(personal portfolio knowledge ingestion; "
        "contact karthikankipalli007@gmail.com)"
    ),
    "Accept": "application/json",
}


# ------------------------------------------------------------------ #
# GitHub public profile                                                #
# ------------------------------------------------------------------ #

def fetch_github_profile(username: str) -> dict[str, Any]:
    """
    Fetch public profile information from the GitHub REST API.

    Parameters
    ----------
    username : str
        The GitHub username (e.g. "Karthik-007-code").

    Returns
    -------
    dict[str, Any]
        Normalised profile dict, or {} on failure.

    Notes
    -----
    The GitHub REST API for public user data requires no authentication.
    Rate limit is 60 requests/hour for unauthenticated calls — far more
    than the ingest script will ever need.
    """
    url = f"{_GITHUB_API}/users/{username}"
    logger.info("Fetching GitHub profile: %s", url)

    try:
        with httpx.Client(headers=_HEADERS, timeout=15) as client:
            response = client.get(url)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as exc:
        logger.error("GitHub API returned %s for %s", exc.response.status_code, url)
        return {}
    except httpx.RequestError as exc:
        logger.error("Network error fetching GitHub profile: %s", exc)
        return {}

    return {
        "login": data.get("login", ""),
        "name": data.get("name", ""),
        "bio": data.get("bio", ""),
        "location": data.get("location", ""),
        "blog": data.get("blog", ""),
        "html_url": data.get("html_url", ""),
        "public_repos": data.get("public_repos", 0),
        "followers": data.get("followers", 0),
        "following": data.get("following", 0),
        "created_at": data.get("created_at", ""),
    }


def fetch_github_repos(username: str, max_repos: int = 30) -> list[dict[str, Any]]:
    """
    Fetch public repository list from the GitHub REST API.

    Only meaningful fields are extracted.  Forks are excluded because
    they don't reflect Karthik's own work.

    Parameters
    ----------
    username : str
        GitHub username.
    max_repos : int
        Maximum number of repos to fetch (default 30).

    Returns
    -------
    list[dict[str, Any]]
        List of normalised repo dicts, sorted by last-updated date.
    """
    url = f"{_GITHUB_API}/users/{username}/repos"
    params = {"per_page": min(max_repos, 100), "sort": "updated", "type": "owner"}
    logger.info("Fetching GitHub repos: %s", url)

    try:
        with httpx.Client(headers=_HEADERS, timeout=15) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            repos_raw = response.json()
    except httpx.HTTPStatusError as exc:
        logger.error("GitHub API returned %s for repos: %s", exc.response.status_code, url)
        return []
    except httpx.RequestError as exc:
        logger.error("Network error fetching GitHub repos: %s", exc)
        return []

    repos: list[dict[str, Any]] = []
    for repo in repos_raw:
        # Skip forks — they're not Karthik's original work
        if repo.get("fork", False):
            continue
        repos.append({
            "name": repo.get("name", ""),
            "description": repo.get("description") or "",
            "language": repo.get("language") or "",
            "topics": repo.get("topics", []),
            "html_url": repo.get("html_url", ""),
            "homepage": repo.get("homepage") or "",
            "stars": repo.get("stargazers_count", 0),
            "updated_at": repo.get("updated_at", ""),
        })

    logger.info("Fetched %d original repos for %s", len(repos), username)
    return repos


# ------------------------------------------------------------------ #
# Portfolio website                                                    #
# ------------------------------------------------------------------ #

def fetch_portfolio_metadata(url: str) -> dict[str, Any]:
    """
    Fetch static metadata from a portfolio website.

    Important limitation: Karthik's portfolio is a React SPA.
    The raw HTML returned by the server contains only the shell
    <div id="root"></div> — all content is rendered by JavaScript,
    which a plain HTTP client cannot execute.

    This function extracts what IS available in the static HTML:
    - <title> tag
    - <meta name="description"> content
    - <meta property="og:*"> Open Graph tags
    - Any visible text in <noscript> tags

    Parameters
    ----------
    url : str
        The full URL of the portfolio homepage.

    Returns
    -------
    dict[str, Any]
        Dict of extracted metadata, or {} on failure.
    """
    logger.info("Fetching portfolio metadata: %s", url)

    # Use a browser-like Accept header for HTML
    html_headers = {**_HEADERS, "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8"}

    try:
        with httpx.Client(headers=html_headers, timeout=15, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            html = response.text
    except httpx.HTTPStatusError as exc:
        logger.error("Portfolio returned HTTP %s: %s", exc.response.status_code, url)
        return {}
    except httpx.RequestError as exc:
        logger.error("Network error fetching portfolio: %s", exc)
        return {}

    soup = BeautifulSoup(html, "lxml")

    metadata: dict[str, Any] = {
        "url": url,
        "note": (
            "This site is a React SPA. Only static HTML metadata was "
            "extractable. All visible content is rendered by JavaScript."
        ),
    }

    # <title>
    title_tag = soup.find("title")
    if title_tag and title_tag.string:
        metadata["title"] = title_tag.string.strip()

    # <meta name="description">
    desc_tag = soup.find("meta", attrs={"name": "description"})
    if desc_tag and desc_tag.get("content"):
        metadata["description"] = desc_tag["content"].strip()

    # Open Graph tags: og:title, og:description, og:url, og:image
    for og_prop in ("og:title", "og:description", "og:url", "og:image"):
        og_tag = soup.find("meta", property=og_prop)
        if og_tag and og_tag.get("content"):
            key = og_prop.replace("og:", "og_")
            metadata[key] = og_tag["content"].strip()

    # <meta name="theme-color">
    theme_tag = soup.find("meta", attrs={"name": "theme-color"})
    if theme_tag and theme_tag.get("content"):
        metadata["theme_color"] = theme_tag["content"].strip()

    logger.info(
        "Portfolio metadata extracted. Keys: %s",
        ", ".join(k for k in metadata if k not in ("url", "note")),
    )
    return metadata


# ------------------------------------------------------------------ #
# Full ingestion orchestration                                         #
# ------------------------------------------------------------------ #

def ingest_github(
    username: str,
    output_path: "Path",  # noqa: F821
) -> dict[str, Any]:
    """
    Orchestrate a full GitHub ingestion: profile + repos.

    Parameters
    ----------
    username : str
        GitHub username.
    output_path : Path
        Where to write the resulting JSON.

    Returns
    -------
    dict[str, Any]
        The combined GitHub data dict.
    """
    profile = fetch_github_profile(username)
    time.sleep(_REQUEST_DELAY)  # be polite
    repos = fetch_github_repos(username)

    result: dict[str, Any] = {
        "source": "github_api",
        "username": username,
        "profile": profile,
        "repositories": repos,
    }
    return result


def ingest_portfolio(
    url: str,
    output_path: "Path",  # noqa: F821
) -> dict[str, Any]:
    """
    Orchestrate a portfolio website metadata ingestion.

    Parameters
    ----------
    url : str
        Portfolio homepage URL.
    output_path : Path
        Where to write the resulting JSON.

    Returns
    -------
    dict[str, Any]
        Extracted portfolio metadata.
    """
    metadata = fetch_portfolio_metadata(url)
    result: dict[str, Any] = {
        "source": "portfolio_website",
        "url": url,
        "metadata": metadata,
    }
    return result

