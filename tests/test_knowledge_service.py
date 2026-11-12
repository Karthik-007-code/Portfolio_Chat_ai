"""
tests/test_knowledge_service.py

Unit tests for knowledge loading and prompt formatters.
"""

from pathlib import Path
from portfolio_chat_ai.services.knowledge_service import (
    load_all_knowledge,
    format_skills,
    format_projects,
    format_education,
    format_experience,
)


def test_load_all_knowledge_loads_expected_keys():
    knowledge_dir = Path("knowledge")
    knowledge = load_all_knowledge(knowledge_dir)

    expected_keys = {"profile", "education", "skills", "experience", "projects", "achievements"}
    for key in expected_keys:
        assert key in knowledge, f"Expected key '{key}' in knowledge base"

    assert knowledge["profile"]["preferred_name"] == "Karthik"
    assert "ThoughtWorks" in str(knowledge["experience"])


def test_format_skills():
    mock_skills = {
        "programming_languages": ["Python", "JavaScript"],
        "backend": ["FastAPI", "SQLite"],
    }
    formatted = format_skills(mock_skills)
    assert "Programming Languages: Python, JavaScript" in formatted
    assert "Backend: FastAPI, SQLite" in formatted


def test_format_education():
    mock_edu = {
        "education": [
            {
                "degree": "B.Tech",
                "field": "CSE",
                "specialization": "AI",
                "institution": "KL University",
                "duration": "2026 – Present",
                "status": "Ongoing",
                "highlights": ["Autonomous Agents"],
            }
        ]
    }
    formatted = format_education(mock_edu)
    assert "B.Tech in AI" in formatted
    assert "KL University" in formatted
    assert "Autonomous Agents" in formatted


def test_format_experience():
    mock_exp = {
        "experience": [
            {
                "title": "Software Engineering Intern",
                "company": "ThoughtWorks",
                "duration": "6 months",
                "period": "2025-2026",
                "responsibilities": ["Full-stack MERN"],
                "technologies": ["MongoDB", "Express.js", "React", "Node.js"],
            }
        ]
    }
    formatted = format_experience(mock_exp)
    assert "ThoughtWorks" in formatted
    assert "Full-stack MERN" in formatted
    assert "MongoDB, Express.js" in formatted

