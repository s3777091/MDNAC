from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_SKILLS_DIR = Path(__file__).resolve().parent
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")
DEFAULT_SKILL_NAME = "grounded-answer"


@dataclass(frozen=True)
class AgentSkill:
    name: str
    description: str
    body: str
    path: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "description": self.description,
            "body": self.body,
            "path": str(self.path),
        }


def load_agent_skills(skills_dir: str | Path | None = None) -> list[AgentSkill]:
    root = Path(skills_dir or DEFAULT_SKILLS_DIR)
    if not root.is_dir():
        return []

    skills: list[AgentSkill] = []
    for skill_path in sorted(root.glob("*/SKILL.md")):
        skill = _read_skill(skill_path)
        if skill is not None:
            skills.append(skill)
    return skills


def select_agent_skills(
    *,
    user_input: str,
    context: str,
    skills: Iterable[AgentSkill] | None = None,
    search_needed: bool = False,
    max_skills: int = 4,
) -> list[dict[str, str]]:
    available = list(skills if skills is not None else load_agent_skills())
    if not available:
        return []

    selected: list[AgentSkill] = []
    by_name = {skill.name: skill for skill in available}
    default_skill = by_name.get(DEFAULT_SKILL_NAME)
    if default_skill is not None:
        selected.append(default_skill)

    request_terms = set(_tokens(f"{user_input} {context}"))
    for skill in available:
        if skill in selected:
            continue
        if skill.name == "public-research" and search_needed:
            selected.append(skill)
            continue
        score = _skill_score(skill, request_terms)
        if score >= 2:
            selected.append(skill)

    selected = _dedupe(selected)
    return [skill.to_dict() for skill in selected[:max_skills]]


def render_selected_skills(selected_skills: Iterable[dict[str, str]]) -> str:
    rendered: list[str] = []
    for skill in selected_skills:
        name = str(skill.get("name") or "").strip()
        description = str(skill.get("description") or "").strip()
        body = str(skill.get("body") or "").strip()
        if not name:
            continue
        rendered.append(
            f"## {name}\n"
            f"Description: {description}\n\n"
            f"{body}"
        )
    return "\n\n".join(rendered)


def _read_skill(path: Path) -> AgentSkill | None:
    text = path.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(text)
    name = frontmatter.get("name") or path.parent.name
    description = frontmatter.get("description") or ""
    name = name.strip()
    if not name:
        return None
    return AgentSkill(
        name=name,
        description=description.strip(),
        body=body.strip(),
        path=path,
    )


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    normalized = text.replace("\r\n", "\n")
    if not normalized.startswith("---\n"):
        return {}, normalized

    end = normalized.find("\n---\n", 4)
    if end == -1:
        return {}, normalized

    raw_frontmatter = normalized[4:end]
    body = normalized[end + len("\n---\n") :]
    parsed: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []
    for line in raw_frontmatter.splitlines():
        if not line.strip():
            continue
        if line.startswith(" ") and current_key is not None:
            current_lines.append(line.strip())
            continue
        if current_key is not None:
            parsed[current_key] = "\n".join(current_lines).strip()
            current_key = None
            current_lines = []
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "|":
            current_key = key
            current_lines = []
        else:
            parsed[key] = value.strip("'\"")
    if current_key is not None:
        parsed[current_key] = "\n".join(current_lines).strip()
    return parsed, body


def _skill_score(skill: AgentSkill, request_terms: set[str]) -> int:
    skill_terms = set(_tokens(f"{skill.name} {skill.description} {skill.body[:800]}"))
    return len(request_terms & skill_terms)


def _tokens(value: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(value or "")]


def _dedupe(skills: Iterable[AgentSkill]) -> list[AgentSkill]:
    seen: set[str] = set()
    deduped: list[AgentSkill] = []
    for skill in skills:
        if skill.name in seen:
            continue
        seen.add(skill.name)
        deduped.append(skill)
    return deduped
