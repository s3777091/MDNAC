"""Clarification logic shared by the protein-span LangGraph agent and the API.

The clarification step is the human-in-the-loop entry point: it decides whether a
user goal is specific enough to drive a protein database search and, when it is
not, proposes a sharper query grounded in the Exa public-research evidence.
"""

from __future__ import annotations

import json
from typing import Any

from ai_agent.config.settings import AISettings
from ai_agent.models.base import ChatModel
from ai_agent.skills import render_selected_skills, select_agent_skills


def clarify_protein_request(
    model: ChatModel,
    settings: AISettings,
    user_input: str,
    context: str,
    evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Ask the model whether ``user_input`` is specific enough for a protein search.

    Returns a dict with keys ``needs_clarification``, ``message``,
    ``proposed_query`` and ``research_query``. Suggestions are grounded in the
    supplied Exa ``evidence`` so the human is asked an informed question.
    """
    selected_skills = select_agent_skills(
        user_input=user_input,
        context=context,
        search_needed=False,
    )
    skills_text = render_selected_skills(selected_skills)
    evidence_text = summarize_evidence(evidence or [])
    prompt = (
        "You are preparing a precise protein database semantic search for MDNAC.\n"
        "Decide whether the user's request is specific enough to search protein records.\n"
        "If the request is vague, propose a better search query and ask for approval.\n"
        "Use the Exa public research evidence below to ground your suggestions in the "
        "actual domain (mechanisms, organisms, proteins) rather than guessing.\n"
        "For example, if the user only says they want to increase crop yield, explain that "
        "the goal needs a mechanism/crop/organism and propose a protein-search query around "
        "plant growth-promoting proteins such as nitrogen fixation, phosphate solubilization, "
        "auxin biosynthesis, ACC deaminase, stress tolerance, or biocontrol.\n\n"
        "Return JSON only with keys: needs_clarification, message, proposed_query, "
        "research_query.\n"
        "`proposed_query` MUST be a concise keyword query for an NCBI protein "
        "sequence search: a few terms only (protein/gene name(s) plus organism "
        "genus if known, e.g. 'nitrogenase nifH Azospirillum'). Do NOT write a full "
        "sentence and do NOT use brackets or placeholders.\n\n"
        "Selected Skills:\n"
        f"{skills_text}\n\n"
        "Exa public research evidence:\n"
        f"{evidence_text or 'No public research evidence available.'}\n\n"
        f"User input: {user_input}\n\n"
        f"Context: {context}"
    )
    messages = [
        {"role": "system", "content": settings.system_prompt},
        {"role": "user", "content": prompt},
    ]
    try:
        raw = model.generate(messages, temperature=0.0)
        parsed = parse_json_object(raw)
        return {
            "needs_clarification": bool(parsed.get("needs_clarification")),
            "message": str(parsed.get("message") or ""),
            "proposed_query": str(parsed.get("proposed_query") or user_input),
            "research_query": str(
                parsed.get("research_query") or parsed.get("proposed_query") or user_input
            ),
        }
    except Exception:
        return heuristic_clarification(user_input, evidence or [])


def heuristic_clarification(
    user_input: str,
    evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Deterministic fallback when the model is unavailable or returns bad JSON."""
    lowered = user_input.lower()
    vague_crop_request = any(term in lowered for term in ("crop", "cay", "nang suat", "yield"))
    has_mechanism = any(
        term in lowered
        for term in (
            "nitrogen",
            "phosphate",
            "auxin",
            "iaa",
            "acc",
            "deaminase",
            "stress",
            "drought",
            "salinity",
            "biocontrol",
            "protein",
        )
    )
    if vague_crop_request and not has_mechanism:
        proposed = (
            "plant growth-promoting protein for crop yield improvement nitrogen fixation "
            "phosphate solubilization auxin biosynthesis ACC deaminase stress tolerance"
        )
        message = (
            "Cau hoi cua ban chua ro co che/cay trong/vi sinh vat muc tieu. "
            "Minh co the chinh lai thanh truy van protein lien quan den tang "
            "nang suat cay trong qua co dinh dam, hoa tan phosphate, auxin/IAA, "
            "ACC deaminase va chong stress. Neu dong y, gui action=approve."
        )
        evidence_text = summarize_evidence(evidence or [])
        if evidence_text:
            message = f"{message}\n\nGoi y tu Exa public research:\n{evidence_text}"
        return {
            "needs_clarification": True,
            "message": message,
            "proposed_query": proposed,
            "research_query": proposed,
        }
    return {
        "needs_clarification": False,
        "message": "Request is specific enough for protein search.",
        "proposed_query": user_input,
        "research_query": user_input,
    }


def summarize_evidence(evidence: list[dict[str, Any]], *, limit: int = 5) -> str:
    """Render Exa results as a short bullet list for prompting and user messages."""
    lines: list[str] = []
    for item in evidence[:limit]:
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        if not title and not url:
            continue
        lines.append(f"- {title or url} ({url})" if url else f"- {title}")
    return "\n".join(lines)


def parse_json_object(value: str) -> dict[str, Any]:
    """Extract the first JSON object from a model response (tolerates code fences)."""
    text = str(value or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Model did not return a JSON object.")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Model JSON response must be an object.")
    return parsed


__all__ = [
    "clarify_protein_request",
    "heuristic_clarification",
    "parse_json_object",
    "summarize_evidence",
]
