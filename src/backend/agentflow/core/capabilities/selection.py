"""Default local selectors used by the capability resolution pipeline."""
from __future__ import annotations

import re
from typing import Iterable, Protocol, Sequence, Tuple

from agentflow.core.capabilities.models import SkillCapability, ToolCapability


class SkillSelector(Protocol):
    def select(
        self,
        task: str,
        skills: Sequence[SkillCapability],
    ) -> Tuple[str, ...]: ...


class ToolRetriever(Protocol):
    def search(
        self,
        current_step: str,
        candidates: Sequence[ToolCapability],
    ) -> Tuple[ToolCapability, ...]: ...


class KeywordSkillSelector:
    def __init__(self, max_skills: int = 3) -> None:
        self.max_skills = max(1, max_skills)

    def select(
        self,
        task: str,
        skills: Sequence[SkillCapability],
    ) -> Tuple[str, ...]:
        scored = []
        for position, skill in enumerate(skills):
            score = _relevance(
                task,
                " ".join(
                    (skill.skill_id, skill.name, skill.description, *skill.examples)
                ),
            )
            if score > 0:
                scored.append((score, -position, skill.skill_id))
        scored.sort(reverse=True)
        return tuple(item[2] for item in scored[: self.max_skills])


class KeywordToolRetriever:
    def __init__(self, max_tools: int = 10) -> None:
        self.max_tools = max(1, max_tools)

    def search(
        self,
        current_step: str,
        candidates: Sequence[ToolCapability],
    ) -> Tuple[ToolCapability, ...]:
        if len(candidates) <= self.max_tools:
            return tuple(candidates)
        scored = [
            (
                _relevance(current_step, f"{item.name} {item.description}"),
                -position,
                item,
            )
            for position, item in enumerate(candidates)
        ]
        scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
        return tuple(row[2] for row in scored[: self.max_tools])


def _relevance(query: str, document: str) -> int:
    query_normalized = (query or "").strip().lower()
    document_normalized = (document or "").strip().lower()
    if not query_normalized or not document_normalized:
        return 0
    score = 8 if document_normalized in query_normalized else 0
    query_terms = _terms(query_normalized)
    document_terms = _terms(document_normalized)
    score += len(query_terms & document_terms) * 2
    for phrase in _phrases(document_normalized):
        if phrase in query_normalized:
            score += 3
    return score


def _terms(value: str) -> set[str]:
    latin = set(re.findall(r"[a-z0-9_]+", value))
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", value))
    chinese_terms = {
        chinese[index : index + 2]
        for index in range(max(0, len(chinese) - 1))
    }
    return latin | chinese_terms


def _phrases(value: str) -> Iterable[str]:
    for part in re.split(r"[,，。;；\n]", value):
        normalized = part.strip()
        if len(normalized) >= 2:
            yield normalized
