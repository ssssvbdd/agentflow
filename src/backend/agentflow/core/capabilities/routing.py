"""Cascaded intent routing: deterministic fast paths before an LLM router."""
from __future__ import annotations

import asyncio
import hashlib
import math
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, Tuple

from agentflow.core.capabilities.intent import IntentDecision
from agentflow.core.capabilities.models import SkillCapability, ToolCapability


class SemanticScorer(Protocol):
    async def score(
        self,
        query: str,
        documents: Sequence[str],
    ) -> Sequence[float]: ...


class EmbeddingSemanticScorer:
    """Embedding scorer with a bounded process cache for capability metadata."""

    _cache: "OrderedDict[str, Tuple[float, ...]]" = OrderedDict()
    _cache_lock = threading.Lock()

    def __init__(self, embedding_model: Any, *, max_cache_entries: int = 4096) -> None:
        self.embedding_model = embedding_model
        self.max_cache_entries = max(128, max_cache_entries)
        self.model_key = "|".join(
            str(value or "")
            for value in (
                getattr(embedding_model, "model", ""),
                getattr(embedding_model, "base_url", ""),
            )
        )

    async def score(
        self,
        query: str,
        documents: Sequence[str],
    ) -> Sequence[float]:
        texts = [query, *documents]
        vectors: list[Tuple[float, ...] | None] = [None] * len(texts)
        missing_indexes: list[int] = []
        missing_texts: list[str] = []
        with self._cache_lock:
            for index, text in enumerate(texts):
                key = self._key(text)
                cached = self._cache.get(key)
                if cached is None:
                    missing_indexes.append(index)
                    missing_texts.append(text)
                else:
                    self._cache.move_to_end(key)
                    vectors[index] = cached

        if missing_texts:
            embedded = await self.embedding_model.embed_async(missing_texts)
            if missing_texts and len(missing_texts) == 1 and embedded:
                if isinstance(embedded[0], (float, int)):
                    embedded = [embedded]
            with self._cache_lock:
                for index, text, vector in zip(
                    missing_indexes,
                    missing_texts,
                    embedded,
                ):
                    normalized = _normalize_vector(vector)
                    vectors[index] = normalized
                    self._cache[self._key(text)] = normalized
                    self._cache.move_to_end(self._key(text))
                while len(self._cache) > self.max_cache_entries:
                    self._cache.popitem(last=False)

        query_vector = vectors[0] or ()
        return tuple(
            max(0.0, min(1.0, _cosine(query_vector, vector or ())))
            for vector in vectors[1:]
        )

    def _key(self, text: str) -> str:
        payload = f"{self.model_key}\0{text}".encode("utf-8", errors="ignore")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class RankedRouteCandidate:
    kind: str
    capability_id: str
    score: float
    lexical_score: float
    semantic_score: float
    value: Any


@dataclass(frozen=True)
class IntentGateResult:
    use_model: bool
    decision: IntentDecision | None
    skills: Tuple[SkillCapability, ...]
    agents: Tuple[Mapping[str, Any], ...]
    tools: Tuple[ToolCapability, ...]
    reason: str
    top_score: float = 0.0
    score_margin: float = 0.0
    semantic_used: bool = False


class IntentRoutingGate:
    """Latency-aware route gate with optional semantic reranking."""

    def __init__(
        self,
        *,
        semantic_scorer: SemanticScorer | None = None,
        small_tool_catalog_size: int = 8,
        deterministic_score_threshold: float = 0.82,
        score_margin_threshold: float = 0.20,
        semantic_no_match_threshold: float = 0.45,
        semantic_timeout_seconds: float = 3.0,
        max_candidates: int = 12,
    ) -> None:
        self.semantic_scorer = semantic_scorer
        self.small_tool_catalog_size = max(0, small_tool_catalog_size)
        self.deterministic_score_threshold = deterministic_score_threshold
        self.score_margin_threshold = score_margin_threshold
        self.semantic_no_match_threshold = semantic_no_match_threshold
        self.semantic_timeout_seconds = max(0.1, semantic_timeout_seconds)
        self.max_candidates = max(3, max_candidates)

    async def evaluate(
        self,
        *,
        messages: Sequence[Any],
        current_task: str,
        skills: Sequence[SkillCapability],
        agents: Sequence[Mapping[str, Any]],
        tools: Sequence[ToolCapability],
    ) -> IntentGateResult:
        task = (current_task or "").strip()
        if _is_smalltalk(task):
            return self._fast_decision(
                task,
                "direct_answer",
                skills,
                agents,
                tools,
                "conservative small-talk fast path",
            )

        contextual = _needs_context_resolution(messages, task)
        complex_task = _looks_multi_step(task)
        retrieval_query = (
            _contextual_retrieval_query(messages, task) if contextual else task
        )
        ranked = _rank_lexically(retrieval_query, skills, agents, tools)
        top_score, margin = _top_score_and_margin(ranked)

        if not contextual and not complex_task:
            deterministic = _deterministic_decision(
                task,
                ranked,
                threshold=self.deterministic_score_threshold,
                margin_threshold=self.score_margin_threshold,
            )
            if deterministic is not None:
                return self._result(
                    use_model=False,
                    decision=deterministic,
                    ranked=ranked,
                    skills=skills,
                    agents=agents,
                    tools=tools,
                    reason="high-score deterministic capability match",
                    top_score=top_score,
                    margin=margin,
                )

        # A small atomic-tool catalog is cheaper and usually more accurate when
        # handled directly by the main model's native function calling.
        if (
            not contextual
            and not complex_task
            and not skills
            and not agents
            and len(tools) <= self.small_tool_catalog_size
        ):
            return self._fast_decision(
                task,
                "react",
                skills,
                agents,
                tools,
                "small tool catalog fast path",
            )

        # General knowledge questions with no lexical capability signal avoid
        # a redundant router call. The main model receives no tools.
        if (
            not contextual
            and not complex_task
            and top_score == 0.0
            and _looks_general_question(task)
        ):
            return self._fast_decision(
                task,
                "direct_answer",
                skills,
                agents,
                tools,
                "general-question fast path",
            )

        semantic_used = False
        if self.semantic_scorer is not None and ranked:
            try:
                documents = [_candidate_document(item) for item in ranked]
                semantic_scores = await asyncio.wait_for(
                    self.semantic_scorer.score(retrieval_query, documents),
                    timeout=self.semantic_timeout_seconds,
                )
                ranked = _fuse_semantic_scores(ranked, semantic_scores)
                semantic_used = True
                top_score, margin = _top_score_and_margin(ranked)
                if not contextual and not complex_task:
                    deterministic = _deterministic_decision(
                        task,
                        ranked,
                        threshold=self.deterministic_score_threshold,
                        margin_threshold=self.score_margin_threshold,
                    )
                    if deterministic is not None:
                        return self._result(
                            use_model=False,
                            decision=deterministic,
                            ranked=ranked,
                            skills=skills,
                            agents=agents,
                            tools=tools,
                            reason="hybrid retrieval deterministic match",
                            top_score=top_score,
                            margin=margin,
                            semantic_used=True,
                        )
                    if top_score < self.semantic_no_match_threshold:
                        return self._fast_decision(
                            task,
                            "direct_answer",
                            skills,
                            agents,
                            tools,
                            "hybrid retrieval found no relevant capability",
                            semantic_used=True,
                        )
            except Exception:
                # Semantic retrieval is an optimization, never an availability
                # dependency. The structured router remains the safe fallback.
                semantic_used = False

        shortlisted = _shortlist(ranked, self.max_candidates)
        return IntentGateResult(
            use_model=True,
            decision=None,
            skills=tuple(
                item.value for item in shortlisted if item.kind == "skill"
            ),
            agents=tuple(
                item.value for item in shortlisted if item.kind == "agent"
            ),
            tools=tuple(
                item.value for item in shortlisted if item.kind == "tool"
            ),
            reason=(
                "context-dependent request"
                if contextual
                else "multi-step request"
                if complex_task
                else "ambiguous capability candidates"
            ),
            top_score=top_score,
            score_margin=margin,
            semantic_used=semantic_used,
        )

    def _fast_decision(
        self,
        task: str,
        route: str,
        skills: Sequence[SkillCapability],
        agents: Sequence[Mapping[str, Any]],
        tools: Sequence[ToolCapability],
        reason: str,
        *,
        semantic_used: bool = False,
    ) -> IntentGateResult:
        decision = IntentDecision(
            normalized_task=task,
            route=route,
            confidence=1.0,
            reason=reason,
        )
        return IntentGateResult(
            use_model=False,
            decision=decision,
            skills=tuple(skills),
            agents=tuple(agents),
            tools=tuple(tools),
            reason=reason,
            semantic_used=semantic_used,
        )

    def _result(
        self,
        *,
        use_model: bool,
        decision: IntentDecision,
        ranked: Sequence[RankedRouteCandidate],
        skills: Sequence[SkillCapability],
        agents: Sequence[Mapping[str, Any]],
        tools: Sequence[ToolCapability],
        reason: str,
        top_score: float,
        margin: float,
        semantic_used: bool = False,
    ) -> IntentGateResult:
        shortlisted = _shortlist(ranked, self.max_candidates)
        return IntentGateResult(
            use_model=use_model,
            decision=decision,
            skills=tuple(
                item.value for item in shortlisted if item.kind == "skill"
            ),
            agents=tuple(
                item.value for item in shortlisted if item.kind == "agent"
            ),
            tools=tuple(
                item.value for item in shortlisted if item.kind == "tool"
            ),
            reason=reason,
            top_score=top_score,
            score_margin=margin,
            semantic_used=semantic_used,
        )


def _rank_lexically(
    query: str,
    skills: Sequence[SkillCapability],
    agents: Sequence[Mapping[str, Any]],
    tools: Sequence[ToolCapability],
) -> list[RankedRouteCandidate]:
    candidates: list[RankedRouteCandidate] = []
    for skill in skills:
        document = " ".join(
            (skill.name, skill.description, *skill.examples[:3])
        )
        candidates.append(
            RankedRouteCandidate(
                "skill",
                skill.skill_id,
                _lexical_score(query, skill.name, document),
                _lexical_score(query, skill.name, document),
                0.0,
                skill,
            )
        )
    for agent in agents:
        agent_id = str(agent.get("agent_id") or "")
        name = str(agent.get("name") or "")
        document = f"{name} {agent.get('description') or ''}"
        score = _lexical_score(query, name, document)
        candidates.append(
            RankedRouteCandidate(
                "agent", agent_id, score, score, 0.0, agent
            )
        )
    for tool in tools:
        score = _lexical_score(
            query,
            tool.name,
            f"{tool.name} {tool.description}",
        )
        candidates.append(
            RankedRouteCandidate(
                "tool", tool.tool_id, score, score, 0.0, tool
            )
        )
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates


def _lexical_score(query: str, name: str, document: str) -> float:
    query_normalized = _normalize_text(query)
    name_normalized = _normalize_text(name)
    document_normalized = _normalize_text(document)
    if not query_normalized or not document_normalized:
        return 0.0
    query_terms = _terms(query_normalized)
    document_terms = _terms(document_normalized)
    if not query_terms or not document_terms:
        return 0.0
    overlap = len(query_terms & document_terms)
    coverage = overlap / max(1, len(query_terms))
    precision = overlap / max(1, min(len(document_terms), 8))
    score = 0.55 * coverage + 0.25 * precision
    if name_normalized and name_normalized in query_normalized:
        score += 0.45
    return max(0.0, min(1.0, score))


def _fuse_semantic_scores(
    ranked: Sequence[RankedRouteCandidate],
    semantic_scores: Sequence[float],
) -> list[RankedRouteCandidate]:
    fused = []
    for item, semantic in zip(ranked, semantic_scores):
        semantic = max(0.0, min(1.0, float(semantic)))
        score = max(item.lexical_score, 0.35 * item.lexical_score + 0.65 * semantic)
        fused.append(
            RankedRouteCandidate(
                kind=item.kind,
                capability_id=item.capability_id,
                score=score,
                lexical_score=item.lexical_score,
                semantic_score=semantic,
                value=item.value,
            )
        )
    fused.sort(key=lambda item: item.score, reverse=True)
    return fused


def _deterministic_decision(
    task: str,
    ranked: Sequence[RankedRouteCandidate],
    *,
    threshold: float,
    margin_threshold: float,
) -> IntentDecision | None:
    if not ranked:
        return None
    top = ranked[0]
    second_score = ranked[1].score if len(ranked) > 1 else 0.0
    if top.score < threshold or top.score - second_score < margin_threshold:
        return None
    if top.kind == "skill":
        return IntentDecision(
            normalized_task=task,
            route="skill",
            skill_ids=[top.capability_id],
            confidence=top.score,
            reason="确定性候选召回唯一命中",
        )
    if top.kind == "agent":
        return IntentDecision(
            normalized_task=task,
            route="delegate_agent",
            agent_id=top.capability_id,
            confidence=top.score,
            reason="确定性子 Agent 候选唯一命中",
        )
    if top.kind == "tool":
        return IntentDecision(
            normalized_task=task,
            route="react",
            confidence=top.score,
            reason="确定性工具候选命中，由主模型提取参数",
        )
    return None


def _shortlist(
    ranked: Sequence[RankedRouteCandidate],
    limit: int,
) -> Tuple[RankedRouteCandidate, ...]:
    selected: list[RankedRouteCandidate] = list(ranked[:limit])
    selected_keys = {(item.kind, item.capability_id) for item in selected}
    # Preserve at least one candidate from each domain to avoid one large tool
    # family starving Skill or Agent routing.
    for kind in ("skill", "agent", "tool"):
        candidate = next((item for item in ranked if item.kind == kind), None)
        if candidate and (candidate.kind, candidate.capability_id) not in selected_keys:
            selected.append(candidate)
            selected_keys.add((candidate.kind, candidate.capability_id))
    return tuple(selected)


def _candidate_document(item: RankedRouteCandidate) -> str:
    if item.kind == "skill":
        skill = item.value
        return " ".join((skill.name, skill.description, *skill.examples[:3]))
    if item.kind == "agent":
        return f"{item.value.get('name') or ''} {item.value.get('description') or ''}"
    return f"{item.value.name} {item.value.description}"


def _top_score_and_margin(
    ranked: Sequence[RankedRouteCandidate],
) -> Tuple[float, float]:
    if not ranked:
        return 0.0, 0.0
    top = ranked[0].score
    second = ranked[1].score if len(ranked) > 1 else 0.0
    return top, max(0.0, top - second)


def _needs_context_resolution(messages: Sequence[Any], task: str) -> bool:
    non_system_messages = [
        item
        for item in messages
        if str(getattr(item, "type", "")).lower() != "system"
    ]
    if len(non_system_messages) < 2:
        return False
    return bool(
        re.search(
            r"(刚才|上面|之前|那个|这个|它|这笔|那笔|继续|退掉|取消它|"
            r"\b(it|that|those|previous|above|continue)\b)",
            task,
            re.IGNORECASE,
        )
    )


def _contextual_retrieval_query(messages: Sequence[Any], task: str) -> str:
    parts: list[str] = []
    for message in reversed(messages):
        if str(getattr(message, "type", "")).lower() == "system":
            continue
        content = getattr(message, "content", "")
        if not isinstance(content, str):
            continue
        content = content.strip()
        if content:
            parts.append(content[-500:])
        if len(parts) >= 4:
            break
    parts.reverse()
    combined = "\n".join(parts)
    return combined[-2_000:] if combined else task


def _looks_multi_step(task: str) -> bool:
    return bool(
        re.search(
            r"(然后|接着|并且|同时|最后|先.+再|分析.+并|检查.+并|"
            r"\b(then|after that|and then|finally|first.+then)\b)",
            task,
            re.IGNORECASE,
        )
    )


def _looks_general_question(task: str) -> bool:
    return bool(
        re.match(
            r"^\s*(请)?(解释|介绍|说明|什么是|为什么|如何理解|怎么理解|"
            r"what\b|why\b|explain\b|describe\b)",
            task,
            re.IGNORECASE,
        )
    )


def _is_smalltalk(task: str) -> bool:
    normalized = re.sub(r"[\s，。！？,.!?~～]+", "", task).lower()
    return normalized in {
        "你好",
        "您好",
        "嗨",
        "hello",
        "hi",
        "谢谢",
        "感谢",
        "thanks",
        "你是谁",
        "你能做什么",
    }


def _normalize_text(value: str) -> str:
    return " ".join(str(value or "").lower().split())


def _terms(value: str) -> set[str]:
    latin = set(re.findall(r"[a-z0-9_]+", value))
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", value))
    chinese_terms = {
        chinese[index : index + 2]
        for index in range(max(0, len(chinese) - 1))
    }
    return latin | chinese_terms


def _normalize_vector(vector: Sequence[float]) -> Tuple[float, ...]:
    values = tuple(float(value) for value in vector)
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return values
    return tuple(value / norm for value in values)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))
