"""Capability resolution and the final permission intersection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol, Sequence, Tuple

from agentflow.core.capabilities.models import SkillCapability, ToolCapability
from agentflow.core.capabilities.registry import CapabilityRegistry
from agentflow.core.capabilities.selection import KeywordToolRetriever, ToolRetriever
from agentflow.core.capabilities.skill_document import SkillLoader


class ApplicationPolicy(Protocol):
    def allows(self, tool: ToolCapability, context: Mapping[str, Any]) -> bool: ...


class AllowAllApplicationPolicy:
    def allows(self, tool: ToolCapability, context: Mapping[str, Any]) -> bool:
        return True


class BlockListApplicationPolicy:
    def __init__(self, blocked_tools: Iterable[str] = ()) -> None:
        self._blocked = frozenset(str(item) for item in blocked_tools)

    def allows(self, tool: ToolCapability, context: Mapping[str, Any]) -> bool:
        return tool.tool_id not in self._blocked and tool.name not in self._blocked


@dataclass(frozen=True)
class CapabilityResolution:
    selected_skills: Tuple[SkillCapability, ...]
    instructions: Tuple[str, ...]
    allowed_tools: Tuple[ToolCapability, ...]
    candidate_tools: Tuple[ToolCapability, ...]
    selected_tools: Tuple[ToolCapability, ...]
    missing_required_tools: Tuple[str, ...]

    @property
    def selected_tool_objects(self) -> Tuple[Any, ...]:
        return tuple(item.tool for item in self.selected_tools)


class CapabilityResolver:
    """Resolve skills and tools without ever turning dependencies into grants.

    Final candidate permissions are:

        skill dependencies
        & agent max tool permissions
        & user permissions
        & application policy

    If no skill is selected, the skill-dependency term is intentionally omitted
    so ordinary tool-only tasks continue to work.
    """

    def __init__(
        self,
        registry: CapabilityRegistry,
        *,
        policy: ApplicationPolicy | None = None,
        tool_retriever: ToolRetriever | None = None,
    ) -> None:
        self.registry = registry
        self.policy = policy or AllowAllApplicationPolicy()
        self.tool_retriever = tool_retriever or KeywordToolRetriever()
        self.skill_loader = SkillLoader(registry.skills)

    def resolve(
        self,
        *,
        agent_id: str,
        selected_skill_ids: Sequence[str],
        user_permissions: Iterable[str],
        current_step: str,
        context: Mapping[str, Any] | None = None,
    ) -> CapabilityResolution:
        agent = self.registry.agents.get(agent_id)
        selected_skills = self._resolve_agent_skills(
            agent.available_skills,
            selected_skill_ids,
        )
        agent_permissions = set(agent.max_tool_permissions)
        user_permissions = set(str(value) for value in user_permissions)
        policy_context = context or {}

        allowed_tools = tuple(
            item
            for item in self.registry.tools.list()
            if _matches_permission(item, agent_permissions)
            and _matches_permission(item, user_permissions)
            and self.policy.allows(item, policy_context)
        )
        allowed_names = _tool_aliases(allowed_tools)

        if selected_skills:
            dependencies = {
                dependency
                for skill in selected_skills
                for dependency in skill.tool_dependencies
            }
            candidate_tools = tuple(
                item
                for item in allowed_tools
                if item.tool_id in dependencies or item.name in dependencies
            )
        else:
            candidate_tools = allowed_tools

        required = {
            dependency
            for skill in selected_skills
            for dependency in skill.required_tools
        }
        missing_required = tuple(sorted(required - allowed_names))
        # Treat retrieval as ranking only, never as authorization.  Even a
        # buggy/custom retriever cannot inject a tool outside the intersection.
        candidates_by_id = {item.tool_id: item for item in candidate_tools}
        retrieved = self.tool_retriever.search(current_step, candidate_tools)
        selected = []
        selected_ids = set()
        for item in retrieved:
            if item.tool_id not in candidates_by_id or item.tool_id in selected_ids:
                continue
            selected_ids.add(item.tool_id)
            selected.append(candidates_by_id[item.tool_id])
        selected_tools = tuple(selected)
        return CapabilityResolution(
            selected_skills=selected_skills,
            instructions=self.skill_loader.load(
                skill.skill_id for skill in selected_skills
            ),
            allowed_tools=allowed_tools,
            candidate_tools=candidate_tools,
            selected_tools=selected_tools,
            missing_required_tools=missing_required,
        )

    def _resolve_agent_skills(
        self,
        available_skill_ids: Sequence[str],
        selected_skill_ids: Sequence[str],
    ) -> Tuple[SkillCapability, ...]:
        available = set(available_skill_ids)
        result = []
        seen = set()
        for skill_id in selected_skill_ids:
            if skill_id in seen or skill_id not in available:
                continue
            skill = self.registry.skills.find(skill_id)
            if skill is not None:
                seen.add(skill_id)
                result.append(skill)
        return tuple(result)


def _matches_permission(tool: ToolCapability, permissions: set[str]) -> bool:
    return "*" in permissions or tool.tool_id in permissions or tool.name in permissions


def is_tool_authorized(
    tool_name: str,
    authorized_tool_names: Iterable[str] | None,
) -> bool:
    """Fail-closed call-time guard; missing and empty allowlists deny all."""
    if authorized_tool_names is None:
        return False
    return tool_name in set(authorized_tool_names)


def _tool_aliases(tools: Sequence[ToolCapability]) -> set[str]:
    return {
        alias
        for item in tools
        for alias in (item.tool_id, item.name)
    }
