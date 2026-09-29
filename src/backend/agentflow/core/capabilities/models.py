"""Domain models for the layered capability registry.

Skills describe workflows, tools perform side effects, and agents define the
maximum capability boundary.  Keeping these models separate prevents a skill
declaration from being mistaken for an execution grant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Tuple


class ToolExecutorType(str, Enum):
    OPENAPI = "openapi"
    MCP = "mcp"
    BUILTIN = "builtin"
    DELEGATED_AGENT = "delegated_agent"


class ToolRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class ToolCapability:
    """An executable tool plus the metadata needed for policy decisions."""

    tool_id: str
    name: str
    description: str
    executor_type: ToolExecutorType
    tool: Any = field(compare=False, repr=False)
    auth_scopes: Tuple[str, ...] = ()
    read_only: bool = False
    idempotent: bool = False
    destructive: bool = False
    timeout_seconds: float = 30.0
    risk_level: ToolRiskLevel = ToolRiskLevel.MEDIUM
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({}),
        compare=False,
    )

    def __post_init__(self) -> None:
        if not self.tool_id.strip():
            raise ValueError("tool_id must not be empty")
        if not self.name.strip():
            raise ValueError("tool name must not be empty")
        object.__setattr__(self, "auth_scopes", tuple(self.auth_scopes))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class SkillCapability:
    """A workflow/knowledge package.  Tool names are dependencies, not grants."""

    skill_id: str
    name: str
    description: str
    instructions_uri: str
    instructions: str = field(repr=False)
    required_tools: Tuple[str, ...] = ()
    optional_tools: Tuple[str, ...] = ()
    examples: Tuple[str, ...] = ()
    version: str = "1"

    def __post_init__(self) -> None:
        if not self.skill_id.strip():
            raise ValueError("skill_id must not be empty")
        if not self.name.strip():
            raise ValueError("skill name must not be empty")
        object.__setattr__(self, "required_tools", _unique(self.required_tools))
        object.__setattr__(self, "optional_tools", _unique(self.optional_tools))
        object.__setattr__(self, "examples", _unique(self.examples))
        object.__setattr__(self, "version", str(self.version or "1"))

    @property
    def tool_dependencies(self) -> Tuple[str, ...]:
        return _unique((*self.required_tools, *self.optional_tools))


@dataclass(frozen=True)
class AgentCapability:
    """Agent prompt, visible skills, delegates, and maximum tool boundary."""

    agent_id: str
    prompt: str
    available_skills: Tuple[str, ...] = ()
    available_sub_agents: Tuple[str, ...] = ()
    max_tool_permissions: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.agent_id.strip():
            raise ValueError("agent_id must not be empty")
        object.__setattr__(self, "available_skills", _unique(self.available_skills))
        object.__setattr__(
            self,
            "available_sub_agents",
            _unique(self.available_sub_agents),
        )
        object.__setattr__(self, "max_tool_permissions", _unique(self.max_tool_permissions))


def _unique(values) -> Tuple[str, ...]:
    seen = set()
    result = []
    for value in values or ():
        normalized = str(value).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return tuple(result)
