"""In-memory registries for skills, tools, and agents."""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Generic, Iterable, Optional, Tuple, TypeVar

from agentflow.core.capabilities.models import (
    AgentCapability,
    SkillCapability,
    ToolCapability,
)


T = TypeVar("T")


class DuplicateCapabilityError(ValueError):
    pass


class UnknownCapabilityError(KeyError):
    pass


class _Registry(Generic[T]):
    def __init__(self, id_attribute: str) -> None:
        self._id_attribute = id_attribute
        self._items: Dict[str, T] = OrderedDict()

    def register(self, capability: T) -> T:
        capability_id = str(getattr(capability, self._id_attribute))
        if capability_id in self._items:
            raise DuplicateCapabilityError(
                f"capability {capability_id!r} is already registered"
            )
        self._items[capability_id] = capability
        return capability

    def get(self, capability_id: str) -> T:
        try:
            return self._items[capability_id]
        except KeyError as err:
            raise UnknownCapabilityError(capability_id) from err

    def find(self, capability_id: str) -> Optional[T]:
        return self._items.get(capability_id)

    def list(self) -> Tuple[T, ...]:
        return tuple(self._items.values())

    def ids(self) -> Tuple[str, ...]:
        return tuple(self._items)

    def __len__(self) -> int:
        return len(self._items)


class SkillRegistry(_Registry[SkillCapability]):
    def __init__(self) -> None:
        super().__init__("skill_id")


class ToolRegistry(_Registry[ToolCapability]):
    def __init__(self) -> None:
        super().__init__("tool_id")
        self._ids_by_name: Dict[str, str] = {}

    def register(self, capability: ToolCapability) -> ToolCapability:
        existing_id = self._ids_by_name.get(capability.name)
        if existing_id is not None:
            raise DuplicateCapabilityError(
                f"tool name {capability.name!r} is already registered by {existing_id!r}"
            )
        registered = super().register(capability)
        self._ids_by_name[capability.name] = capability.tool_id
        return registered

    def find_by_name(self, name: str) -> Optional[ToolCapability]:
        tool_id = self._ids_by_name.get(name)
        return self.find(tool_id) if tool_id is not None else None

    def resolve(self, names_or_ids: Iterable[str]) -> Tuple[ToolCapability, ...]:
        resolved = []
        seen = set()
        for value in names_or_ids:
            item = self.find(str(value)) or self.find_by_name(str(value))
            if item is not None and item.tool_id not in seen:
                seen.add(item.tool_id)
                resolved.append(item)
        return tuple(resolved)


class AgentRegistry(_Registry[AgentCapability]):
    def __init__(self) -> None:
        super().__init__("agent_id")


class CapabilityRegistry:
    """The aggregate registry; the three domains remain independently governed."""

    def __init__(self) -> None:
        self.skills = SkillRegistry()
        self.tools = ToolRegistry()
        self.agents = AgentRegistry()
