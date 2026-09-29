"""Permission helpers for controlled Agent-as-Tool delegation."""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

from agentflow.core.capabilities.models import ToolCapability


def capability_aliases(tools: Sequence[ToolCapability]) -> Tuple[str, ...]:
    """Return stable tool ids and names suitable for an invocation grant."""
    aliases: list[str] = []
    seen: set[str] = set()
    for tool in tools:
        for alias in (tool.tool_id, tool.name):
            if alias and alias not in seen:
                seen.add(alias)
                aliases.append(alias)
    return tuple(aliases)


def inherit_tool_permissions(
    parent_permissions: Iterable[str],
    child_tools: Sequence[ToolCapability],
) -> Tuple[str, ...]:
    """Project a parent's effective grant onto a child tool catalog.

    This function can only reduce authority.  A child tool is inherited when
    either its id or its public name is present in the parent grant.  ``*`` is
    accepted for callers that deliberately use a wildcard grant, but normal
    runtime delegation passes the parent's concrete, policy-filtered aliases.
    """
    parent = {str(value) for value in parent_permissions}
    wildcard = "*" in parent
    inherited: list[str] = []
    seen: set[str] = set()
    for tool in child_tools:
        if not wildcard and tool.tool_id not in parent and tool.name not in parent:
            continue
        for alias in (tool.tool_id, tool.name):
            if alias and alias not in seen:
                seen.add(alias)
                inherited.append(alias)
    return tuple(inherited)
