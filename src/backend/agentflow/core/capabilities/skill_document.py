"""SKILL.md frontmatter parsing and progressive instruction loading."""
from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Tuple

try:  # PyYAML is present in the application image but not required by this core module.
    import yaml
except ImportError:  # pragma: no cover - exercised in minimal test environments.
    yaml = None

from agentflow.core.capabilities.models import SkillCapability
from agentflow.core.capabilities.registry import SkillRegistry


def find_skill_markdown(folder: Mapping[str, Any] | None) -> Tuple[str, str]:
    """Return ``(path, content)`` for the first SKILL.md in a folder tree."""
    if not isinstance(folder, Mapping):
        return "", ""
    for item in folder.get("folder", []) or []:
        if not isinstance(item, Mapping):
            continue
        if item.get("type", "file") == "file" and item.get("name") == "SKILL.md":
            return str(item.get("path") or "SKILL.md"), str(item.get("content") or "")
        if item.get("type") == "folder":
            path, content = find_skill_markdown(item)
            if path:
                return path, content
    return "", ""


def parse_skill_document(
    *,
    skill_id: str,
    fallback_name: str,
    fallback_description: str,
    folder: Mapping[str, Any] | None,
) -> SkillCapability:
    path, raw = find_skill_markdown(folder)
    frontmatter, body = _split_frontmatter(raw)
    name = str(frontmatter.get("name") or fallback_name or skill_id).strip()
    description = str(
        frontmatter.get("description") or fallback_description or ""
    ).strip()
    instructions = body.strip() or raw.strip()
    return SkillCapability(
        skill_id=skill_id,
        name=name,
        description=description,
        instructions_uri=path or f"skill://{skill_id}/SKILL.md",
        instructions=instructions,
        required_tools=_as_string_tuple(frontmatter.get("required_tools")),
        optional_tools=_as_string_tuple(frontmatter.get("optional_tools")),
        examples=_as_string_tuple(frontmatter.get("examples")),
        version=str(frontmatter.get("version") or "1"),
    )


class SkillLoader:
    """Loads full instructions only after the selector chooses a skill."""

    def __init__(self, registry: SkillRegistry) -> None:
        self._registry = registry

    def load(self, skill_ids: Iterable[str]) -> Tuple[str, ...]:
        return tuple(
            self._registry.get(skill_id).instructions
            for skill_id in skill_ids
        )


def _split_frontmatter(raw: str) -> Tuple[dict, str]:
    normalized = (raw or "").replace("\r\n", "\n")
    if not normalized.startswith("---\n"):
        return {}, normalized
    end = normalized.find("\n---\n", 4)
    if end < 0:
        return {}, normalized
    frontmatter = normalized[4:end]
    if yaml is None:
        metadata = _parse_basic_yaml(frontmatter)
    else:
        try:
            metadata = yaml.safe_load(frontmatter) or {}
        except yaml.YAMLError:
            metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    return metadata, normalized[end + 5 :]


def _parse_basic_yaml(raw: str) -> dict:
    """Small fallback for the scalar/list subset used by SKILL.md metadata."""
    result: dict[str, Any] = {}
    active_list_key = ""
    for raw_line in raw.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("-") and active_list_key:
            result[active_list_key].append(_parse_scalar(stripped[1:].strip()))
            continue
        if ":" not in raw_line:
            active_list_key = ""
            continue
        key, value = raw_line.split(":", 1)
        key = key.strip()
        value = value.strip()
        active_list_key = ""
        if not value:
            result[key] = []
            active_list_key = key
        elif value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            result[key] = [
                _parse_scalar(item.strip())
                for item in inner.split(",")
                if item.strip()
            ]
        else:
            result[key] = _parse_scalar(value)
    return result


def _parse_scalar(value: str) -> Any:
    if not value:
        return ""
    if value[0:1] in {'"', "'"} and value[-1:] == value[0]:
        return value[1:-1]
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "Null", "NULL", "~"}:
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def _as_string_tuple(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        values = [str(part).strip() for part in value]
    else:
        values = [str(value).strip()]
    return tuple(part for part in values if part)
