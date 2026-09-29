"""
工具调用治理：并发池 + 分布式限流 + 超时 + 幂等结果复用。

Agent 场景里工具通常比模型更容易成为瓶颈：外部 SaaS 有 QPS 限制，
MCP Server/自定义 OpenAPI 工具也可能被多个用户同时打满。本模块把工具
调用前的保护收敛到一个入口，避免每个工具各自实现一套防护逻辑。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, replace
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from agentflow.core.contracts import QuotaExceeded


class ToolGuardError(Exception):
    """工具治理层错误基类。"""


class ToolBusyError(ToolGuardError):
    """工具并发池已满，调用快速失败。"""


@dataclass(frozen=True)
class ToolPolicy:
    max_concurrency: int = 8
    acquire_timeout_seconds: float = 0.5
    timeout_seconds: float = 30.0
    qps_capacity: int = 60
    qps_refill_rate: float = 10.0
    user_qps_capacity: int = 20
    user_qps_refill_rate: float = 2.0
    idempotency_ttl_seconds: int = 3600


class ToolGuard:
    """统一保护所有 Agent 工具调用。

    - 并发池：同一个工具最多同时执行 N 个调用；
    - 限流：按 tool 和 user+tool 两个维度扣 Redis 令牌桶；
    - 超时：限制一次工具调用总耗时；
    - 幂等：相同幂等键 + 参数的成功结果在 TTL 内直接复用。
    """

    def __init__(self) -> None:
        self._default_policy = ToolPolicy()
        self._policies: Dict[str, ToolPolicy] = {}
        self._semaphores: Dict[str, asyncio.BoundedSemaphore] = {}
        self._active_by_tool: Dict[str, int] = {}
        self._idempotency_cache: Dict[str, Tuple[float, Any]] = {}
        self._lock = asyncio.Lock()
        self._quota_limiter = None

    def configure(self, config: Optional[dict]) -> None:
        config = config or {}
        default_cfg = config.get("default") or {}
        self._default_policy = self._build_policy(self._default_policy, default_cfg)

        policies: Dict[str, ToolPolicy] = {}
        for tool_name, policy_cfg in (config.get("tools") or {}).items():
            policies[str(tool_name)] = self._build_policy(self._default_policy, policy_cfg or {})
        self._policies = policies

    def bind_quota_limiter(self, quota_limiter) -> None:
        self._quota_limiter = quota_limiter

    async def run(
        self,
        *,
        tool_name: str,
        user_id: str,
        args: Any,
        idempotency_key: str,
        call: Callable[[], Awaitable[Any]],
    ) -> Any:
        policy = self._policy_for(tool_name)
        cache_key = self._cache_key(user_id, tool_name, args, idempotency_key)
        if cache_key:
            cached = await self._get_cached(cache_key)
            if cached is not None:
                return cached

        await self._acquire_quota(tool_name, user_id, policy)
        semaphore = await self._semaphore_for(tool_name, policy.max_concurrency)

        acquired = False
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=policy.acquire_timeout_seconds)
            acquired = True
            await self._mark_active(tool_name, 1)

            result = await asyncio.wait_for(call(), timeout=policy.timeout_seconds)
            if cache_key:
                await self._set_cached(cache_key, policy.idempotency_ttl_seconds, result)
            return result
        except asyncio.TimeoutError as err:
            if not acquired:
                raise ToolBusyError(
                    f"tool [{tool_name}] is busy, concurrency limit={policy.max_concurrency}"
                ) from err
            raise
        finally:
            if acquired:
                semaphore.release()
                await self._mark_active(tool_name, -1)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "default": self._policy_to_dict(self._default_policy),
            "active_by_tool": dict(self._active_by_tool),
            "configured_tools": {
                name: self._policy_to_dict(policy)
                for name, policy in self._policies.items()
            },
            "idempotency_cache_size": len(self._idempotency_cache),
        }

    def _policy_for(self, tool_name: str) -> ToolPolicy:
        return self._policies.get(tool_name, self._default_policy)

    def _build_policy(self, base: ToolPolicy, values: dict) -> ToolPolicy:
        allowed = {
            "max_concurrency",
            "acquire_timeout_seconds",
            "timeout_seconds",
            "qps_capacity",
            "qps_refill_rate",
            "user_qps_capacity",
            "user_qps_refill_rate",
            "idempotency_ttl_seconds",
        }
        patch = {key: values[key] for key in allowed if key in values}
        if "max_concurrency" in patch:
            patch["max_concurrency"] = max(1, int(patch["max_concurrency"]))
        if "idempotency_ttl_seconds" in patch:
            patch["idempotency_ttl_seconds"] = max(0, int(patch["idempotency_ttl_seconds"]))
        return replace(base, **patch)

    async def _semaphore_for(self, tool_name: str, max_concurrency: int) -> asyncio.BoundedSemaphore:
        async with self._lock:
            semaphore = self._semaphores.get(tool_name)
            if semaphore is None:
                semaphore = asyncio.BoundedSemaphore(max_concurrency)
                self._semaphores[tool_name] = semaphore
            return semaphore

    async def _acquire_quota(self, tool_name: str, user_id: str, policy: ToolPolicy) -> None:
        if self._quota_limiter is None:
            return
        await self._quota_limiter.acquire(
            f"tool:{tool_name}",
            capacity=policy.qps_capacity,
            refill_rate=policy.qps_refill_rate,
        )
        if user_id:
            await self._quota_limiter.acquire(
                f"user:{user_id}:tool:{tool_name}",
                capacity=policy.user_qps_capacity,
                refill_rate=policy.user_qps_refill_rate,
            )

    def _cache_key(
        self,
        user_id: str,
        tool_name: str,
        args: Any,
        idempotency_key: str,
    ) -> Optional[str]:
        if not idempotency_key:
            return None
        payload = {
            "user_id": user_id,
            "tool_name": tool_name,
            "idempotency_key": idempotency_key,
            "args": args,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    async def _get_cached(self, key: str) -> Optional[Any]:
        now = time.monotonic()
        async with self._lock:
            item = self._idempotency_cache.get(key)
            if item is None:
                return None
            expires_at, value = item
            if expires_at <= now:
                self._idempotency_cache.pop(key, None)
                return None
            return value

    async def _set_cached(self, key: str, ttl_seconds: int, value: Any) -> None:
        if ttl_seconds <= 0:
            return
        async with self._lock:
            self._idempotency_cache[key] = (time.monotonic() + ttl_seconds, value)
            self._cleanup_expired_locked()

    def _cleanup_expired_locked(self) -> None:
        now = time.monotonic()
        expired = [key for key, (expires_at, _) in self._idempotency_cache.items() if expires_at <= now]
        for key in expired:
            self._idempotency_cache.pop(key, None)

    async def _mark_active(self, tool_name: str, delta: int) -> None:
        async with self._lock:
            active = max(0, self._active_by_tool.get(tool_name, 0) + delta)
            if active:
                self._active_by_tool[tool_name] = active
            else:
                self._active_by_tool.pop(tool_name, None)

    def _policy_to_dict(self, policy: ToolPolicy) -> Dict[str, Any]:
        return {
            "max_concurrency": policy.max_concurrency,
            "acquire_timeout_seconds": policy.acquire_timeout_seconds,
            "timeout_seconds": policy.timeout_seconds,
            "qps_capacity": policy.qps_capacity,
            "qps_refill_rate": policy.qps_refill_rate,
            "user_qps_capacity": policy.user_qps_capacity,
            "user_qps_refill_rate": policy.user_qps_refill_rate,
            "idempotency_ttl_seconds": policy.idempotency_ttl_seconds,
        }


tool_guard = ToolGuard()
