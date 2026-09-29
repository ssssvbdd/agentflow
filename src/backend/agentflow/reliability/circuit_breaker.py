"""
异步熔断器（closed -> open -> half-open），对齐 trpc-agent-go 对下游调用的快速失败保护。

下游指：LLM API / 检索服务 / MCP Server。连续失败达到阈值后熔断 open，
在 cooldown 期间直接快速失败（不打出站请求），half-open 放行探测请求，
成功则恢复 closed，失败则回到 open。避免雪崩。
"""
from __future__ import annotations

import time
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Optional


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half-open"


class CircuitOpenError(Exception):
    """熔断开启期间调用被快速失败。"""


class AsyncCircuitBreaker:
    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        success_threshold: int = 2,
        cooldown_seconds: float = 30.0,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.cooldown_seconds = cooldown_seconds

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._opened_at = 0.0

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._opened_at >= self.cooldown_seconds:
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
        return self._state

    def _record_success(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
        else:
            self._failure_count = 0

    def _record_failure(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            self._trip()
        else:
            self._failure_count += 1
            if self._failure_count >= self.failure_threshold:
                self._trip()

    def _trip(self) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = time.monotonic()
        self._failure_count = 0

    async def call(self, fn: Awaitable, *args: Any, **kwargs: Any) -> Any:
        """执行受熔断保护的异步调用。"""
        if self.state == CircuitState.OPEN:
            raise CircuitOpenError(
                f"circuit [{self.name}] is open, fast-failing"
            )
        try:
            result = await fn(*args, **kwargs)
        except CircuitOpenError:
            raise
        except Exception:
            self._record_failure()
            raise
        self._record_success()
        return result


class CircuitBreakerRegistry:
    """按下游名称（如 llm:qwen-max / mcp:filesystem / rag:default）注册熔断器。"""

    def __init__(self) -> None:
        self._breakers: Dict[str, AsyncCircuitBreaker] = {}

    def get(
        self,
        name: str,
        failure_threshold: int = 5,
        cooldown_seconds: float = 30.0,
    ) -> AsyncCircuitBreaker:
        if name not in self._breakers:
            self._breakers[name] = AsyncCircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                cooldown_seconds=cooldown_seconds,
            )
        return self._breakers[name]

    async def call(
        self,
        name: str,
        fn: Callable[..., Awaitable],
        *args: Any,
        failure_threshold: int = 5,
        cooldown_seconds: float = 30.0,
        **kwargs: Any,
    ) -> Any:
        breaker = self.get(name, failure_threshold=failure_threshold, cooldown_seconds=cooldown_seconds)
        return await breaker.call(fn, *args, **kwargs)

    def snapshot(self) -> Dict[str, str]:
        return {name: b.state.value for name, b in self._breakers.items()}


circuit_registry = CircuitBreakerRegistry()
