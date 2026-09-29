"""
重试 + 超时 + 退避（可靠性，对齐 trpc-agent-go 对下游调用的重试策略）。

- 仅对可重试错误重试：429 / 5xx / 超时 / 连接错误（异常消息匹配）；
- 指数退避 + 抖动（jitter），避免重试风暴；
- asyncio.wait_for 超时控制，超时会取消整个子任务树（配合 TaskGroup 级联）。

不引入额外依赖（tenacity 可选场景下用本模块即可覆盖）。
"""
from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Awaitable, Callable, Iterable, Optional, Type

# 可重试的错误特征（异常类名或消息子串）
_RETRYABLE_MARKERS = (
    "429", "too many requests", "rate limit", "timeout", "timed out",
    "connection", "temporarily unavailable", "502", "503", "504",
    "service unavailable", "bad gateway", "overloaded",
)


def is_retryable(err: BaseException) -> bool:
    if isinstance(err, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True
    text = f"{type(err).__name__} {err}".lower()
    return any(marker in text for marker in _RETRYABLE_MARKERS)


async def retry_with_backoff(
    fn: Callable[..., Awaitable],
    *args: Any,
    max_attempts: int = 3,
    initial_delay: float = 0.5,
    max_delay: float = 8.0,
    timeout: Optional[float] = None,
    retryable: Optional[Callable[[BaseException], bool]] = None,
    **kwargs: Any,
) -> Any:
    """带指数退避 + 抖动的重试执行器。

    Args:
        fn: 异步函数（同步函数请用 asyncio.to_thread 包装后传入）。
        max_attempts: 最大尝试次数（含首次）。
        timeout: 每次尝试的超时秒数（wait_for，超时取消子任务树）。
        retryable: 自定义可重试判定；缺省用 is_retryable。
    """
    check = retryable or is_retryable
    last_err: Optional[BaseException] = None
    for attempt in range(1, max_attempts + 1):
        try:
            if timeout is not None:
                return await asyncio.wait_for(fn(*args, **kwargs), timeout=timeout)
            return await fn(*args, **kwargs)
        except Exception as err:
            last_err = err
            if attempt >= max_attempts or not check(err):
                raise
            # 指数退避 + 全抖动（full jitter）
            delay = min(max_delay, initial_delay * (2 ** (attempt - 1)))
            await asyncio.sleep(random.uniform(0, delay))
    raise last_err  # pragma: no cover


class RetryPolicy:
    """可组合的重试策略（供下游客户端统一装配）。"""

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 0.5,
        max_delay: float = 8.0,
        timeout: Optional[float] = 30.0,
        retryable: Optional[Callable[[BaseException], bool]] = None,
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self.retryable = retryable

    async def run(self, fn: Callable[..., Awaitable], *args: Any, **kwargs: Any) -> Any:
        return await retry_with_backoff(
            fn, *args,
            max_attempts=self.max_attempts,
            initial_delay=self.initial_delay,
            max_delay=self.max_delay,
            timeout=self.timeout,
            retryable=self.retryable,
            **kwargs,
        )


# 常用策略预设
LLM_RETRY_POLICY = RetryPolicy(max_attempts=3, initial_delay=1.0, timeout=120.0)
INTENT_RETRY_POLICY = RetryPolicy(max_attempts=2, initial_delay=0.2, timeout=8.0)
TOOL_RETRY_POLICY = RetryPolicy(max_attempts=2, initial_delay=0.5, timeout=30.0)
RAG_RETRY_POLICY = RetryPolicy(max_attempts=3, initial_delay=0.3, timeout=15.0)


async def with_timeout(fn: Callable[..., Awaitable], *args: Any, timeout: float = 30.0, **kwargs: Any) -> Any:
    """单次协程超时（对齐沙箱的 context.WithTimeout，默认 30s）。"""
    return await asyncio.wait_for(fn(*args, **kwargs), timeout=timeout)
