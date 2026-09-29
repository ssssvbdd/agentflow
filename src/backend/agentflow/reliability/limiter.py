"""
限流 + 配额熔断（EDOS 防护，对齐 trpc-agent-go Redis 配额）。

双层防护：
1. 进程内计数 —— Invocation.max_llm_calls / max_tool_iterations（见 core/invocation.py）；
2. 分布式配额 —— RedisQuotaLimiter（Lua 脚本原子扣减令牌桶，防并发竞态）。

多维度配额：user_id / session_id / tool_name / model 分别限流。
Redis 不可用时自动降级为进程内滑窗，业务不中断。
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

from agentflow.core.contracts import QuotaExceeded

# 原子扣减：返回 [allowed(0/1), remaining]
_LUA_TOKEN_BUCKET = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])
local now = tonumber(ARGV[4])

local bucket = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(bucket[1])
local ts = tonumber(bucket[2])

if tokens == nil then
    tokens = capacity
    ts = now
end

-- 按速率补充令牌
tokens = math.min(capacity, tokens + (now - ts) * refill_rate)

local allowed = 0
if tokens >= cost then
    tokens = tokens - cost
    allowed = 1
end

redis.call('HMSET', key, 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', key, math.ceil(capacity / math.max(refill_rate, 0.001)) + 60)
return {allowed, math.floor(tokens)}
"""


class InProcessTokenBucket:
    """进程内令牌桶（单实例部署或 Redis 降级时使用）。"""

    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._tokens: Dict[str, Tuple[float, float]] = {}  # key -> (tokens, ts)

    def try_acquire(self, key: str, cost: int = 1) -> Tuple[bool, float]:
        now = time.monotonic()
        tokens, ts = self._tokens.get(key, (float(self.capacity), now))
        tokens = min(self.capacity, tokens + (now - ts) * self.refill_rate)
        allowed = tokens >= cost
        if allowed:
            tokens -= cost
        self._tokens[key] = (tokens, now)
        return allowed, tokens


class RedisQuotaLimiter:
    """基于 Redis 的分布式配额，抵御 EDOS 资产耗尽。

    用法::

        limiter = RedisQuotaLimiter(redis_client)
        await limiter.acquire(f"user:{user_id}", capacity=100, refill_rate=1.0, cost=1)
        # 超配额 -> QuotaExceeded -> 上层 emit 限流事件
    """

    KEY_PREFIX = "agchat:quota:"

    def __init__(self, redis_client=None, default_capacity: int = 100, default_refill_rate: float = 1.0):
        self._redis = redis_client
        self._default_capacity = default_capacity
        self._default_refill_rate = default_refill_rate
        self._fallback = InProcessTokenBucket(default_capacity, default_refill_rate)
        self._lua_sha: Optional[str] = None

    async def _ensure_lua(self) -> Optional[str]:
        if self._redis is None:
            return None
        if self._lua_sha is None:
            try:
                self._lua_sha = await self._redis.script_load(_LUA_TOKEN_BUCKET)
            except Exception:
                return None
        return self._lua_sha

    async def acquire(
        self,
        key: str,
        cost: int = 1,
        capacity: Optional[int] = None,
        refill_rate: Optional[float] = None,
    ) -> int:
        """扣减配额；返回剩余令牌，超配额抛 QuotaExceeded。

        Redis 不可用时降级为进程内令牌桶（仅单实例有效，但保证可用性）。
        """
        capacity = capacity if capacity is not None else self._default_capacity
        refill_rate = refill_rate if refill_rate is not None else self._default_refill_rate
        redis_key = f"{self.KEY_PREFIX}{key}"

        sha = await self._ensure_lua()
        if sha is not None:
            try:
                allowed, remaining = await self._redis.evalsha(
                    sha, 1, redis_key, capacity, refill_rate, cost, time.time()
                )
                if int(allowed) != 1:
                    raise QuotaExceeded(f"quota exceeded for {key}: remaining={remaining}")
                return int(remaining)
            except QuotaExceeded:
                raise
            except Exception:
                pass  # Redis 故障 -> 降级

        allowed, remaining = self._fallback.try_acquire(key, cost)
        if not allowed:
            raise QuotaExceeded(f"quota exceeded for {key}: remaining={remaining}")
        return int(remaining)


# ---------------------------------------------------------------------------
# 输出字节上限（防输出爆炸，对齐 EDOS 输出限制）
# ---------------------------------------------------------------------------

class OutputBudget:
    """调用级输出预算：限制单次回答的字节总量（默认 1MB）。"""

    def __init__(self, max_bytes: int = 1 * 1024 * 1024):
        self.max_bytes = max_bytes
        self._used = 0

    def consume(self, chunk: str) -> None:
        self._used += len(chunk.encode("utf-8", errors="ignore"))
        if self._used > self.max_bytes:
            raise QuotaExceeded(
                f"output budget exceeded: {self._used}/{self.max_bytes} bytes"
            )

    @property
    def used(self) -> int:
        return self._used
