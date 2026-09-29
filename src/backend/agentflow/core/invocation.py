"""
Invocation —— 贯穿全链路的运行上下文（对齐 trpc-agent-go agent.Invocation）。

这是解决「链路追踪、限流计数、并发归因、取消传播」的单一载体：
- invocation_id + branch 可把整棵调用树串起来（trace 关联键）；
- max_llm_calls / max_tool_iterations 提供进程内配额（与分布式 Redis 配额形成双层防护）；
- cancel_event 提供取消传播；
- parent 提供并行子调用的事件归因（对齐 ParentInvocationMetadata）；
- _state + _state_lock 提供调用级状态（对齐 stateMu 保护的 state）。
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from agentflow.core.contracts import QuotaExceeded


@dataclass
class ParentMeta:
    """并行子调用消歧用的父调用元数据（对齐 ParentInvocationMetadata）。"""
    invocation_id: str
    branch: str
    agent_name: str = ""


@dataclass
class Invocation:
    # —— 身份与归因 ——
    invocation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    branch: str = "root"                       # 多 agent 执行链路径，如 "chain.parallel[0]"
    parent: Optional[ParentMeta] = None

    # —— 业务上下文（松散类型，避免与 LangChain 模型强耦合）——
    user_id: str = ""
    dialog_id: str = ""
    agent_name: str = "agent"
    model_name: str = ""
    message: Any = None
    context: Dict[str, Any] = field(default_factory=dict)

    # —— 限流计数（对齐 MaxLLMCalls / MaxToolIterations）——
    max_llm_calls: int = 20
    max_tool_iterations: int = 30
    _llm_call_count: int = 0
    _tool_iter_count: int = 0

    # —— 并发 / 取消 ——
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)

    # —— 可观测性开关（对齐 DisableTracing）——
    disable_tracing: bool = False

    # —— 调用级状态（对齐 stateMu 保护的 state）——
    _state: Dict[str, Any] = field(default_factory=dict)
    _state_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # ------------------------------------------------------------------ quota
    def inc_llm_call(self) -> int:
        """LLM 调用计数 +1，超限抛 QuotaExceeded（上层 emit 限流事件并中止）。"""
        self._llm_call_count += 1
        if self._llm_call_count > self.max_llm_calls:
            raise QuotaExceeded(
                f"LLM call limit reached: {self._llm_call_count}/{self.max_llm_calls}"
            )
        return self._llm_call_count

    def inc_tool_iteration(self) -> int:
        """工具迭代计数 +1，超限抛 QuotaExceeded。"""
        self._tool_iter_count += 1
        if self._tool_iter_count > self.max_tool_iterations:
            raise QuotaExceeded(
                f"Tool iteration limit reached: {self._tool_iter_count}/{self.max_tool_iterations}"
            )
        return self._tool_iter_count

    def absorb_child_usage(self, child: "Invocation") -> None:
        """Charge a completed delegated invocation back to its parent budget."""
        self._llm_call_count += child.llm_call_count
        self._tool_iter_count += child.tool_iter_count
        if self._llm_call_count > self.max_llm_calls:
            raise QuotaExceeded(
                f"LLM call limit reached: {self._llm_call_count}/{self.max_llm_calls}"
            )
        if self._tool_iter_count > self.max_tool_iterations:
            raise QuotaExceeded(
                "Tool iteration limit reached: "
                f"{self._tool_iter_count}/{self.max_tool_iterations}"
            )

    @property
    def llm_call_count(self) -> int:
        return self._llm_call_count

    @property
    def tool_iter_count(self) -> int:
        return self._tool_iter_count

    # ------------------------------------------------------------------ cancel
    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def cancel(self) -> None:
        """请求取消：reasoning / tool 任务内部循环需轮询 cancelled 并 break。"""
        self.cancel_event.set()

    # ------------------------------------------------------------------ state
    async def set_state(self, key: str, value: Any) -> None:
        async with self._state_lock:
            self._state[key] = value

    async def get_state(self, key: str, default: Any = None) -> Any:
        async with self._state_lock:
            return self._state.get(key, default)

    def get_state_sync(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def set_state_sync(self, key: str, value: Any) -> None:
        self._state[key] = value

    # ------------------------------------------------------------------ clone
    def clone(self, branch: Optional[str] = None, **overrides: Any) -> "Invocation":
        """为子 agent 创建隔离的子 Invocation（分支隔离事件流、继承取消信号）。

        子 Invocation 共享父的 cancel_event，保证取消可级联传播到整棵调用树。
        """
        child = Invocation(
            invocation_id=self.invocation_id,      # 同一次调用，trace 关联键一致
            branch=branch if branch is not None else self.branch,
            parent=ParentMeta(
                invocation_id=self.invocation_id,
                branch=self.branch,
                agent_name=self.agent_name,
            ),
            user_id=self.user_id,
            dialog_id=self.dialog_id,
            agent_name=overrides.pop("agent_name", self.agent_name),
            model_name=self.model_name,
            message=self.message,
            context=dict(self.context),
            max_llm_calls=self.max_llm_calls,
            max_tool_iterations=self.max_tool_iterations,
            cancel_event=self.cancel_event,        # 共享取消信号
            disable_tracing=self.disable_tracing,
        )
        for key, value in overrides.items():
            setattr(child, key, value)
        return child

    # ------------------------------------------------------------------ repr
    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Invocation(id={self.invocation_id[:8]}, branch={self.branch}, "
            f"llm={self._llm_call_count}/{self.max_llm_calls}, "
            f"tool={self._tool_iter_count}/{self.max_tool_iterations}, "
            f"cancelled={self.cancelled})"
        )
