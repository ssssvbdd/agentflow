"""
统一契约层（对齐 trpc-agent-go 的 agent.Agent / Tool / Event 抽象）。

核心决策：
1. Agent.run() 返回 AsyncIterator[Event]（只读事件流）而非阻塞返回，
   五种范式、工具事件、reasoning 流全部统一为「事件流」。
2. 接口隔离：CodeExecutor / SubAgentSetter 拆成可选 Protocol，避免大接口污染。
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, Optional, Protocol, Sequence, runtime_checkable


class QuotaExceeded(Exception):
    """调用配额超限（进程内计数或分布式配额触发）。"""


class EventKind(str, Enum):
    """事件类型，对齐 trpc-agent-go event.Event 的 kind 体系。"""

    RUN_STARTED = "run_started"
    RESPONSE_CHUNK = "response_chunk"      # reasoning / 最终回答 token 流
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    TOOL_CALL_ERROR = "tool_call_error"
    MODEL_USAGE = "model_usage"            # token 用量（含 cache）
    NOTICE = "notice"                      # 切流 / 工具结果就绪等控制信号
    ERROR = "error"
    RUN_FINISHED = "run_finished"
    RUN_CANCELLED = "run_cancelled"


@dataclass
class Event:
    """贯穿所有 Agent 范式的统一事件。"""
    kind: EventKind
    data: Dict[str, Any] = field(default_factory=dict)
    invocation_id: str = ""
    branch: str = ""
    agent_name: str = ""
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.kind.value,
            "timestamp": self.timestamp,
            "event_id": self.event_id,
            "invocation_id": self.invocation_id,
            "branch": self.branch,
            "agent_name": self.agent_name,
            "data": self.data,
        }


@dataclass
class AgentInfo:
    """Agent 元信息（对齐 trpc-agent-go agent.Info）。"""
    name: str
    description: str = ""
    kind: str = "llm"          # llm / chain / parallel / cycle / graph / agenttool
    tools_names: Sequence[str] = field(default_factory=list)


@runtime_checkable
class Tool(Protocol):
    """工具契约：名称、描述、异步执行。"""

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    async def aexecute(self, **kwargs: Any) -> Any: ...


@runtime_checkable
class Agent(Protocol):
    """所有 Agent 的统一契约。返回异步事件流而非阻塞返回。"""

    async def run(self, inv: Any) -> AsyncIterator[Event]: ...

    def tools(self) -> Sequence[Tool]: ...

    def info(self) -> AgentInfo: ...

    def sub_agents(self) -> Sequence["Agent"]: ...

    def find_sub_agent(self, name: str) -> Optional["Agent"]: ...


@runtime_checkable
class SubAgentSetter(Protocol):
    """运行时热更子 Agent 的可选能力（对齐 trpc-agent-go SubAgentSetter）。"""

    async def set_sub_agents(self, agents: Sequence[Agent]) -> None: ...


@runtime_checkable
class CodeExecutor(Protocol):
    """代码执行沙箱的可选能力（隔离 + 超时由可靠性层保证）。"""

    async def execute(self, code: str, timeout: float = 30.0) -> str: ...


def error_event(agent_name: str, error: BaseException, invocation_id: str = "", branch: str = "") -> Event:
    """单子流异常隔离时使用的归因错误事件。"""
    return Event(
        kind=EventKind.ERROR,
        agent_name=agent_name,
        invocation_id=invocation_id,
        branch=branch,
        data={"error": type(error).__name__, "message": str(error)},
    )
