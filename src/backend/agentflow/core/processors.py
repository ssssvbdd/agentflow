"""
Request / Response Processor 管线（对齐 trpc-agent-go 的 Processor 体系）。

把现有 AgentMiddleware 的三类能力拆成两类正交处理器：
- RequestProcessor  —— 进入模型前：配额计数、动态工具列表注入；
- ResponseProcessor —— 模型/工具响应后：工具调用事件实时推送、切流信号。

关键点：processor 直接持有 emit（事件发射器），可自行往事件流发事件——
这正是 after_model 中间件「检测工具结果就绪 → 触发切流」的优雅实现方式。
"""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, List, Optional, Protocol, Sequence, runtime_checkable

from agentflow.core.contracts import Event, EventKind
from agentflow.core.invocation import Invocation

# 事件发射器：向当前调用的事件流推送事件
Emitter = Callable[[Event], Awaitable[None]]


@runtime_checkable
class RequestProcessor(Protocol):
    async def process_request(self, inv: Invocation, req: Any, emit: Emitter) -> None: ...


@runtime_checkable
class ResponseProcessor(Protocol):
    async def process_response(self, inv: Invocation, req: Any, rsp: Any, emit: Emitter) -> None: ...


class ProcessorPipeline:
    """有序执行 request / response 两类处理器，单个处理器异常不拖垮整体。"""

    def __init__(
        self,
        request_processors: Optional[Sequence[RequestProcessor]] = None,
        response_processors: Optional[Sequence[ResponseProcessor]] = None,
    ):
        self.request_processors: List[RequestProcessor] = list(request_processors or [])
        self.response_processors: List[ResponseProcessor] = list(response_processors or [])

    def add_request(self, p: RequestProcessor) -> "ProcessorPipeline":
        self.request_processors.append(p)
        return self

    def add_response(self, p: ResponseProcessor) -> "ProcessorPipeline":
        self.response_processors.append(p)
        return self

    async def process_request(self, inv: Invocation, req: Any, emit: Emitter) -> None:
        for p in self.request_processors:
            await p.process_request(inv, req, emit)

    async def process_response(self, inv: Invocation, req: Any, rsp: Any, emit: Emitter) -> None:
        for p in self.response_processors:
            await p.process_response(inv, req, rsp, emit)


# ---------------------------------------------------------------------------
# 内置处理器：与现有 GeneralAgent 中间件能力一一对应
# ---------------------------------------------------------------------------

class QuotaRequestProcessor:
    """调用计数限流：进入模型前 inv.inc_llm_call()，超限 emit 限流事件并中止。"""

    async def process_request(self, inv: Invocation, req: Any, emit: Emitter) -> None:
        try:
            inv.inc_llm_call()
        except Exception as err:  # QuotaExceeded
            await emit(Event(
                kind=EventKind.ERROR,
                invocation_id=inv.invocation_id,
                branch=inv.branch,
                data={"error": "QuotaExceeded", "message": str(err)},
            ))
            raise


class DynamicToolInjectionProcessor:
    """动态工具列表注入：按 context 重算 req.tools（对齐 MCP ToolSet.Tools(ctx) 动态刷新）。"""

    def __init__(self, resolver: Optional[Callable[[Invocation, Any], Any]] = None):
        # resolver(inv, req) -> tools；缺省从 req.state["available_tools"] 取
        self._resolver = resolver

    async def process_request(self, inv: Invocation, req: Any, emit: Emitter) -> None:
        if self._resolver is not None:
            req.tools = await self._resolver(inv, req) or req.tools
            return
        state = getattr(req, "state", None) or {}
        if available_tools := state.get("available_tools", []):
            req.tools = available_tools


class ToolCallEventProcessor:
    """工具调用事件实时推送：检测到 tool_call 即 emit 对应事件。"""

    def __init__(self, name_resolver: Optional[Callable[[str], Any]] = None):
        self._name_resolver = name_resolver

    async def process_response(self, inv: Invocation, req: Any, rsp: Any, emit: Emitter) -> None:
        tool_calls = getattr(rsp, "tool_calls", None)
        if not tool_calls:
            return
        for call in tool_calls:
            name = call.get("name", "unknown") if isinstance(call, dict) else getattr(call, "name", "unknown")
            display = name
            if self._name_resolver:
                resolved = self._name_resolver(name)
                display = resolved if isinstance(resolved, str) else (resolved[1] if resolved else name)
            await emit(Event(
                kind=EventKind.TOOL_CALL_START,
                invocation_id=inv.invocation_id,
                branch=inv.branch,
                agent_name=inv.agent_name,
                data={"tool": name, "title": str(display)},
            ))


async def noop_emitter(event: Event) -> None:  # pragma: no cover
    pass
