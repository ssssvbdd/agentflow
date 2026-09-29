"""
统一 Span 入口（对齐 trpc-agent-go internal/trace.StartSpan）。

- 所有埋点走单一入口，支持按请求关闭追踪（inv.disable_tracing -> NoOp，零开销）；
- FastAPI 入站 span 通过 W3C traceparent 头做跨服务传播
  （对齐 trpc-agent-go 的 propagation.TraceContext{}）；
- 严格采用 OTel GenAI semantic conventions（gen_ai.*），
  后端（Langfuse/Jaeger/Tempo/Grafana）可开箱识别。

依赖可选：未安装 opentelemetry-api 时全部降级为 NoOp，不影响业务。
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from agentflow.core.invocation import Invocation

try:  # OTel 可选依赖，未安装时降级
    from opentelemetry import trace as _otel_trace
    from opentelemetry.trace import Span, Tracer, Status, StatusCode

    _TRACER: Optional[Tracer] = _otel_trace.get_tracer("agchat")
    _OTEL_AVAILABLE = True
except Exception:  # pragma: no cover
    _TRACER = None
    _OTEL_AVAILABLE = False


class _NoOpSpan:
    """零开销空 Span。"""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, status: Any) -> None:
        pass

    def record_exception(self, exception: BaseException) -> None:
        pass

    def end(self) -> None:
        pass

    @property
    def is_recording(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def start_span(inv: Optional[Invocation], name: str, **attributes: Any):
    """统一 Span 入口：注入 invocation_id / branch 归因属性。"""
    if _TRACER is None or (inv is not None and inv.disable_tracing):
        return _NoOpSpan()
    span = _TRACER.start_span(name)
    if inv is not None:
        span.set_attribute("agchat.invocation_id", inv.invocation_id)
        span.set_attribute("agchat.branch", inv.branch)
        if inv.agent_name:
            span.set_attribute("agchat.agent_name", inv.agent_name)
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, value)
    return span


@contextmanager
def span_scope(inv: Optional[Invocation], name: str, **attributes: Any):
    """上下文管理器形态：异常自动记录并标记 ERROR。"""
    span = start_span(inv, name, **attributes)
    try:
        yield span
    except Exception as err:
        span.record_exception(err)
        if _OTEL_AVAILABLE:
            span.set_status(Status(StatusCode.ERROR, str(err)))
        raise
    finally:
        span.end()


# ---------------------------------------------------------------------------
# 三层埋点（Model / Tool / Runner），GenAI 语义约定
#   对齐 trpc-agent-go：
#   - Model:  span "chat {model}"  -> gen_ai.request.model / gen_ai.usage.*
#   - Tool:   span "execute_tool {name}" -> gen_ai.tool.* / error.type
#   - Runner: span "runner.run" / "runner.event_loop" / "runner.event.persist"
# ---------------------------------------------------------------------------

def start_model_span(inv: Optional[Invocation], model_name: str):
    return start_span(
        inv,
        f"chat {model_name}",
        **{"gen_ai.request.model": model_name, "gen_ai.operation.name": "chat"},
    )


def record_model_usage(
    span,
    model_name: str,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    cached_tokens: Optional[int] = None,
    finish_reason: Optional[str] = None,
) -> None:
    """记录模型 token 用量（对齐 ChatMetricsTracker 上报的属性集）。"""
    if not getattr(span, "is_recording", False):
        return
    span.set_attribute("gen_ai.response.model", model_name)
    if input_tokens is not None:
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
    if output_tokens is not None:
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
    if cached_tokens is not None:
        span.set_attribute("gen_ai.usage.cached_tokens", cached_tokens)
    if finish_reason:
        span.set_attribute("gen_ai.response.finish_reason", finish_reason)


def start_tool_span(inv: Optional[Invocation], tool_name: str, tool_call_id: str = ""):
    return start_span(
        inv,
        f"execute_tool {tool_name}",
        **{
            "gen_ai.tool.name": tool_name,
            "gen_ai.tool.call.id": tool_call_id,
            "gen_ai.operation.name": "execute_tool",
        },
    )


def record_tool_result(span, arguments: Any = None, result: Any = None, error_type: Optional[str] = None) -> None:
    if not getattr(span, "is_recording", False):
        return
    if arguments is not None:
        span.set_attribute("gen_ai.tool.call.arguments", str(arguments)[:2048])
    if result is not None:
        span.set_attribute("gen_ai.tool.call.result", str(result)[:2048])
    if error_type:
        span.set_attribute("error.type", error_type)


def start_runner_span(inv: Optional[Invocation], agent_name: str = "agent"):
    return start_span(inv, "runner.run", **{"agchat.runner.agent": agent_name})


class TTFTTracker:
    """首 Token 时延（TTFT）追踪器——核心 KPI（目标 < 800ms）。"""

    def __init__(self) -> None:
        self._start = time.perf_counter()
        self._ttft: Optional[float] = None

    def mark_first_token(self) -> Optional[float]:
        """首个 response_chunk 到达时调用；返回 TTFT（毫秒）。"""
        if self._ttft is None:
            self._ttft = (time.perf_counter() - self._start) * 1000.0
        return self._ttft

    @property
    def ttft_ms(self) -> Optional[float]:
        return self._ttft


def extract_traceparent(headers: Any) -> Optional[str]:
    """提取 W3C traceparent 头（跨服务传播，对齐 propagation.TraceContext{}）。"""
    getter = headers.get if hasattr(headers, "get") else None
    if getter is None:
        return None
    value = getter("traceparent") or getter("Traceparent")
    return value if isinstance(value, str) else None
