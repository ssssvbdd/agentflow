"""
标准业务指标（对齐 trpc-agent-go ChatMetricsTracker）：

- TTFT（首 Token 时延）Histogram —— 核心 KPI，可按 model/agent 维度切分并配告警；
- 每 Token 时延 / Token 吞吐 / token 用量（含 cache）—— 成本与性能监控；
- 工具调用次数 / 耗时 / 失败率 —— 定位慢工具、错误工具；
- RAG 召回耗时 / rerank 耗时 / Top-K 命中 —— 量化检索质量。

依赖可选：未安装 opentelemetry-api 时全部降级为 NoOp，不影响业务。
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

try:
    from opentelemetry import metrics as _otel_metrics

    _METER = _otel_metrics.get_meter("agchat")
    _OTEL_AVAILABLE = True
except Exception:  # pragma: no cover
    _METER = None
    _OTEL_AVAILABLE = False


class _NoOpMetric:
    def record(self, *args: Any, **kwargs: Any) -> None:
        pass

    def add(self, *args: Any, **kwargs: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# 指标构建（OTel 存在则真实上报，否则 NoOp）
# ---------------------------------------------------------------------------

if _OTEL_AVAILABLE:
    ttft_histogram = _METER.create_histogram(
        "agchat_ttft_ms", unit="ms", description="Time to first token (TTFT)"
    )
    token_latency_histogram = _METER.create_histogram(
        "agchat_token_latency_ms", unit="ms", description="Per-token latency"
    )
    token_usage_counter = _METER.create_counter(
        "agchat_token_usage", unit="{token}", description="Token usage (input/output/cached)"
    )
    tool_duration_histogram = _METER.create_histogram(
        "agchat_tool_duration_ms", unit="ms", description="Tool execution duration"
    )
    tool_calls_counter = _METER.create_counter(
        "agchat_tool_calls_total", unit="{call}", description="Tool call count (by status)"
    )
    rag_duration_histogram = _METER.create_histogram(
        "agchat_rag_duration_ms", unit="ms", description="RAG retrieval duration (by stage)"
    )
    rag_hits_counter = _METER.create_counter(
        "agchat_rag_hits", unit="{doc}", description="Retrieved/reranked document count"
    )
    active_runs_gauge_updown = _METER.create_up_down_counter(
        "agchat_active_runs", unit="{run}", description="In-flight invocations"
    )
    intent_route_duration_histogram = _METER.create_histogram(
        "agchat_intent_route_duration_ms",
        unit="ms",
        description="Intent routing latency",
    )
    intent_route_counter = _METER.create_counter(
        "agchat_intent_routes_total",
        unit="{route}",
        description="Intent routes by source and result",
    )
else:
    ttft_histogram = _NoOpMetric()          # type: ignore[assignment]
    token_latency_histogram = _NoOpMetric()  # type: ignore[assignment]
    token_usage_counter = _NoOpMetric()      # type: ignore[assignment]
    tool_duration_histogram = _NoOpMetric()  # type: ignore[assignment]
    tool_calls_counter = _NoOpMetric()       # type: ignore[assignment]
    rag_duration_histogram = _NoOpMetric()   # type: ignore[assignment]
    rag_hits_counter = _NoOpMetric()         # type: ignore[assignment]
    active_runs_gauge_updown = _NoOpMetric() # type: ignore[assignment]
    intent_route_duration_histogram = _NoOpMetric() # type: ignore[assignment]
    intent_route_counter = _NoOpMetric() # type: ignore[assignment]


def record_ttft(ttft_ms: float, model: str = "", agent: str = "") -> None:
    attrs: Dict[str, Any] = {}
    if model:
        attrs["gen_ai.request.model"] = model
    if agent:
        attrs["agchat.agent_name"] = agent
    ttft_histogram.record(ttft_ms, attrs)


def record_token_latency(latency_ms: float, model: str = "") -> None:
    attrs = {"gen_ai.request.model": model} if model else {}
    token_latency_histogram.record(latency_ms, attrs)


def record_token_usage(
    model: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_tokens: int = 0,
) -> None:
    attrs = {"gen_ai.request.model": model} if model else {}
    if input_tokens:
        token_usage_counter.add(input_tokens, {**attrs, "agchat.token.type": "input"})
    if output_tokens:
        token_usage_counter.add(output_tokens, {**attrs, "agchat.token.type": "output"})
    if cached_tokens:
        token_usage_counter.add(cached_tokens, {**attrs, "agchat.token.type": "cached"})


def record_tool_call(tool: str, duration_ms: float, success: bool) -> None:
    status = "success" if success else "error"
    attrs = {"gen_ai.tool.name": tool, "agchat.tool.status": status}
    tool_duration_histogram.record(duration_ms, attrs)
    tool_calls_counter.add(1, attrs)


def record_rag_stage(stage: str, duration_ms: float) -> None:
    rag_duration_histogram.record(duration_ms, {"agchat.rag.stage": stage})


def record_rag_hits(rerank_count: int, top_k: int) -> None:
    rag_hits_counter.add(rerank_count, {"agchat.rag.stage": "rerank_total"})
    rag_hits_counter.add(top_k, {"agchat.rag.stage": "top_k"})


def record_intent_route(
    *,
    route: str,
    source: str,
    duration_ms: float,
    used_model: bool,
    semantic_used: bool = False,
) -> None:
    attrs = {
        "agchat.intent.route": route,
        "agchat.intent.source": source,
        "agchat.intent.used_model": used_model,
        "agchat.intent.semantic_used": semantic_used,
    }
    intent_route_duration_histogram.record(duration_ms, attrs)
    intent_route_counter.add(1, attrs)


@contextmanager
def track_duration(record_fn) -> Iterator[None]:  # type: ignore[no-untyped-def]
    """计时上下文：`with track_duration(lambda ms: record_x(ms)): ...`"""
    start = time.perf_counter()
    try:
        yield
    finally:
        record_fn((time.perf_counter() - start) * 1000.0)


@contextmanager
def track_active_run() -> Iterator[None]:
    """在途 Invocation 计数（优雅退出前可观测存量）。"""
    active_runs_gauge_updown.add(1)
    try:
        yield
    finally:
        active_runs_gauge_updown.add(-1)
