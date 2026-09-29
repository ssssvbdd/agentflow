import json
from typing import List, Callable
from fastapi import APIRouter, Depends, HTTPException, Request
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage

from agentflow.core.agents.general_agent import GeneralAgent, AgentConfig
from agentflow.core.contracts import EventKind, QuotaExceeded
from agentflow.api.services.history import HistoryService
from agentflow.api.services.dialog import DialogService
from agentflow.api.responses.streaming import WatchedStreamingResponse
from agentflow.api.services.user import UserPayload, get_login_user
from agentflow.prompts.completion import SYSTEM_PROMPT
from agentflow.schemas.completion import CompletionReq
from agentflow.reliability.admission import OverloadedError, admission_controller
from agentflow.reliability.limiter import OutputBudget
from agentflow.services.memory.client import memory_client
from agentflow.services.memory_pipeline import memory_pipeline
from agentflow.settings import app_settings
from agentflow.utils.common import count_tokens_usage
from agentflow.utils.contexts import set_user_id_context, set_agent_name_context
from agentflow.utils.helpers import build_completion_system_prompt, build_completion_user_input

router = APIRouter(tags=["Completion"])

async def _acquire_entry_quota(request: Request, user_id: str, dialog_id: str) -> None:
    """Distributed token buckets before expensive Agent initialization."""
    limiter = getattr(request.app.state, "quota_limiter", None)
    if limiter is None:
        return

    reliability_cfg = app_settings.reliability or {}
    await limiter.acquire(
        "global:completion",
        capacity=int(reliability_cfg.get("global_qps_capacity", 300)),
        refill_rate=float(reliability_cfg.get("global_qps_refill_rate", 300)),
    )
    await limiter.acquire(
        f"user:{user_id}:completion",
        capacity=int(reliability_cfg.get("user_qps_capacity", 60)),
        refill_rate=float(reliability_cfg.get("user_qps_refill_rate", 2)),
    )
    # Dialog dimension smooths repeated reconnects or client retry storms.
    await limiter.acquire(
        f"dialog:{dialog_id}:completion",
        capacity=int(reliability_cfg.get("dialog_qps_capacity", 20)),
        refill_rate=float(reliability_cfg.get("dialog_qps_refill_rate", 1)),
    )

@router.post("/completion", description="对话接口")
async def completion(
    *,
    req: CompletionReq,
    request: Request,
    login_user: UserPayload = Depends(get_login_user)
):
    """
    实时对话接口（SSE流式）
    """

    ticket = None
    try:
        await _acquire_entry_quota(request, login_user.user_id, req.dialog_id)
        ticket = await admission_controller.acquire(login_user.user_id)

        # Agent 初始化
        db_config = await DialogService.get_agent_by_dialog_id(req.dialog_id)
        agent_config = AgentConfig(**db_config)
        agent_config.user_id = login_user.user_id
    except QuotaExceeded as err:
        raise HTTPException(status_code=429, detail=str(err)) from err
    except OverloadedError as err:
        raise HTTPException(status_code=503, detail=str(err)) from err
    except Exception:
        await admission_controller.release(ticket)
        raise

    # 设置上下文信息
    set_user_id_context(login_user.user_id)
    set_agent_name_context(agent_config.name)

    chat_agent = GeneralAgent(agent_config)
    await chat_agent.init_agent()

    # 输入处理
    raw_input = req.user_input

    user_input = build_completion_user_input(
        file_url=req.file_url,
        user_input=raw_input
    )

    # Prompt 构建
    system_prompt = agent_config.system_prompt.strip() or SYSTEM_PROMPT

    short_history = await HistoryService.get_short_term_messages(
        req.dialog_id, login_user.user_id
    )
    history_summary = await DialogService.get_dialog_history_summary(req.dialog_id)

    long_memory = None
    if agent_config.enable_memory:
        memories = await memory_client.search(
            query=raw_input,
            run_id=req.dialog_id
        )
        long_memory = "\n".join(
            m.get("memory", "") for m in memories.get("results", [])
        )

    system_prompt = build_completion_system_prompt(
        system_prompt,
        history_summary,
        long_memory
    )

    messages: List[BaseMessage] = [
        SystemMessage(content=system_prompt),
        *short_history,
        HumanMessage(content=user_input),
    ]

    # 事件 & 流式响应
    events: list = []
    output_budget = OutputBudget(
        max_bytes=int((app_settings.reliability or {}).get("output_max_bytes", 1024 * 1024))
    )

    async def stream():
        response_content = " "
        run_id = ""
        try:
            async for event in chat_agent.astream(messages):
                if event.get("type") == EventKind.RUN_STARTED.value:
                    run_id = event.get("data", {}).get("run_id", "")

                if event.get("type") == "response_chunk":
                    chunk = event["data"].get("chunk", "")
                    try:
                        output_budget.consume(chunk)
                    except QuotaExceeded as err:
                        chat_agent.stop_streaming_callback()
                        error_event = {
                            "type": EventKind.ERROR.value,
                            "timestamp": event.get("timestamp"),
                            "data": {
                                "error": "OutputBudgetExceeded",
                                "message": str(err),
                            },
                        }
                        events.append(error_event)
                        yield f"data: {json.dumps(error_event)}\n\n"
                        break
                    response_content += chunk
                else:
                    events.append(event)

                if run_id:
                    stream_id = await request.app.state.event_store.append(run_id, login_user.user_id, event)
                    if stream_id:
                        event["event_id"] = stream_id
                yield f"data: {json.dumps(event)}\n\n"

        finally:
            try:
                if agent_config.enable_memory:
                    await memory_pipeline.publish(
                        messages=[
                            {"role": "user", "content": raw_input},
                            {"role": "assistant", "content": response_content}
                        ],
                        run_id=req.dialog_id,
                        user_id=login_user.user_id,
                        agent_id=db_config.get("id", ""),
                    )

                await HistoryService.save_chat_history(
                    role="assistant",
                    content=response_content,
                    events=events,
                    dialog_id=req.dialog_id,
                    token_usage=count_tokens_usage(response_content),
                    memory_enable=agent_config.enable_memory
                )

                await DialogService.update_dialog_summary(
                    dialog_id=req.dialog_id,
                    user_id=login_user.user_id,
                )
            finally:
                await admission_controller.release(ticket)

    # 用户消息先落库；若这里失败，流不会启动，需要归还 admission slot。
    try:
        await HistoryService.save_chat_history(
            role="user",
            content=raw_input,
            events=events,
            dialog_id=req.dialog_id,
            token_usage=count_tokens_usage(raw_input),
            memory_enable=agent_config.enable_memory
        )
    except Exception:
        await admission_controller.release(ticket)
        raise

    # 返回流式响应
    return WatchedStreamingResponse(
        content=stream(),
        callback=chat_agent.stop_streaming_callback,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
