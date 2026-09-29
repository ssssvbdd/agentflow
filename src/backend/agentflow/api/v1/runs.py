"""
Run 生命周期路由（对齐 trpc-agent-go Runner 的 Cancel(requestID)）。

- DELETE /api/v1/runs/{run_id}：服务端主动取消指定调用（强制 owner == current_user，防 IDOR）；
- GET  /api/v1/runs：查看当前活跃 run（仅自己名下）。
"""
import json

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from agentflow.api.services.user import UserPayload, get_login_user
from agentflow.core.concurrency import request_registry
from agentflow.reliability.admission import admission_controller
from agentflow.reliability.circuit_breaker import circuit_registry
from agentflow.reliability.tool_guard import tool_guard

router = APIRouter(tags=["Runs"])


@router.delete("/runs/{run_id}", description="取消指定的流式调用")
async def cancel_run(
    run_id: str,
    login_user: UserPayload = Depends(get_login_user),
):
    # 属主校验：只能取消自己的 run（防水平越权）
    cancelled = request_registry.cancel(run_id, owner=login_user.user_id)
    if not cancelled:
        return {"code": -1, "error_msg": "10804: run 不存在或无权取消"}
    return {"code": 0, "run_id": run_id, "status": "cancelling"}


@router.get("/runs", description="查询当前用户的活跃调用")
async def list_runs(
    login_user: UserPayload = Depends(get_login_user),
):
    runs = [
        {"run_id": rid, **info}
        for rid, info in request_registry.snapshot().items()
        if info.get("owner") == login_user.user_id
    ]
    return {"code": 0, "runs": runs}


@router.get("/runs/runtime", description="查询运行时并发与熔断状态")
async def runtime_snapshot(
    login_user: UserPayload = Depends(get_login_user),
):
    return {
        "code": 0,
        "runs": request_registry.snapshot(),
        "admission": admission_controller.snapshot(),
        "tool_guard": tool_guard.snapshot(),
        "circuits": circuit_registry.snapshot(),
    }


@router.get("/runs/{run_id}/events", description="重放指定 run 的持久化 SSE 事件")
async def replay_run_events(
    run_id: str,
    request: Request,
    after_id: str = "0-0",
    login_user: UserPayload = Depends(get_login_user),
):
    async def stream():
        async for event_id, event in request.app.state.event_store.replay(
            run_id, login_user.user_id, after_id=after_id
        ):
            yield f"id: {event_id}\n"
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
