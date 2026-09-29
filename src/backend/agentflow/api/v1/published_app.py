import json
import time
from typing import List

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from agentflow.api.services.published_app import PublishedAppError, PublishedAppService
from agentflow.api.services.user import UserPayload, get_login_user
from agentflow.core.agents.general_agent import AgentConfig, GeneralAgent
from agentflow.core.contracts import EventKind
from agentflow.prompts.completion import SYSTEM_PROMPT
from agentflow.schemas.published_app import (
    CreateApiKeyReq,
    PublicCompletionReq,
    PublishAppReq,
    UpdateAppReq,
)
from agentflow.utils.helpers import build_completion_system_prompt, build_completion_user_input

router = APIRouter(prefix="/apps", tags=["Published Apps"])
public_router = APIRouter(prefix="/public/apps", tags=["Public Apps"])


def _http_error(err: Exception) -> HTTPException:
    return HTTPException(status_code=getattr(err, "status_code", 400), detail=str(err))


@router.post("/publish", description="发布 Agent 为可调用应用")
async def publish_app(req: PublishAppReq, login_user: UserPayload = Depends(get_login_user)):
    try:
        return {"code": 0, "data": await PublishedAppService.publish(req, login_user.user_id)}
    except Exception as err:
        raise _http_error(err) from err


@router.get("", description="获取当前用户发布的应用")
async def list_apps(login_user: UserPayload = Depends(get_login_user)):
    return {"code": 0, "data": await PublishedAppService.list_apps(login_user.user_id)}


@router.get("/{app_id}", description="获取发布应用详情")
async def get_app_detail(app_id: str, login_user: UserPayload = Depends(get_login_user)):
    try:
        data = await PublishedAppService.get_app_detail(app_id, login_user.user_id)
        return {"code": 0, "data": data}
    except Exception as err:
        raise _http_error(err) from err


@router.put("/{app_id}", description="更新发布应用元数据和访问方式")
async def update_app(
    app_id: str,
    req: UpdateAppReq,
    login_user: UserPayload = Depends(get_login_user),
):
    try:
        data = await PublishedAppService.update_app(app_id, login_user.user_id, req)
        return {"code": 0, "data": data}
    except Exception as err:
        raise _http_error(err) from err


@router.post("/keys", description="为已发布应用创建 API Key")
async def create_api_key(req: CreateApiKeyReq, login_user: UserPayload = Depends(get_login_user)):
    try:
        return {"code": 0, "data": await PublishedAppService.create_api_key(req.app_id, login_user.user_id, req.name)}
    except Exception as err:
        raise _http_error(err) from err


@router.get("/{app_id}/keys", description="列出应用 API Key（不返回密钥原文）")
async def list_api_keys(app_id: str, login_user: UserPayload = Depends(get_login_user)):
    try:
        data = await PublishedAppService.list_api_keys(app_id, login_user.user_id)
        return {"code": 0, "data": data}
    except Exception as err:
        raise _http_error(err) from err


@router.delete("/{app_id}/keys/{key_id}", description="吊销应用 API Key")
async def revoke_api_key(
    app_id: str,
    key_id: str,
    login_user: UserPayload = Depends(get_login_user),
):
    try:
        await PublishedAppService.revoke_api_key(app_id, key_id, login_user.user_id)
        return {"code": 0}
    except Exception as err:
        raise _http_error(err) from err


@router.post("/{app_id}/keys/{key_id}/rotate", description="轮换应用 API Key")
async def rotate_api_key(
    app_id: str,
    key_id: str,
    login_user: UserPayload = Depends(get_login_user),
):
    try:
        data = await PublishedAppService.rotate_api_key(app_id, key_id, login_user.user_id)
        return {"code": 0, "data": data}
    except Exception as err:
        raise _http_error(err) from err


@router.post("/{app_id}/pause", description="暂停发布应用")
async def pause_app(app_id: str, login_user: UserPayload = Depends(get_login_user)):
    try:
        await PublishedAppService.pause(app_id, login_user.user_id)
        return {"code": 0}
    except Exception as err:
        raise _http_error(err) from err


@router.post("/{app_id}/resume", description="恢复发布应用")
async def resume_app(app_id: str, login_user: UserPayload = Depends(get_login_user)):
    try:
        data = await PublishedAppService.set_status(app_id, login_user.user_id, PublishedAppService.ACTIVE)
        return {"code": 0, "data": data}
    except Exception as err:
        raise _http_error(err) from err


@router.post("/{app_id}/republish", description="以当前 Agent 配置发布新版本")
async def republish_app(app_id: str, login_user: UserPayload = Depends(get_login_user)):
    try:
        data = await PublishedAppService.republish(app_id, login_user.user_id)
        return {"code": 0, "data": data}
    except Exception as err:
        raise _http_error(err) from err


@router.get("/{app_id}/versions", description="获取应用发布版本")
async def list_app_versions(app_id: str, login_user: UserPayload = Depends(get_login_user)):
    try:
        data = await PublishedAppService.list_versions(app_id, login_user.user_id)
        return {"code": 0, "data": data}
    except Exception as err:
        raise _http_error(err) from err


@router.post("/{app_id}/versions/{version_id}/rollback", description="回滚到指定应用版本")
async def rollback_app_version(
    app_id: str,
    version_id: str,
    login_user: UserPayload = Depends(get_login_user),
):
    try:
        data = await PublishedAppService.rollback(app_id, version_id, login_user.user_id)
        return {"code": 0, "data": data}
    except Exception as err:
        raise _http_error(err) from err


@router.get("/{app_id}/audit-logs", description="获取应用审计日志")
async def list_app_audit_logs(
    app_id: str,
    action: str = Query(default="", max_length=64),
    status: str = Query(default="", max_length=32),
    limit: int = Query(default=100, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    login_user: UserPayload = Depends(get_login_user),
):
    try:
        data = await PublishedAppService.list_audit_logs(
            app_id,
            login_user.user_id,
            action=action,
            status=status,
            limit=limit,
            offset=offset,
        )
        return {"code": 0, "data": data}
    except Exception as err:
        raise _http_error(err) from err


@public_router.post("/{slug}/completion", description="通过 API Key 调用发布应用")
async def public_completion(
    slug: str,
    req: PublicCompletionReq,
    request: Request,
    authorization: str = Header(default=""),
):
    api_key = authorization.removeprefix("Bearer ").strip()
    if not api_key:
        raise HTTPException(status_code=401, detail="missing api key")
    try:
        app, key = await PublishedAppService.authenticate_key(api_key)
    except Exception as err:
        raise HTTPException(status_code=401, detail=str(err)) from err
    if app.slug != slug:
        raise HTTPException(status_code=404, detail="app not found")

    try:
        agent_data, app_version = await PublishedAppService.resolve_runtime_config(app)
    except PublishedAppError as err:
        raise _http_error(err) from err

    agent_config = AgentConfig(**agent_data)
    agent_config.user_id = app.owner_id
    chat_agent = GeneralAgent(agent_config)
    await chat_agent.init_agent()

    user_input = build_completion_user_input(file_url=req.file_url, user_input=req.user_input)
    system_prompt = agent_config.system_prompt.strip() or SYSTEM_PROMPT
    system_prompt = build_completion_system_prompt(system_prompt, history_summary="", long_term_memory="")
    messages: List[BaseMessage] = [SystemMessage(content=system_prompt), HumanMessage(content=user_input)]

    async def stream():
        run_id = ""
        response_content = ""
        status = "success"
        try:
            async for event in chat_agent.astream(messages):
                if event.get("type") == EventKind.RUN_STARTED.value:
                    run_id = event.get("data", {}).get("run_id", "")
                if event.get("type") == "response_chunk":
                    response_content += event.get("data", {}).get("chunk", "")
                if run_id:
                    stream_id = await request.app.state.event_store.append(run_id, app.owner_id, event)
                    if stream_id:
                        event["event_id"] = stream_id
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as err:
            status = "error"
            error_event = {
                "type": EventKind.ERROR.value,
                "timestamp": time.time(),
                "data": {"error": type(err).__name__, "message": str(err)},
            }
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
        finally:
            await PublishedAppService.audit(
                owner_id=app.owner_id,
                actor_id=f"api_key:{key.key_prefix}",
                app_id=app.app_id,
                agent_id=app.agent_id,
                action="public_completion",
                trace_id=run_id,
                status=status,
                ip=request.client.host if request.client else "",
                user_agent=request.headers.get("user-agent", ""),
                detail={
                    "input_chars": len(req.user_input),
                    "output_chars": len(response_content),
                    "version": app_version.version if app_version else None,
                    "version_id": app_version.version_id if app_version else None,
                },
            )

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
