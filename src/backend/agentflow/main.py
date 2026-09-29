import asyncio
import logging
import warnings
import redis.asyncio as aioredis
from contextlib import asynccontextmanager
from fastapi import FastAPI
from agentflow.auth import AuthJWT
from agentflow.auth.exceptions import AuthJWTException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from agentflow.api.JWT import Settings as AuthJwtSettings
from agentflow.core.concurrency import request_registry
from agentflow.mcp_proxy.session.manager import SessionManager
from agentflow.middleware.trace_id_middleware import TraceIDMiddleware
from agentflow.middleware.white_list_middleware import WhitelistMiddleware
from agentflow.reliability.admission import admission_controller
from agentflow.reliability.limiter import RedisQuotaLimiter
from agentflow.reliability.tool_guard import tool_guard
from agentflow.services.event_stream import event_store
from agentflow.services.memory_pipeline import memory_pipeline
from agentflow.settings import init_app_settings
from agentflow.settings import app_settings
from agentflow.telemetry.setup import setup_observability

warnings.filterwarnings("ignore")
logging.getLogger("chromadb").setLevel(logging.WARNING)

async def register_router(app: FastAPI):
    from agentflow.api.router import router

    app.include_router(router)

    # 健康探针
    @app.get("/health")
    def check_health():
        return {'status': 'OK'}


def register_middleware(app: FastAPI):
    origins = [
        '*',
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    # Trace ID的中间件操作
    app.add_middleware(TraceIDMiddleware)

    # 注册白名单中间件
    app.add_middleware(WhitelistMiddleware)

    return app


async def init_config():
    await init_app_settings()

    from agentflow.database.init_data import init_agentflow_system
    await init_agentflow_system()

def print_logo():
    from pyfiglet import Figlet

    f = Figlet(font="slant")
    print(f.renderText("Agent Chat"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_config()
    setup_observability(app)

    reliability_cfg = app_settings.reliability or {}
    admission_controller.configure(
        max_active_runs=reliability_cfg.get("max_active_runs", 100),
        max_active_runs_per_user=reliability_cfg.get("max_active_runs_per_user", 20),
        acquire_timeout_seconds=reliability_cfg.get("admission_timeout_seconds", 0.25),
    )

    redis_client = aioredis.from_url(
        app_settings.redis.get("endpoint"),
        decode_responses=True
    )
    app.state.session_manager = SessionManager(redis_client)
    app.state.event_store = event_store
    event_store.bind(redis_client)

    # 可靠性层：分布式配额限流（EDOS 防护，Redis 故障自动降级进程内）
    app.state.quota_limiter = RedisQuotaLimiter(redis_client)
    tool_guard.configure(reliability_cfg.get("tool_guard", {}))
    tool_guard.bind_quota_limiter(app.state.quota_limiter)

    await memory_pipeline.start()

    await register_router(app)
    print_logo()

    yield

    # ---- 优雅退出（对齐 trpc-agent-go 退出序列）----
    # 1) 广播取消所有活跃 Invocation（cancel_event 级联到整棵子任务树）
    cancelled = request_registry.cancel_all()
    if cancelled:
        logging.getLogger("agchat").info(f"graceful shutdown: cancelled {cancelled} active run(s)")
        # 2) 给活跃流一点 drain 时间（K8s 配 preStop + terminationGracePeriodSeconds）
        await asyncio.sleep(1.0)
    # 3) flush OTel span/metric
    try:
        from opentelemetry import trace as _trace
        provider = _trace.get_tracer_provider()
        if hasattr(provider, "force_flush"):
            await provider.force_flush(timeout_millis=3000)
        if hasattr(provider, "shutdown"):
            provider.shutdown()
    except Exception:
        pass  # OTel 未安装 / 已是默认 Provider 时静默
    # 4) 关闭下游连接
    await memory_pipeline.stop()
    await redis_client.close()


def create_app():
    app = FastAPI(
        title=app_settings.server.name,
        version=app_settings.server.version,
        lifespan=lifespan
    )

    app = register_middleware(app)

    # 配置 AuthJWT
    @AuthJWT.load_config
    def get_config():
        return AuthJwtSettings()

    # 处理 AuthJWT 异常
    @app.exception_handler(AuthJWTException)
    def authjwt_exception_handler(request, exc):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message}
        )

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agentflow.main:app", host="0.0.0.0", port=7860)
