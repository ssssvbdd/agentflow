from fastapi import APIRouter
from agentflow.api.mcp_proxy.router import mcp_proxy_router
from agentflow.api.v1.router import api_v1_router

router = APIRouter()

router.include_router(api_v1_router)
router.include_router(mcp_proxy_router)