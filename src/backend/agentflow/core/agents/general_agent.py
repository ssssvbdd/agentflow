import copy
import time
import asyncio
import json
import re
from dataclasses import replace
from loguru import logger
from pydantic import BaseModel, Field as PydanticField
from typing import List, Dict, Any, AsyncGenerator, Callable, Iterable, NotRequired, Optional
from langgraph.runtime import Runtime
from langgraph.types import Command
from langchain_core.tools import BaseTool, tool, StructuredTool
from langchain.tools.tool_node import ToolCallRequest
from langchain.agents import create_agent, AgentState
from langgraph.config import get_stream_writer
from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain.agents.middleware import ModelRequest, ModelResponse, AgentMiddleware

from agentflow.api.services.agent_skill import AgentSkillService
from agentflow.api.services.mcp_user_config import MCPUserConfigService
from agentflow.core.capabilities import (
    AgentCapability,
    ApplicationPolicy,
    BlockListApplicationPolicy,
    CapabilityRegistry,
    CapabilityResolution,
    CapabilityResolver,
    KeywordSkillSelector,
    KeywordToolRetriever,
    IntentDecision,
    EmbeddingSemanticScorer,
    IntentRoutingGate,
    IntentRouter,
    SkillSelector,
    ToolCapability,
    ToolExecutorType,
    ToolRetriever,
    ToolRiskLevel,
    capability_aliases,
    inherit_tool_permissions,
    fallback_intent_decision,
    is_tool_authorized,
    parse_skill_document,
    validate_intent_decision,
)
from agentflow.core.callbacks import usage_metadata_callback
from agentflow.core.contracts import EventKind, QuotaExceeded
from agentflow.core.invocation import Invocation
from agentflow.core.run_context import current_invocation, set_current_invocation
from agentflow.core.concurrency import request_registry
from agentflow.utils.common import count_tokens_usage
from agentflow.reliability.circuit_breaker import circuit_registry
from agentflow.reliability.retry import (
    INTENT_RETRY_POLICY,
    LLM_RETRY_POLICY,
    TOOL_RETRY_POLICY,
)
from agentflow.reliability.tool_guard import tool_guard
from agentflow.telemetry import metrics
from agentflow.telemetry.tracing import (
    TTFTTracker, record_model_usage, record_tool_result,
    start_model_span, start_tool_span, start_runner_span,
)
from agentflow.database.models.user import AdminUser, SystemUser
from agentflow.tools import AgentToolsWithName
from agentflow.api.services.llm import LLMService
from agentflow.core.models.manager import ModelManager
from agentflow.api.services.tool import ToolService
from agentflow.api.services.agent import AgentService
from agentflow.services.rag.handler import RagHandler
from agentflow.core.agents.mcp_agent import MCPAgent, MCPConfig
from agentflow.api.services.mcp_server import MCPService
from agentflow.tools.openapi_tool.adapter import OpenAPIToolAdapter


class StreamAgentState(AgentState):
    tool_call_count: NotRequired[int]
    model_call_count: NotRequired[int]
    user_id: NotRequired[str]
    available_tools: NotRequired[List[BaseTool]]
    authorized_tool_names: NotRequired[List[str]]
    selected_skill_ids: NotRequired[List[str]]
    intent_route: NotRequired[str]


MAX_TOOLS_SIZE = 10

# 进程内调用配额（对齐 trpc-agent-go MaxLLMCalls / MaxToolIterations，
# 与 RedisQuotaLimiter 分布式配额形成双层防护）
MAX_LLM_CALLS = 20
MAX_TOOL_ITERATIONS = 30

class AgentConfig(BaseModel):
    id: Optional[str] = None
    user_id: str
    description: str = ""
    llm_id: str
    mcp_ids: List[str]
    knowledge_ids: List[str]
    tool_ids: List[str]
    agent_skill_ids: List[str]
    sub_agent_ids: List[str] = []
    system_prompt: str
    enable_memory: bool = False
    name: str = None
    # Optional runtime restrictions.  ``None`` means all tools already granted
    # to this agent are available to the current user; an empty list means none.
    user_tool_permissions: Optional[List[str]] = None
    blocked_tool_names: List[str] = []
    max_selected_tools: int = MAX_TOOLS_SIZE
    max_selected_skills: int = 3
    max_delegation_depth: int = 3
    enable_intent_router: bool = True
    intent_confidence_threshold: float = PydanticField(default=0.65, ge=0.0, le=1.0)
    intent_history_messages: int = PydanticField(default=12, ge=1, le=50)
    enable_semantic_intent_retrieval: bool = True
    intent_small_tool_catalog_size: int = PydanticField(default=8, ge=0, le=50)
    intent_deterministic_score_threshold: float = PydanticField(default=0.82, ge=0.0, le=1.0)
    intent_score_margin_threshold: float = PydanticField(default=0.20, ge=0.0, le=1.0)
    intent_candidate_limit: int = PydanticField(default=12, ge=3, le=50)
    intent_semantic_timeout_seconds: float = PydanticField(default=3.0, ge=0.1, le=30.0)


class DelegatedAgentInput(BaseModel):
    task: str



class EmitEventAgentMiddleware(AgentMiddleware):
    def __init__(self, name_resolver_func):
        super().__init__()

        self.name_resolver_func = name_resolver_func

    async def aafter_model(
        self, state: StreamAgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return {
                "model_call_count": state["model_call_count"] + 1
            }

        return {
            "jump_to": "end"
        }

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        # Model 层埋点（OTel GenAI 语义约定）+ 进程内配额计数
        inv = current_invocation.get()
        model_obj = getattr(request, "model", None)
        model_name = (
            getattr(model_obj, "model_name", None)
            or getattr(model_obj, "model", None)
            or (inv.model_name if inv else "")
            or "unknown"
        )
        span = start_model_span(inv, str(model_name))
        try:
            # Invocation-scoped capability resolution is authoritative.
            # Missing and empty lists both fail closed; neither may fall back
            # to the agent's complete ToolNode catalog.
            request.tools = list(request.state.get("available_tools") or [])
            if inv is not None:
                inv.inc_llm_call()   # 进程内配额（与分布式 Redis 配额形成双层防护）

            async def guarded_model_call():
                return await LLM_RETRY_POLICY.run(handler, request)

            response = await circuit_registry.call(
                f"llm:{model_name}",
                guarded_model_call,
                failure_threshold=5,
                cooldown_seconds=30.0,
            )

            # 记录 token 用量（gen_ai.usage.*）
            message = getattr(response, "result", response)
            usage = getattr(message, "usage_metadata", None) or {}
            record_model_usage(
                span, str(model_name),
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                cached_tokens=(usage.get("input_token_details") or {}).get("cached"),
                finish_reason=getattr(message, "response_metadata", {}).get("finish_reason")
                if isinstance(getattr(message, "response_metadata", None), dict) else None,
            )
            if usage:
                metrics.record_token_usage(
                    model=str(model_name),
                    input_tokens=usage.get("input_tokens") or 0,
                    output_tokens=usage.get("output_tokens") or 0,
                    cached_tokens=(usage.get("input_token_details") or {}).get("cached") or 0,
                )
            return response
        except QuotaExceeded:
            raise
        except Exception as err:
            logger.error(f"Model call error: {err}")
            span.record_exception(err)
            raise ValueError(err)
        finally:
            span.end()

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        # Tool 层埋点（gen_ai.tool.*）+ 耗时/失败率指标 + 工具迭代配额
        inv = current_invocation.get()
        tool_name = request.tool_call["name"]
        tool_call_id = request.tool_call.get("id", "")
        authorized_tool_names = request.state.get("authorized_tool_names")
        if not is_tool_authorized(tool_name, authorized_tool_names):
            reason = (
                f"工具 {tool_name!r} 不在本次调用的最终权限集合中；"
                "Skill 声明依赖不会自动授予工具权限。"
            )
            return ToolMessage(
                content=json.dumps(
                    {
                        "status": "denied",
                        "tool": tool_name,
                        "reason": reason,
                    },
                    ensure_ascii=False,
                ),
                name=tool_name,
                tool_call_id=tool_call_id,
            )
        writer = get_stream_writer()
        span = start_tool_span(inv, tool_name, tool_call_id)
        started = time.perf_counter()
        success = False

        tool_call_count = request.state.get("tool_call_count", 0)
        # 发送工具分析开始事件
        tool_type, display_tool_name = self.name_resolver_func(tool_name)
        writer({
            "status": "START",
            "title": f"执行可用{tool_type}: {display_tool_name}",
            "message": f"正在调用插件工具 {display_tool_name}..."
            })
        request.state["tool_call_count"] = tool_call_count + 1
        try:
            if inv is not None:
                inv.inc_tool_iteration()

            async def guarded_tool_call():
                return await tool_guard.run(
                    tool_name=tool_name,
                    user_id=inv.user_id if inv else request.state.get("user_id", ""),
                    args=request.tool_call.get("args", {}),
                    idempotency_key=f"{tool_call_id}:{tool_name}",
                    call=lambda: TOOL_RETRY_POLICY.run(handler, request),
                )

            tool_result = await circuit_registry.call(
                f"tool:{tool_name}",
                guarded_tool_call,
                failure_threshold=5,
                cooldown_seconds=30.0,
            )
            success = True
            record_tool_result(span, arguments=request.tool_call.get("args"), result=tool_result.content if hasattr(tool_result, "content") else tool_result)
            writer({
                "status": "END",
                "title": f"执行可用{tool_type}: {display_tool_name}",
                "message": tool_result.content
                })
            return tool_result
        except QuotaExceeded:
            raise
        except Exception as err:
            record_tool_result(span, error_type=type(err).__name__)
            span.record_exception(err)
            writer({
                "status": "ERROR",
                "title": f"执行可用{tool_type}: {display_tool_name}",
                "message": str(err)
            })
            return ToolMessage(content=str(err), name=tool_name, tool_call_id=tool_call_id)
        finally:
            span.end()
            metrics.record_tool_call(
                tool_name, (time.perf_counter() - started) * 1000.0, success
            )

class GeneralAgent:
    def __init__(
        self,
        agent_config: AgentConfig,
        *,
        application_policy: ApplicationPolicy | None = None,
        skill_selector: SkillSelector | None = None,
        tool_retriever: ToolRetriever | None = None,
        intent_routing_gate: IntentRoutingGate | None = None,
        _ancestor_agent_ids: tuple[str, ...] = (),
        _delegation_depth_limit: int | None = None,
    ):
        self.agent_config = agent_config

        self.conversation_model = None
        self.tool_invocation_model = None
        self.intent_router: IntentRouter | None = None
        self.intent_routing_gate = intent_routing_gate
        self.react_agent = None

        self.tools = []
        self.mcp_tools = []
        # Kept as compatibility aliases for callers that inspect these fields.
        # They no longer mean that MCP/Skill agents are flattened into tools.
        self.mcp_agent_as_tools = []
        self.middlewares = []
        self.skill_agent_as_tools = []
        self.agent_as_tools: List[BaseTool] = []
        self.sub_agents_by_id: Dict[str, "GeneralAgent"] = {}
        self.tool_metadata_map: Dict[str, Dict[str, str]] = {}

        self.capability_registry = CapabilityRegistry()
        self.skill_selector = skill_selector or KeywordSkillSelector(
            max_skills=agent_config.max_selected_skills
        )
        self.tool_retriever = tool_retriever or KeywordToolRetriever(
            max_tools=agent_config.max_selected_tools
        )
        self.application_policy = application_policy or BlockListApplicationPolicy(
            agent_config.blocked_tool_names
        )
        self.capability_resolver = CapabilityResolver(
            self.capability_registry,
            policy=self.application_policy,
            tool_retriever=self.tool_retriever,
        )
        self.agent_capability_id = agent_config.id or agent_config.name or "GeneralAgent"
        self._ancestor_agent_ids = tuple(_ancestor_agent_ids)
        configured_limit = max(0, agent_config.max_delegation_depth)
        self._delegation_depth_limit = (
            configured_limit
            if _delegation_depth_limit is None
            else min(configured_limit, max(0, _delegation_depth_limit))
        )
        self.last_capability_resolution: CapabilityResolution | None = None
        self.last_intent_decision: IntentDecision | None = None
        self.last_intent_routing: Dict[str, Any] = {}

        # 流式事件队列
        self.event_queue = asyncio.Queue()
        self.stop_streaming = False
        # 当前调用的 Invocation（取消传播 / 归因）
        self.current_invocation: Invocation | None = None

    def wrap_event(self, data: Dict[Any, Any]):
        """发送流式事件"""
        event = {
            "type": "event",
            "timestamp": time.time(),
            "data": data
        }
        return event

    async def init_agent(self):
        self.tools = await self.setup_tools()
        self.mcp_tools = await self.setup_mcp_tools()
        self.mcp_agent_as_tools = self.mcp_tools
        await self.setup_skill_registry()

        await self.setup_knowledge_tool()
        await self.setup_language_model()
        self.agent_as_tools = await self.setup_sub_agents_as_tools()

        self._register_agent_capability()
        self.middlewares = await self.setup_agent_middleware()
        self.react_agent = self.setup_react_agent()

    async def setup_agent_middleware(self):
        emit_event_middleware = EmitEventAgentMiddleware(self.get_tool_display_name)

        return [emit_event_middleware]


    async def setup_language_model(self):
        # 普通对话模型
        if self.agent_config.llm_id:
            model_config = await LLMService.get_llm_by_id(self.agent_config.llm_id)
            self.conversation_model = ModelManager.get_user_model(**model_config)
        else:
            self.conversation_model = ModelManager.get_conversation_model()

        # 意图识别模型
        self.tool_invocation_model = ModelManager.get_tool_invocation_model()
        self.intent_router = IntentRouter(
            self.tool_invocation_model,
            max_history_messages=self.agent_config.intent_history_messages,
            max_tool_candidates=self.agent_config.intent_candidate_limit,
        )
        if self.intent_routing_gate is None:
            semantic_scorer = None
            if self.agent_config.enable_semantic_intent_retrieval:
                try:
                    semantic_scorer = EmbeddingSemanticScorer(
                        ModelManager.get_embedding_model()
                    )
                except Exception as err:
                    logger.warning(
                        "Semantic intent retrieval disabled: {}",
                        err,
                    )
            self.intent_routing_gate = IntentRoutingGate(
                semantic_scorer=semantic_scorer,
                small_tool_catalog_size=(
                    self.agent_config.intent_small_tool_catalog_size
                ),
                deterministic_score_threshold=(
                    self.agent_config.intent_deterministic_score_threshold
                ),
                score_margin_threshold=(
                    self.agent_config.intent_score_margin_threshold
                ),
                semantic_timeout_seconds=(
                    self.agent_config.intent_semantic_timeout_seconds
                ),
                max_candidates=self.agent_config.intent_candidate_limit,
            )

    def setup_react_agent(self):
        return create_agent(
            model=self.conversation_model,
            # create_agent keeps the complete executable catalog for ToolNode,
            # while middleware exposes only the invocation-scoped intersection.
            # Skill definitions are intentionally absent from this list.
            tools=self.tools + self.mcp_tools + self.agent_as_tools,
            middleware=self.middlewares,
            state_schema=StreamAgentState
        )


    async def setup_tools(self) -> List[BaseTool]:
        def create_openapi_tool_executor(tool_adapter, tool_name):
            """闭包创建一个执行OpenAPI Tool的方法"""
            async def _execute_wrapper(**kwargs):
                return await tool_adapter.execute(
                    _tool_name=tool_name,
                    **kwargs
                )

            return _execute_wrapper

        tools = []
        db_tools = await ToolService.get_tools_from_id(self.agent_config.tool_ids)
        for db_tool in db_tools:
            if not self._user_can_access(db_tool.user_id):
                logger.warning(
                    "Skip unauthorized tool {} for user {}",
                    db_tool.tool_id,
                    self.agent_config.user_id,
                )
                continue
            if db_tool.is_user_defined:
                tool_adapter = OpenAPIToolAdapter(
                    auth_config=db_tool.auth_config,
                    openapi_schema=db_tool.openapi_schema
                )

                for openapi_tool in tool_adapter.tools:
                    tool_name = openapi_tool["function"].get("name", "")
                    langchain_tool = StructuredTool(
                        name=tool_name,
                        description=openapi_tool["function"].get("description", ""),
                        coroutine=create_openapi_tool_executor(tool_adapter, tool_name),
                        args_schema=openapi_tool,
                    )
                    tools.append(langchain_tool)

                    operation_metadata = tool_adapter.get_tool_metadata(tool_name)
                    method = str(operation_metadata.get("method") or "").upper()
                    read_only = method in {"GET", "HEAD", "OPTIONS"}
                    destructive = method == "DELETE"
                    self._register_tool(
                        langchain_tool,
                        executor_type=ToolExecutorType.OPENAPI,
                        read_only=read_only,
                        idempotent=method in {"GET", "HEAD", "OPTIONS", "PUT", "DELETE"},
                        destructive=destructive,
                        risk_level=(
                            ToolRiskLevel.HIGH if destructive
                            else ToolRiskLevel.LOW if read_only
                            else ToolRiskLevel.MEDIUM
                        ),
                        metadata={
                            "database_tool_id": db_tool.tool_id,
                            "http_method": method,
                            "path": operation_metadata.get("path", ""),
                        },
                    )

                    self.tool_metadata_map[tool_name] = {
                        "name": db_tool.display_name,
                        "type": "工具"
                    }
            else:
                agent_tool = AgentToolsWithName.get(db_tool.name)
                if agent_tool:
                    tools.append(agent_tool)
                    self._register_tool(
                        agent_tool,
                        executor_type=ToolExecutorType.BUILTIN,
                        metadata={"database_tool_id": db_tool.tool_id},
                    )
                metadata_name = agent_tool.name if agent_tool else db_tool.name
                self.tool_metadata_map[metadata_name] = {
                    "name": db_tool.display_name,
                    "type": "工具"
                }

        return tools

    async def setup_skill_registry(self) -> None:
        """Register skill metadata without adding skills to the tool surface."""
        agent_skills = await AgentSkillService.get_agent_skills_by_ids(self.agent_config.agent_skill_ids)
        for agent_skill in agent_skills:
            if not self._user_can_access(agent_skill.user_id):
                logger.warning(
                    "Skip unauthorized skill {} for user {}",
                    agent_skill.id,
                    self.agent_config.user_id,
                )
                continue
            self.capability_registry.skills.register(
                parse_skill_document(
                    skill_id=agent_skill.id,
                    fallback_name=agent_skill.name,
                    fallback_description=agent_skill.description,
                    folder=agent_skill.folder,
                )
            )

    async def setup_agent_skill_as_tools(self) -> List[BaseTool]:
        """Compatibility shim: skills are no longer auto-adapted as tools."""
        if len(self.capability_registry.skills) == 0:
            await self.setup_skill_registry()
        return []

    async def setup_sub_agents_as_tools(self) -> List[BaseTool]:
        """Create controlled delegation entries for explicitly configured agents.

        The wrapper is a Tool only at the execution boundary.  The delegated
        agent keeps its own prompt, skills, registry and tool-selection loop.
        """
        if not self.agent_config.sub_agent_ids:
            return []

        current_path = (*self._ancestor_agent_ids, self.agent_capability_id)
        current_depth = len(self._ancestor_agent_ids)
        if current_depth >= self._delegation_depth_limit:
            logger.warning(
                "Skip sub-agent setup for {}: delegation depth limit {} reached",
                self.agent_capability_id,
                self._delegation_depth_limit,
            )
            return []

        delegated_tools: List[BaseTool] = []
        for child_id in dict.fromkeys(self.agent_config.sub_agent_ids):
            if child_id in current_path:
                logger.warning(
                    "Skip cyclic sub-agent delegation: {} -> {}",
                    " -> ".join(current_path),
                    child_id,
                )
                continue

            child_data = await AgentService.select_agent_by_id(child_id)
            if not child_data:
                logger.warning("Skip unknown sub-agent {}", child_id)
                continue
            if not self._user_can_access(child_data.get("user_id")):
                logger.warning(
                    "Skip unauthorized sub-agent {} for user {}",
                    child_id,
                    self.agent_config.user_id,
                )
                continue

            # The caller remains the security principal. Resource ownership is
            # checked while the child builds its own tool catalog.
            child_data = dict(child_data)
            child_data["user_id"] = self.agent_config.user_id
            child_config = AgentConfig(**child_data)
            child_agent = GeneralAgent(
                child_config,
                _ancestor_agent_ids=current_path,
                _delegation_depth_limit=self._delegation_depth_limit,
            )
            await child_agent.init_agent()
            self.sub_agents_by_id[child_id] = child_agent

            child_capability = child_agent.capability_registry.agents.get(
                child_agent.agent_capability_id
            )
            if self.capability_registry.agents.find(child_capability.agent_id) is None:
                self.capability_registry.agents.register(child_capability)

            tool_name = self._delegated_agent_tool_name(child_id)

            async def delegate_to_agent(
                task: str,
                _child_id: str = child_id,
            ) -> str:
                return await self._execute_delegated_agent(_child_id, task)

            child_name = child_config.name or child_id
            description = (
                f"将完整任务委派给专职子 Agent「{child_name}」。"
                f"适用范围：{child_config.description or child_config.system_prompt[:160]}。"
                "子 Agent 使用独立上下文执行，且只能继承父调用已经拥有的工具权限。"
            )
            agent_tool = StructuredTool(
                name=tool_name,
                description=description,
                args_schema=DelegatedAgentInput,
                coroutine=delegate_to_agent,
            )
            delegated_tools.append(agent_tool)
            self._register_tool(
                agent_tool,
                executor_type=ToolExecutorType.DELEGATED_AGENT,
                metadata={
                    "delegated_agent_id": child_id,
                    "delegated_agent_name": child_name,
                },
            )
            self.tool_metadata_map[tool_name] = {
                "name": child_name,
                "type": "子 Agent",
            }

        return delegated_tools

    async def _execute_delegated_agent(self, child_id: str, task: str) -> str:
        parent_invocation = current_invocation.get()
        if parent_invocation is None:
            raise RuntimeError("子 Agent 委派必须发生在有效 Invocation 内")

        stack = tuple(
            parent_invocation.context.get(
                "delegation_stack",
                (self.agent_capability_id,),
            )
        )
        if child_id in stack:
            raise RuntimeError(f"检测到循环 Agent 委派: {' -> '.join((*stack, child_id))}")
        if len(stack) - 1 >= self._delegation_depth_limit:
            raise RuntimeError(
                f"Agent 委派深度超过上限 {self._delegation_depth_limit}"
            )

        child_agent = self.sub_agents_by_id.get(child_id)
        if child_agent is None:
            raise RuntimeError(f"子 Agent {child_id!r} 未注册")

        parent_permissions = tuple(
            parent_invocation.context.get("effective_tool_permissions", ())
        )
        inherited_permissions = inherit_tool_permissions(
            parent_permissions,
            child_agent.capability_registry.tools.list(),
        )
        child_invocation = parent_invocation.clone(
            branch=f"{parent_invocation.branch}.delegate[{child_id}]",
            agent_name=child_agent.agent_config.name or child_id,
            message=task,
            max_llm_calls=max(
                0,
                parent_invocation.max_llm_calls - parent_invocation.llm_call_count,
            ),
            max_tool_iterations=max(
                0,
                parent_invocation.max_tool_iterations
                - parent_invocation.tool_iter_count,
            ),
        )
        child_invocation.context.update(
            {
                "delegation_stack": (*stack, child_id),
                "delegation_depth_limit": self._delegation_depth_limit,
                "inherited_tool_permissions": inherited_permissions,
            }
        )

        child_messages: List[BaseMessage] = []
        if child_agent.agent_config.system_prompt:
            child_messages.append(
                SystemMessage(content=child_agent.agent_config.system_prompt)
            )
        child_messages.append(HumanMessage(content=task))

        response_content = ""
        error_messages: list[str] = []
        try:
            async for event in child_agent.astream(
                child_messages,
                invocation=child_invocation,
                inherited_tool_permissions=inherited_permissions,
            ):
                if event.get("type") == "response_chunk":
                    data = event.get("data") or {}
                    response_content = data.get("accumulated") or (
                        response_content + str(data.get("chunk") or "")
                    )
                elif event.get("type") == EventKind.ERROR.value:
                    data = event.get("data") or {}
                    error_messages.append(str(data.get("message") or data))
        finally:
            parent_invocation.absorb_child_usage(child_invocation)

        if response_content:
            return response_content
        if error_messages:
            raise RuntimeError("; ".join(error_messages))
        return "子 Agent 已完成执行，但没有返回文本结果。"

    @staticmethod
    def _delegated_agent_tool_name(agent_id: str) -> str:
        safe_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(agent_id)).strip("_")
        return f"delegate_agent_{safe_id[:48] or 'unknown'}"

    async def setup_mcp_tools(self) -> List[BaseTool]:
        """Expose MCP's atomic tools instead of one MCP-Agent wrapper tool."""
        mcp_tools: List[BaseTool] = []
        for mcp_id in self.agent_config.mcp_ids:
            mcp_server = await MCPService.get_mcp_server_from_id(mcp_id)
            if not self._user_can_access(mcp_server.get("user_id")):
                logger.warning(
                    "Skip unauthorized MCP server {} for user {}",
                    mcp_id,
                    self.agent_config.user_id,
                )
                continue
            mcp_config = MCPConfig(**mcp_server)

            mcp_agent = MCPAgent(mcp_config, self.agent_config.user_id)
            server_tools = await mcp_agent.setup_mcp_tools()
            for server_tool in server_tools:
                wrapped_tool = self._wrap_mcp_tool(server_tool, mcp_id)
                mcp_tools.append(wrapped_tool)

                annotations = dict(getattr(server_tool, "metadata", None) or {})
                read_only = bool(annotations.get("readOnlyHint", False))
                destructive = bool(annotations.get("destructiveHint", False))
                self._register_tool(
                    wrapped_tool,
                    executor_type=ToolExecutorType.MCP,
                    read_only=read_only,
                    idempotent=bool(annotations.get("idempotentHint", False)),
                    destructive=destructive,
                    risk_level=(
                        ToolRiskLevel.HIGH if destructive
                        else ToolRiskLevel.LOW if read_only
                        else ToolRiskLevel.MEDIUM
                    ),
                    metadata={
                        **annotations,
                        "mcp_server_id": mcp_id,
                        "mcp_server_name": mcp_config.server_name,
                    },
                )
                self.tool_metadata_map[wrapped_tool.name] = {
                    "name": f"{mcp_config.server_name} / {wrapped_tool.name}",
                    "type": "MCP 工具",
                }
        return mcp_tools

    async def setup_mcp_agent_as_tools(self) -> List[BaseTool]:
        """Compatibility alias for the old method name."""
        return await self.setup_mcp_tools()

    async def setup_knowledge_tool(self):
        @tool(parse_docstring=True)
        async def retrival_knowledge(query: str) -> str:
            """
            通过检索知识库来获取信息

            Args:
                query (str): 用户问题

            Returns:
                str: 返回从知识库检索来的信息
            """
            knowledge_message = await RagHandler.retrieve_ranked_documents(
                query, self.agent_config.knowledge_ids
            )
            return knowledge_message

        if self.agent_config.knowledge_ids: # 当绑定知识库ID后才 As Tool
            self.tools.append(retrival_knowledge)
            self._register_tool(
                retrival_knowledge,
                executor_type=ToolExecutorType.BUILTIN,
                read_only=True,
                idempotent=True,
                risk_level=ToolRiskLevel.LOW,
            )
            self.tool_metadata_map[retrival_knowledge.name] = {
                "name": "检索知识库",
                "type": "工具"
            }

    def _register_tool(
        self,
        langchain_tool: BaseTool,
        *,
        executor_type: ToolExecutorType,
        read_only: bool = False,
        idempotent: bool = False,
        destructive: bool = False,
        risk_level: ToolRiskLevel = ToolRiskLevel.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.capability_registry.tools.register(
            ToolCapability(
                tool_id=langchain_tool.name,
                name=langchain_tool.name,
                description=langchain_tool.description or "",
                executor_type=executor_type,
                tool=langchain_tool,
                read_only=read_only,
                idempotent=idempotent,
                destructive=destructive,
                risk_level=risk_level,
                metadata=metadata or {},
            )
        )

    def _user_can_access(self, owner_id: str | None) -> bool:
        return self.agent_config.user_id == AdminUser or owner_id in {
            self.agent_config.user_id,
            SystemUser,
        }

    def _wrap_mcp_tool(self, mcp_tool: BaseTool, mcp_server_id: str) -> BaseTool:
        async def call_mcp_tool(**kwargs):
            # User-owned secrets are injected after model arguments, so the
            # model cannot override them.
            personal_config = await MCPUserConfigService.get_mcp_user_config(
                self.agent_config.user_id,
                mcp_server_id,
            )
            call_args = {**kwargs, **(personal_config or {})}
            return await mcp_tool.ainvoke(call_args)

        return StructuredTool(
            name=mcp_tool.name,
            description=mcp_tool.description or "",
            args_schema=mcp_tool.args_schema,
            coroutine=call_mcp_tool,
            metadata=getattr(mcp_tool, "metadata", None),
        )

    def _register_agent_capability(self) -> None:
        self.capability_registry.agents.register(
            AgentCapability(
                agent_id=self.agent_capability_id,
                prompt=self.agent_config.system_prompt,
                available_skills=self.capability_registry.skills.ids(),
                available_sub_agents=tuple(self.sub_agents_by_id),
                max_tool_permissions=self.capability_registry.tools.ids(),
            )
        )

    def resolve_capabilities_for_task(
        self,
        task: str,
        *,
        selected_skill_ids: Iterable[str] | None = None,
        inherited_tool_permissions: Iterable[str] | None = None,
    ) -> CapabilityResolution:
        agent = self.capability_registry.agents.get(self.agent_capability_id)
        available_skills = tuple(
            self.capability_registry.skills.get(skill_id)
            for skill_id in agent.available_skills
        )
        if selected_skill_ids is None:
            selected_skill_ids = self.skill_selector.select(task, available_skills)
        else:
            selected_skill_ids = tuple(selected_skill_ids)
        configured_permissions = set(
            self.agent_config.user_tool_permissions
            if self.agent_config.user_tool_permissions is not None
            else ("*",)
        )
        inherited_permissions = (
            None
            if inherited_tool_permissions is None
            else {str(value) for value in inherited_tool_permissions}
        )

        def permission_matches(tool: ToolCapability, permissions: set[str]) -> bool:
            return (
                "*" in permissions
                or tool.tool_id in permissions
                or tool.name in permissions
            )

        # Convert the two independent grants into one concrete allowlist before
        # resolution. An empty inherited grant must remain fail-closed.
        user_permissions: list[str] = []
        for tool_capability in self.capability_registry.tools.list():
            if not permission_matches(tool_capability, configured_permissions):
                continue
            if inherited_permissions is not None and not permission_matches(
                tool_capability,
                inherited_permissions,
            ):
                continue
            user_permissions.extend(
                (tool_capability.tool_id, tool_capability.name)
            )
        resolution = self.capability_resolver.resolve(
            agent_id=self.agent_capability_id,
            selected_skill_ids=selected_skill_ids,
            user_permissions=user_permissions,
            current_step=task,
            context={
                "user_id": self.agent_config.user_id,
                "agent": self.agent_capability_id,
            },
        )
        self.last_capability_resolution = resolution
        return resolution

    async def _route_intent(
        self,
        messages: List[BaseMessage],
        current_task: str,
        *,
        invocation: Invocation,
        inherited_tool_permissions: Iterable[str] | None,
    ) -> IntentDecision:
        started = time.perf_counter()
        agent_capability = self.capability_registry.agents.get(
            self.agent_capability_id
        )
        available_skills = tuple(
            self.capability_registry.skills.get(skill_id)
            for skill_id in agent_capability.available_skills
        )
        # Policy and permission filtering happens before candidate metadata is
        # exposed to the intent model. No Skill is selected at this stage.
        pre_resolution = self.resolve_capabilities_for_task(
            current_task,
            selected_skill_ids=(),
            inherited_tool_permissions=inherited_tool_permissions,
        )
        allowed_delegate_ids = {
            str(item.metadata.get("delegated_agent_id"))
            for item in pre_resolution.allowed_tools
            if item.executor_type is ToolExecutorType.DELEGATED_AGENT
            and item.metadata.get("delegated_agent_id")
        }
        agent_candidates = [
            {
                "agent_id": child_id,
                "name": child.agent_config.name or child_id,
                "description": child.agent_config.description,
            }
            for child_id, child in self.sub_agents_by_id.items()
            if child_id in allowed_delegate_ids
        ]

        keyword_skill_ids = self.skill_selector.select(
            current_task,
            available_skills,
        )

        def finalize(
            decision: IntentDecision,
            *,
            source: str,
            used_model: bool,
            semantic_used: bool = False,
            gate_reason: str = "",
            top_score: float = 0.0,
            score_margin: float = 0.0,
            candidate_tools: Iterable[ToolCapability] = (),
        ) -> IntentDecision:
            duration_ms = (time.perf_counter() - started) * 1000.0
            routing_data = {
                "source": source,
                "used_model": used_model,
                "semantic_used": semantic_used,
                "gate_reason": gate_reason,
                "top_score": top_score,
                "score_margin": score_margin,
                "duration_ms": round(duration_ms, 3),
                "candidate_tool_ids": [
                    item.tool_id for item in candidate_tools
                ],
            }
            self.last_intent_routing = routing_data
            invocation.context["intent_routing"] = routing_data
            metrics.record_intent_route(
                route=decision.route,
                source=source,
                duration_ms=duration_ms,
                used_model=used_model,
                semantic_used=semantic_used,
            )
            return decision

        if not self.agent_config.enable_intent_router or self.intent_router is None:
            return finalize(fallback_intent_decision(
                current_task=current_task,
                selected_skill_ids=keyword_skill_ids,
                reason="意图路由未启用，使用关键词选择器",
            ), source="disabled", used_model=False,
                candidate_tools=pre_resolution.selected_tools)

        gate_result = None
        if self.intent_routing_gate is not None:
            try:
                gate_result = await self.intent_routing_gate.evaluate(
                    messages=messages,
                    current_task=current_task,
                    skills=available_skills,
                    agents=agent_candidates,
                    tools=pre_resolution.allowed_tools,
                )
            except Exception as err:
                logger.warning("Intent routing gate fallback: {}", err)

        if gate_result is not None and not gate_result.use_model:
            decision = gate_result.decision or fallback_intent_decision(
                current_task=current_task,
                reason=gate_result.reason,
            )
            return finalize(
                decision,
                source="fast_path",
                used_model=False,
                semantic_used=gate_result.semantic_used,
                gate_reason=gate_result.reason,
                top_score=gate_result.top_score,
                score_margin=gate_result.score_margin,
                candidate_tools=gate_result.tools,
            )

        route_skills = (
            gate_result.skills if gate_result is not None else available_skills
        )
        route_agents = (
            gate_result.agents if gate_result is not None else tuple(agent_candidates)
        )
        route_tools = (
            gate_result.tools
            if gate_result is not None
            else pre_resolution.allowed_tools
        )
        route_skill_ids = {item.skill_id for item in route_skills}
        route_agent_ids = {
            str(item.get("agent_id"))
            for item in route_agents
            if item.get("agent_id")
        }

        model_name = (
            getattr(self.tool_invocation_model, "model_name", None)
            or getattr(self.tool_invocation_model, "model", None)
            or "intent-router"
        )
        span = start_model_span(invocation, str(model_name))
        try:
            invocation.inc_llm_call()

            async def call_router() -> IntentDecision:
                return await self.intent_router.route(
                    messages=messages,
                    current_task=current_task,
                    skills=route_skills,
                    agents=route_agents,
                    tools=route_tools,
                )

            raw_decision = await circuit_registry.call(
                f"llm:intent:{model_name}",
                lambda: INTENT_RETRY_POLICY.run(call_router),
                failure_threshold=5,
                cooldown_seconds=30.0,
            )
            decision = validate_intent_decision(
                raw_decision,
                current_task=current_task,
                available_skill_ids=route_skill_ids,
                available_agent_ids=route_agent_ids,
                min_confidence=self.agent_config.intent_confidence_threshold,
            )
            if len(decision.skill_ids) > self.agent_config.max_selected_skills:
                decision = decision.model_copy(
                    update={
                        "skill_ids": decision.skill_ids[
                            : self.agent_config.max_selected_skills
                        ]
                    }
                )
            span.set_attribute("agchat.intent.route", decision.route)
            span.set_attribute("agchat.intent.confidence", decision.confidence)
            return finalize(
                decision,
                source="llm",
                used_model=True,
                semantic_used=(
                    gate_result.semantic_used if gate_result is not None else False
                ),
                gate_reason=(gate_result.reason if gate_result is not None else ""),
                top_score=(gate_result.top_score if gate_result is not None else 0.0),
                score_margin=(
                    gate_result.score_margin if gate_result is not None else 0.0
                ),
                candidate_tools=route_tools,
            )
        except QuotaExceeded:
            raise
        except Exception as err:
            logger.warning("Intent router fallback: {}", err)
            span.record_exception(err)
            return finalize(fallback_intent_decision(
                current_task=current_task,
                selected_skill_ids=keyword_skill_ids,
                reason=f"意图模型不可用，降级到关键词选择器: {type(err).__name__}",
            ), source="fallback", used_model=True,
                candidate_tools=pre_resolution.selected_tools)
        finally:
            span.end()

    def _apply_intent_route(
        self,
        resolution: CapabilityResolution,
        decision: IntentDecision,
    ) -> CapabilityResolution:
        if decision.route in {"direct_answer", "clarify"}:
            return replace(
                resolution,
                candidate_tools=(),
                selected_tools=(),
            )
        if decision.route == "delegate_agent" and decision.agent_id:
            delegated = tuple(
                item
                for item in resolution.allowed_tools
                if item.executor_type is ToolExecutorType.DELEGATED_AGENT
                and item.metadata.get("delegated_agent_id") == decision.agent_id
            )
            return replace(
                resolution,
                candidate_tools=delegated,
                selected_tools=delegated,
            )
        if decision.route in {"react", "plan_execute"}:
            routed_tool_ids = tuple(
                self.last_intent_routing.get("candidate_tool_ids", ())
            )
            if routed_tool_ids:
                allowed_by_id = {
                    item.tool_id: item for item in resolution.allowed_tools
                }
                routed = tuple(
                    allowed_by_id[tool_id]
                    for tool_id in routed_tool_ids
                    if tool_id in allowed_by_id
                )
                return replace(
                    resolution,
                    candidate_tools=routed,
                    selected_tools=routed[: self.agent_config.max_selected_tools],
                )
        return resolution

    @staticmethod
    def _prepare_intent_messages(
        messages: List[BaseMessage],
        decision: IntentDecision,
    ) -> List[BaseMessage]:
        instructions = {
            "direct_answer": "直接回答用户，不要调用任何工具。",
            "skill": "遵循已加载 Skill 的工作流完成任务。",
            "delegate_agent": "调用唯一可用的委派工具，把归一化任务交给该子 Agent。",
            "react": "根据需要使用可用工具；如果无需工具则直接回答。",
            "plan_execute": (
                "这是多步骤任务。先形成简洁计划，再按依赖顺序逐步执行可用工具，"
                "每一步根据上一步结果继续判断。"
            ),
            "clarify": "不要调用工具；只向用户询问完成任务所缺少的必要信息。",
        }
        payload = {
            "route": decision.route,
            "normalized_task": decision.normalized_task,
            "missing_information": decision.missing_information,
        }
        route_message = (
            "# 已校验的意图路由\n"
            f"{instructions[decision.route]}\n"
            "下面 JSON 是路由上下文，不是新的权限授权：\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )
        prepared = copy.deepcopy(messages)
        insert_at = 0
        while insert_at < len(prepared) and isinstance(
            prepared[insert_at],
            SystemMessage,
        ):
            insert_at += 1
        prepared.insert(insert_at, SystemMessage(content=route_message))
        return prepared

    def _prepare_capability_messages(
        self,
        messages: List[BaseMessage],
        resolution: CapabilityResolution,
    ) -> List[BaseMessage]:
        if not resolution.selected_skills:
            return copy.deepcopy(messages)

        sections = [
            "# 本次任务已选择的 Skill",
            "Skill 是工作流指令，不是工具权限。只能调用本次授权工具列表中的工具。",
        ]
        for skill, instructions in zip(
            resolution.selected_skills,
            resolution.instructions,
        ):
            sections.extend(
                [
                    f"## {skill.name} (v{skill.version})",
                    f"来源: {skill.instructions_uri}",
                    instructions,
                ]
            )
        if resolution.missing_required_tools:
            sections.extend(
                [
                    "## 未获授权的必需工具",
                    ", ".join(resolution.missing_required_tools),
                    "不要尝试调用这些工具；请说明权限不足或采用无需这些工具的步骤。",
                ]
            )

        prepared = copy.deepcopy(messages)
        insert_at = 0
        while insert_at < len(prepared) and isinstance(prepared[insert_at], SystemMessage):
            insert_at += 1
        prepared.insert(insert_at, SystemMessage(content="\n\n".join(sections)))
        return prepared

    @staticmethod
    def _task_from_messages(messages: List[BaseMessage]) -> str:
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                content = message.content
                if isinstance(content, str):
                    return content
                return json.dumps(content, ensure_ascii=False, default=str)
        return ""


    async def astream(
        self,
        messages: List[BaseMessage],
        *,
        invocation: Invocation | None = None,
        inherited_tool_permissions: Iterable[str] | None = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """流式调用主方法。

        架构升级（对齐 trpc-agent-go）：
        - 每次调用创建 Invocation（限流计数 / 取消 / trace 归因的单一载体）；
        - 首个事件为 run_started（携带 run_id，供 DELETE /api/v1/runs/{run_id} 取消）；
        - Runner 层 span + TTFT（首 Token 时延）指标 + token 时延指标；
        - 取消传播：cancel_event 置位后停止消费并 drain，emit run_cancelled。
        """
        task = self._task_from_messages(messages)
        # Runner 层埋点 + Invocation 上下文
        owns_request_registration = invocation is None
        inv = invocation or Invocation(
                user_id=self.agent_config.user_id,
                agent_name=self.agent_config.name or "GeneralAgent",
                model_name=getattr(self.conversation_model, "model_name", "")
                or getattr(self.conversation_model, "model", "") or "",
                max_llm_calls=MAX_LLM_CALLS,
                max_tool_iterations=MAX_TOOL_ITERATIONS,
            )
        inv.context.setdefault("delegation_stack", (self.agent_capability_id,))
        inv.context.setdefault(
            "delegation_depth_limit",
            self._delegation_depth_limit,
        )
        self.current_invocation = inv
        if owns_request_registration:
            request_registry.register(inv.invocation_id, owner=inv.user_id)
            request_registry.bind(inv.invocation_id, inv)

        token = set_current_invocation(inv)
        runner_span = start_runner_span(inv, inv.agent_name)
        ttft = TTFTTracker()
        response_content = ""
        last_chunk_at = time.perf_counter()

        try:
            # 首事件：run_started（客户端据此获得 run_id 以便取消）
            yield {
                "type": EventKind.RUN_STARTED.value,
                "timestamp": time.time(),
                "data": {"run_id": inv.invocation_id, "agent": inv.agent_name},
            }

            with metrics.track_active_run():
                intent_decision = await self._route_intent(
                    messages,
                    task,
                    invocation=inv,
                    inherited_tool_permissions=inherited_tool_permissions,
                )
                self.last_intent_decision = intent_decision
                capability_resolution = self.resolve_capabilities_for_task(
                    intent_decision.normalized_task,
                    selected_skill_ids=intent_decision.skill_ids,
                    inherited_tool_permissions=inherited_tool_permissions,
                )
                capability_resolution = self._apply_intent_route(
                    capability_resolution,
                    intent_decision,
                )
                self.last_capability_resolution = capability_resolution
                prepared_messages = self._prepare_capability_messages(
                    messages,
                    capability_resolution,
                )
                prepared_messages = self._prepare_intent_messages(
                    prepared_messages,
                    intent_decision,
                )
                selected_tool_objects = list(
                    capability_resolution.selected_tool_objects
                )
                authorized_tool_names = [
                    item.name for item in capability_resolution.selected_tools
                ]
                inv.context["intent_decision"] = intent_decision.model_dump()
                inv.context["effective_tool_permissions"] = capability_aliases(
                    capability_resolution.allowed_tools
                )
                runner_span.set_attribute(
                    "agchat.intent.route",
                    intent_decision.route,
                )
                runner_span.set_attribute(
                    "agchat.intent.confidence",
                    intent_decision.confidence,
                )
                yield {
                    "type": EventKind.NOTICE.value,
                    "timestamp": time.time(),
                    "data": {
                        "stage": "intent_routing",
                        "route": intent_decision.route,
                        "confidence": intent_decision.confidence,
                        "source": self.last_intent_routing.get("source", ""),
                        "routing_duration_ms": self.last_intent_routing.get(
                            "duration_ms",
                            0.0,
                        ),
                        "selected_skill_ids": intent_decision.skill_ids,
                        "selected_agent_id": intent_decision.agent_id,
                    },
                }

                async for chunk_event, metadata in self.react_agent.astream(
                        input={
                            "messages": prepared_messages,
                            "model_call_count": 0,
                            "user_id": inv.user_id,
                            "available_tools": selected_tool_objects,
                            "authorized_tool_names": authorized_tool_names,
                            "selected_skill_ids": [
                                skill.skill_id
                                for skill in capability_resolution.selected_skills
                            ],
                            "intent_route": intent_decision.route,
                        },
                        config={"callbacks": [usage_metadata_callback]},
                        stream_mode=["messages", "custom"],
                ):
                    if inv.cancelled:
                        break   # 取消传播：停止消费事件流

                    if chunk_event == "custom":
                        yield self.wrap_event(metadata)
                    elif isinstance(metadata[0], AIMessageChunk) and metadata[0].content:
                        content = metadata[0].content
                        response_content += content
                        # TTFT：首 Token 时延（核心 KPI，目标 < 800ms）
                        if (first := ttft.mark_first_token()) is not None:
                            metrics.record_ttft(first, model=inv.model_name, agent=inv.agent_name)
                            runner_span.set_attribute("agchat.ttft_ms", round(first, 3))
                        else:
                            metrics.record_token_latency(
                                (time.perf_counter() - last_chunk_at) * 1000.0,
                                model=inv.model_name,
                            )
                        last_chunk_at = time.perf_counter()
                        yield {
                            "type": "response_chunk",
                            "timestamp": time.time(),
                            "data": {
                                "chunk": content,
                                "accumulated": response_content
                            }
                        }

            if response_content:
                metrics.record_token_usage(
                    model=inv.model_name,
                    output_tokens=count_tokens_usage(response_content),
                )

            yield {
                "type": EventKind.RUN_CANCELLED.value if inv.cancelled else EventKind.RUN_FINISHED.value,
                "timestamp": time.time(),
                "data": {
                    "run_id": inv.invocation_id,
                    "llm_calls": inv.llm_call_count,
                    "tool_iterations": inv.tool_iter_count,
                },
            }

        # 配额超限：emit 限流事件后终止（对齐调用计数限流中间件行为）
        except QuotaExceeded as err:
            logger.warning(f"Quota exceeded: {err}")
            yield {
                "type": EventKind.ERROR.value,
                "timestamp": time.time(),
                "data": {"error": "QuotaExceeded", "message": str(err)},
            }
        # 针对模型回复进行兜底操作，错误类型包括：敏感词，模型问题
        except Exception as err:
            logger.error(f"LLM Model Error: {err}")
            yield {
                "type": "response_chunk",
                "timestamp": time.time(),
                "data": {
                    "chunk": "您的问题触及到我的知识盲区，请换个问题吧✨",
                    "accumulated": response_content
                }
            }
        finally:
            runner_span.set_attribute("agchat.llm_calls", inv.llm_call_count)
            runner_span.set_attribute("agchat.tool_iterations", inv.tool_iter_count)
            runner_span.end()
            current_invocation.reset(token)
            if owns_request_registration:
                request_registry.unregister(inv.invocation_id)
            self.current_invocation = None

    def stop_streaming_callback(self):
        """客户端断连 / 主动停止：通过 cancel_event 级联取消整棵子任务树。"""
        self.stop_streaming = True
        if self.current_invocation is not None:
            self.current_invocation.cancel()

    def get_tool_display_name(self, tool_name: str):
        """
        根据工具的原始名称，解析出带有类型后缀的展示名称
        例如:
        - "gaode_weather" -> "执行Skill：高德天气"
        - "mcp_filesystem" -> "执行MCP：文件系统"
        - "search" -> "执行工具：search"
        """
        metadata = self.tool_metadata_map.get(tool_name)

        if not metadata:
            # 如果没有记录元数据，直接返回原始名称
            return "工具", tool_name

        friendly_name = metadata.get("name", tool_name)
        tool_type = metadata.get("type", "工具")

        return tool_type, friendly_name
