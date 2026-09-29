from typing import List, Optional
from pydantic import BaseModel, Field

class AgentCreateReq(BaseModel):
    name: str = Field(..., description="Agent 名称")
    description: str = Field(..., description="Agent 描述")
    tool_ids: List[str] = Field(default=[], description="绑定的工具ID")
    llm_id: Optional[str] = Field(None, description="Agent 绑定的LLM ID")
    mcp_ids: List[str] = Field(default=[], description="绑定的MCP Server")
    knowledge_ids: List[str] = Field(default=[], description="绑定的知识库ID")
    agent_skill_ids: List[str] = Field(default=[], description="绑定的技能")
    sub_agent_ids: List[str] = Field(default=[], description="允许委派的子 Agent ID")
    enable_intent_router: bool = Field(True, description="是否启用结构化意图路由")
    intent_confidence_threshold: float = Field(0.65, ge=0, le=1, description="意图路由最低置信度")
    intent_history_messages: int = Field(12, ge=1, le=50, description="意图路由读取的最近消息数")
    enable_semantic_intent_retrieval: bool = Field(True, description="是否启用语义候选召回")
    intent_small_tool_catalog_size: int = Field(8, ge=0, le=50, description="跳过意图模型的小工具集上限")
    intent_deterministic_score_threshold: float = Field(0.82, ge=0, le=1, description="确定性路由分数阈值")
    intent_score_margin_threshold: float = Field(0.20, ge=0, le=1, description="确定性路由候选分差阈值")
    intent_candidate_limit: int = Field(12, ge=3, le=50, description="进入意图模型的候选上限")
    intent_semantic_timeout_seconds: float = Field(3.0, ge=0.1, le=30, description="语义召回超时秒数")
    enable_memory: bool = Field(True, description="是否使用嵌入")
    system_prompt: str = Field(..., description="Agent 系统提示词")
    logo_url: str = Field(..., description="Logo URL")


class AgentUpdateReq(BaseModel):
    agent_id: str = Field(..., description="要更新的 Agent ID")
    name: Optional[str] = Field(None, description="Agent 名称")
    description: Optional[str] = Field(None, description="Agent 描述")
    tool_ids: Optional[List[str]] = Field(None, description="绑定的工具ID")
    knowledge_ids: Optional[List[str]] = Field(None, description="绑定的知识库ID")
    mcp_ids: Optional[List[str]] = Field(None, description="绑定的MCP Server")
    llm_id: Optional[str] = Field(None, description="Agent 绑定的LLM ID")
    agent_skill_ids: List[str] = Field(default=[], description="绑定的技能")
    sub_agent_ids: Optional[List[str]] = Field(None, description="允许委派的子 Agent ID")
    enable_intent_router: Optional[bool] = Field(None, description="是否启用结构化意图路由")
    intent_confidence_threshold: Optional[float] = Field(None, ge=0, le=1, description="意图路由最低置信度")
    intent_history_messages: Optional[int] = Field(None, ge=1, le=50, description="意图路由读取的最近消息数")
    enable_semantic_intent_retrieval: Optional[bool] = Field(None, description="是否启用语义候选召回")
    intent_small_tool_catalog_size: Optional[int] = Field(None, ge=0, le=50, description="跳过意图模型的小工具集上限")
    intent_deterministic_score_threshold: Optional[float] = Field(None, ge=0, le=1, description="确定性路由分数阈值")
    intent_score_margin_threshold: Optional[float] = Field(None, ge=0, le=1, description="确定性路由候选分差阈值")
    intent_candidate_limit: Optional[int] = Field(None, ge=3, le=50, description="进入意图模型的候选上限")
    intent_semantic_timeout_seconds: Optional[float] = Field(None, ge=0.1, le=30, description="语义召回超时秒数")
    enable_memory: Optional[bool] = Field(True, description="是否使用嵌入")
    logo_url: Optional[str] = Field(None, description="Logo URL")
    system_prompt: str = Field(None, description="Agent 系统提示词")


class AgentDeleteReq(BaseModel):
    agent_id: str

class AgentSearchReq(BaseModel):
    name: str
