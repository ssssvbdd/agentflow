from sqlmodel import Field, SQLModel
from typing import Literal, Optional, List
from datetime import datetime
from uuid import uuid4
from sqlalchemy import JSON, Column, text, DateTime

from agentflow.settings import app_settings
from agentflow.database.models.base import SQLModelSerializable


class AgentTable(SQLModelSerializable, table=True):
    __tablename__ = "agent"

    id: str = Field(default_factory=lambda: uuid4().hex, primary_key=True)
    name: str = Field(default="", description="Agent 的名称")
    description: str = Field(default="", description="Agent 的描述")
    logo_url: str = Field(default=app_settings.default_config.get("agent_logo_url"))
    user_id: Optional[str] = Field(index=True, description="Agent绑定的用户ID")
    is_custom: bool = Field(default=True, description="Agent是否为用户自定义")
    system_prompt: str = Field(default="", description="Agent设定的系统提示词")
    llm_id: str = Field(default="", description="Agent绑定的LLM模型")
    enable_memory: bool = Field(default=True, description="是否开启记忆功能")
    mcp_ids: List[str] = Field(default=[], sa_column=Column(JSON), description="Agent绑定的MCP Server")
    tool_ids: List[str] = Field(default=[], sa_column=Column(JSON), description="Agent绑定的工具列表")
    agent_skill_ids: List[str] = Field(default=[], sa_column=Column(JSON), description="Agent绑定的技能")
    sub_agent_ids: List[str] = Field(default=[], sa_column=Column(JSON), description="允许委派的子 Agent")
    enable_intent_router: bool = Field(default=True, description="是否启用结构化意图路由")
    intent_confidence_threshold: float = Field(default=0.65, description="意图路由最低置信度")
    intent_history_messages: int = Field(default=12, description="意图路由读取的最近消息数")
    enable_semantic_intent_retrieval: bool = Field(default=True, description="是否启用语义候选召回")
    intent_small_tool_catalog_size: int = Field(default=8, description="跳过意图模型的小工具集上限")
    intent_deterministic_score_threshold: float = Field(default=0.82, description="确定性路由分数阈值")
    intent_score_margin_threshold: float = Field(default=0.20, description="确定性路由候选分差阈值")
    intent_candidate_limit: int = Field(default=12, description="进入意图模型的候选上限")
    intent_semantic_timeout_seconds: float = Field(default=3.0, description="语义召回超时秒数")
    knowledge_ids: List[str] = Field(default=[], sa_column=Column(JSON), description="Agent 绑定的知识库")

    # 修改时间，默认为当前时间戳，自动更新
    update_time: Optional[datetime] = Field(
        sa_column=Column(
            DateTime,
            nullable=False,
            server_default=text('CURRENT_TIMESTAMP'),
            onupdate=text('CURRENT_TIMESTAMP')
        ),
        description="修改时间"
    )

    # 创建时间，默认为当前时间戳
    create_time: Optional[datetime] = Field(
        sa_column=Column(
            DateTime,
            nullable=False,
            server_default=text('CURRENT_TIMESTAMP')
        ),
        description="创建时间"
    )
