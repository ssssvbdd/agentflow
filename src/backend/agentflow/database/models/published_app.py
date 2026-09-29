from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import JSON, Column, DateTime, Integer, Text, UniqueConstraint, text
from sqlmodel import Field

from agentflow.database.models.base import SQLModelSerializable


class PublishedAppTable(SQLModelSerializable, table=True):
    __tablename__ = "published_app"

    app_id: str = Field(default_factory=lambda: uuid4().hex, primary_key=True)
    agent_id: str = Field(index=True, description="Published agent id")
    owner_id: str = Field(index=True, description="Owner user id")
    name: str = Field(default="", description="Published app name")
    slug: str = Field(index=True, unique=True, description="Public app slug")
    description: str = Field(default="", sa_column=Column(Text))
    status: str = Field(default="active", index=True, description="active/paused")
    api_enabled: bool = Field(default=True)
    web_enabled: bool = Field(default=True)
    update_time: Optional[datetime] = Field(
        sa_column=Column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"), onupdate=text("CURRENT_TIMESTAMP"))
    )
    create_time: Optional[datetime] = Field(
        sa_column=Column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"))
    )


class AppVersionTable(SQLModelSerializable, table=True):
    """Immutable runtime snapshot created whenever an app is published."""

    __tablename__ = "app_version"
    __table_args__ = (
        UniqueConstraint("app_id", "version", name="uq_app_version_app_version"),
    )

    version_id: str = Field(default_factory=lambda: uuid4().hex, primary_key=True)
    app_id: str = Field(index=True)
    agent_id: str = Field(index=True)
    version: int = Field(sa_column=Column(Integer, nullable=False))
    config_snapshot: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    config_hash: str = Field(index=True, description="SHA256 of the canonical config snapshot")
    created_by: str = Field(default="", index=True)
    create_time: Optional[datetime] = Field(
        sa_column=Column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"))
    )


class AppApiKeyTable(SQLModelSerializable, table=True):
    __tablename__ = "app_api_key"

    key_id: str = Field(default_factory=lambda: uuid4().hex, primary_key=True)
    app_id: str = Field(index=True)
    owner_id: str = Field(index=True)
    name: str = Field(default="default")
    key_prefix: str = Field(index=True, description="Non-secret prefix for lookup")
    key_hash: str = Field(description="SHA256 hash of the full API key")
    enabled: bool = Field(default=True, index=True)
    last_used_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime, nullable=True))
    create_time: Optional[datetime] = Field(
        sa_column=Column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"))
    )


class AuditLogTable(SQLModelSerializable, table=True):
    __tablename__ = "audit_log"

    audit_id: str = Field(default_factory=lambda: uuid4().hex, primary_key=True)
    owner_id: str = Field(default="", index=True)
    actor_id: str = Field(default="", index=True)
    app_id: str = Field(default="", index=True)
    agent_id: str = Field(default="", index=True)
    action: str = Field(index=True, description="publish/call/create_key/etc")
    trace_id: str = Field(default="", index=True)
    status: str = Field(default="success", index=True)
    ip: str = Field(default="")
    user_agent: str = Field(default="", sa_column=Column(Text))
    detail: str = Field(default="", sa_column=Column(Text))
    create_time: Optional[datetime] = Field(
        sa_column=Column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"))
    )
