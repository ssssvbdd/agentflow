from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class AppRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class PublishAppReq(AppRequest):
    agent_id: str = Field(description="Agent id to publish")
    name: str = Field(min_length=1, max_length=128, description="Published app name")
    slug: Optional[str] = Field(default=None, max_length=128, description="Public slug; generated from name when omitted")
    description: str = Field(default="", max_length=2000)
    api_enabled: bool = True
    web_enabled: bool = True


class UpdateAppReq(AppRequest):
    name: Optional[str] = Field(default=None, min_length=1, max_length=128)
    slug: Optional[str] = Field(default=None, min_length=1, max_length=128)
    description: Optional[str] = Field(default=None, max_length=2000)
    api_enabled: Optional[bool] = None
    web_enabled: Optional[bool] = None


class CreateApiKeyReq(AppRequest):
    app_id: str
    name: str = Field(default="default", min_length=1, max_length=128)


class PublicCompletionReq(AppRequest):
    user_input: str = Field(min_length=1, max_length=100_000)
    file_url: Optional[str] = None
