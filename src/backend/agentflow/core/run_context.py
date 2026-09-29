"""
运行上下文：ContextVar 形式传播当前 Invocation。

选择 ContextVar 而非 LangGraph state 传递的原因：
1. 避免 asyncio.Event 等不可序列化对象进入 state schema；
2. 中间件 / 工具 / 检索层任何深度都可零侵入读取（对齐 trpc-agent-go 的 ctx 传递）。
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Optional

from agentflow.core.invocation import Invocation

current_invocation: ContextVar[Optional[Invocation]] = ContextVar(
    "current_invocation", default=None
)


def set_current_invocation(inv: Optional[Invocation]):
    return current_invocation.set(inv)


def get_current_invocation() -> Optional[Invocation]:
    return current_invocation.get()
