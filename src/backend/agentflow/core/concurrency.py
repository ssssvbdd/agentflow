"""
结构化并发协调层（对齐 trpc-agent-go ParallelAgent / Runner 的并发模型）。

从裸 asyncio.create_task 升级为：
1. TaskGroup 结构化并发 —— 任一失败自动取消其余 + 传播（对齐 goroutine errgroup）；
2. 有界队列背压 —— merged queue maxsize=256，队列满则自然背压，防内存爆炸；
3. 异常隔离 —— 单个子流崩溃转为 ErrorEvent，不拖垮整体（对齐 ParallelAgent 的 recover()）；
4. 取消传播 —— RequestRegistry 绑定 request_id -> asyncio.Event，
   中断必须靠 cancel_event，不能只 break 事件循环（否则后台任务会阻塞在 queue.put）。
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator, Dict, List, Optional, Sequence

from agentflow.core.contracts import Agent, Event, EventKind, error_event
from agentflow.core.invocation import Invocation

DEFAULT_BACKPRESSURE = 256


async def run_parallel(
    inv: Invocation,
    sub_agents: Sequence[Agent],
    backpressure: int = DEFAULT_BACKPRESSURE,
) -> AsyncIterator[Event]:
    """并行执行多个子 Agent，事件流归并输出（分支隔离 + 异常隔离 + 背压）。

    使用注意：这是异步生成器，内部通过 TaskGroup 保证所有 pump 任务
    在生成器关闭/耗尽时一定被 join，不产生任务泄漏。
    """
    merged: asyncio.Queue[Optional[Event]] = asyncio.Queue(maxsize=backpressure)
    active = len(sub_agents)

    async def pump(agent: Agent) -> None:
        branch = f"{inv.branch}.parallel[{agent.info().name}]"
        try:
            async for ev in agent.run(inv.clone(branch=branch)):
                await merged.put(ev)   # 队列满则背压
        except asyncio.CancelledError:
            raise                      # 取消必须向上传播，交给 TaskGroup 级联
        except Exception as err:       # 隔离：单子流异常不影响其他
            await merged.put(error_event(agent.info().name, err, inv.invocation_id, branch))
        finally:
            await merged.put(None)     # 该子流结束信号

    finished = 0
    tg = asyncio.TaskGroup()
    try:
        async with tg:
            for agent in sub_agents:
                tg.create_task(pump(agent))
            # 消费 merged 直到所有 pump 结束（或被取消）
            while finished < active:
                if inv.cancelled:
                    # 取消后继续 drain 直到所有子流关闭，避免 SSE 连接挂起
                    pass
                item = await merged.get()
                if item is None:
                    finished += 1
                    continue
                yield item
    finally:
        if inv.cancelled:
            yield Event(
                kind=EventKind.RUN_CANCELLED,
                invocation_id=inv.invocation_id,
                branch=inv.branch,
                data={"reason": "cancelled", "completed": finished, "total": active},
            )


class RequestRegistry:
    """request_id -> CancelToken 注册表（对齐 Runner 的 Cancel(requestID)）。

    配合 HTTP DELETE /api/v1/runs/{run_id} 实现服务端主动取消。
    每个 run 绑定属主 user_id：取消操作强制校验 owner == current_user（防 IDOR）。
    """

    def __init__(self) -> None:
        self._tokens: Dict[str, asyncio.Event] = {}
        self._owners: Dict[str, str] = {}
        self._bound: Dict[str, List[Invocation]] = {}

    def register(self, run_id: str, owner: str = "") -> asyncio.Event:
        ev = asyncio.Event()
        self._tokens[run_id] = ev
        self._owners[run_id] = owner
        self._bound.setdefault(run_id, [])
        return ev

    def bind(self, run_id: str, inv: Invocation) -> Invocation:
        """将 run_id 的取消信号绑定到 Invocation：cancel(run_id) 即 inv.cancel()。

        直接持有引用而非 watcher 任务：取消是同步、确定性的，
        且注册表可在任何线程/上下文中使用（不依赖运行中的事件循环）。
        """
        self._bound.setdefault(run_id, []).append(inv)
        return inv

    def cancel(self, run_id: str, owner: Optional[str] = None) -> bool:
        """取消指定 run；owner 提供时强制校验属主（防水平越权）。返回是否存在且有权取消。"""
        ev = self._tokens.get(run_id)
        if ev is None:
            return False
        if owner is not None and self._owners.get(run_id, "") != owner:
            return False
        ev.set()
        for inv in self._bound.get(run_id, []):
            inv.cancel()
        return True

    def cancel_all(self) -> int:
        """优雅退出用：广播取消所有活跃 run（cancel_event 级联到整棵子任务树）。"""
        count = 0
        for run_id, ev in self._tokens.items():
            if not ev.is_set():
                ev.set()
                count += 1
            for inv in self._bound.get(run_id, []):
                inv.cancel()
        return count

    def unregister(self, run_id: str) -> None:
        self._tokens.pop(run_id, None)
        self._owners.pop(run_id, None)
        self._bound.pop(run_id, None)

    @property
    def active_runs(self) -> int:
        return len(self._tokens)

    def snapshot(self) -> Dict[str, Dict[str, object]]:
        return {
            rid: {"cancelled": ev.is_set(), "owner": self._owners.get(rid, "")}
            for rid, ev in self._tokens.items()
        }


# 进程级单例：路由层 / 优雅退出均使用
request_registry = RequestRegistry()
