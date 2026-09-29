"""
Admission control for streaming chat runs.

This is the service-side pressure gate before expensive model/tool work starts:
- global in-flight limit protects the process event loop and downstream pools;
- per-user in-flight limit prevents one user from occupying all slots;
- short acquisition timeout returns fast under overload instead of piling up tasks.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Dict


class OverloadedError(Exception):
    """Raised when the server is already at its configured concurrency limit."""


@dataclass(frozen=True)
class AdmissionTicket:
    owner: str
    admitted_at: float


class AdmissionController:
    def __init__(
        self,
        max_active_runs: int = 100,
        max_active_runs_per_user: int = 20,
        acquire_timeout_seconds: float = 0.25,
    ) -> None:
        self._max_active_runs = max(1, max_active_runs)
        self._max_active_runs_per_user = max(1, max_active_runs_per_user)
        self._acquire_timeout_seconds = max(0.0, acquire_timeout_seconds)
        self._semaphore = asyncio.BoundedSemaphore(self._max_active_runs)
        self._lock = asyncio.Lock()
        self._active_by_owner: Dict[str, int] = {}
        self._active_total = 0

    def configure(
        self,
        *,
        max_active_runs: int | None = None,
        max_active_runs_per_user: int | None = None,
        acquire_timeout_seconds: float | None = None,
    ) -> None:
        """Configure at startup before requests are admitted."""
        if max_active_runs is not None:
            self._max_active_runs = max(1, int(max_active_runs))
            self._semaphore = asyncio.BoundedSemaphore(self._max_active_runs)
        if max_active_runs_per_user is not None:
            self._max_active_runs_per_user = max(1, int(max_active_runs_per_user))
        if acquire_timeout_seconds is not None:
            self._acquire_timeout_seconds = max(0.0, float(acquire_timeout_seconds))

    async def acquire(self, owner: str) -> AdmissionTicket:
        owner = owner or "anonymous"
        async with self._lock:
            if self._active_by_owner.get(owner, 0) >= self._max_active_runs_per_user:
                raise OverloadedError(f"too many active runs for owner={owner}")

        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self._acquire_timeout_seconds,
            )
        except asyncio.TimeoutError as err:
            raise OverloadedError("server is overloaded") from err

        async with self._lock:
            # Re-check after the global slot is acquired to avoid a race between
            # concurrent requests from the same owner.
            if self._active_by_owner.get(owner, 0) >= self._max_active_runs_per_user:
                self._semaphore.release()
                raise OverloadedError(f"too many active runs for owner={owner}")
            self._active_by_owner[owner] = self._active_by_owner.get(owner, 0) + 1
            self._active_total += 1
        return AdmissionTicket(owner=owner, admitted_at=time.time())

    async def release(self, ticket: AdmissionTicket | None) -> None:
        if ticket is None:
            return
        async with self._lock:
            current = self._active_by_owner.get(ticket.owner, 0)
            if current <= 1:
                self._active_by_owner.pop(ticket.owner, None)
            else:
                self._active_by_owner[ticket.owner] = current - 1
            if self._active_total > 0:
                self._active_total -= 1
        self._semaphore.release()

    def snapshot(self) -> dict:
        return {
            "active_total": self._active_total,
            "active_by_owner": dict(self._active_by_owner),
            "max_active_runs": self._max_active_runs,
            "max_active_runs_per_user": self._max_active_runs_per_user,
        }


admission_controller = AdmissionController()
