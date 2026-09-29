import json
from typing import Any, AsyncIterator, Optional


class RedisStreamEventStore:
    """Redis Stream-backed event store for SSE replay."""

    def __init__(self, redis_client=None, *, prefix: str = "agchat:events", maxlen: int = 2000):
        self.redis = redis_client
        self.prefix = prefix
        self.maxlen = maxlen

    def bind(self, redis_client) -> None:
        self.redis = redis_client

    def key(self, run_id: str) -> str:
        return f"{self.prefix}:{run_id}"

    async def append(self, run_id: str, owner: str, event: dict[str, Any]) -> Optional[str]:
        if self.redis is None or not run_id:
            return None
        payload = json.dumps(event, ensure_ascii=False)
        try:
            return await self.redis.xadd(
                self.key(run_id),
                {"owner": owner, "payload": payload},
                maxlen=self.maxlen,
                approximate=True,
            )
        except Exception:
            return None

    async def replay(self, run_id: str, owner: str, after_id: str = "0-0", count: int = 200) -> AsyncIterator[tuple[str, dict]]:
        if self.redis is None:
            return
        key = self.key(run_id)
        try:
            entries = await self.redis.xrange(key, min=after_id, max="+", count=count)
        except Exception:
            return
        for event_id, fields in entries:
            event_owner = fields.get("owner") if isinstance(fields, dict) else None
            payload = fields.get("payload") if isinstance(fields, dict) else None
            if isinstance(event_id, bytes):
                event_id = event_id.decode()
            if isinstance(event_owner, bytes):
                event_owner = event_owner.decode()
            if isinstance(payload, bytes):
                payload = payload.decode()
            if event_id == after_id:
                continue
            if event_owner != owner:
                continue
            try:
                yield event_id, json.loads(payload)
            except Exception:
                continue


event_store = RedisStreamEventStore()
