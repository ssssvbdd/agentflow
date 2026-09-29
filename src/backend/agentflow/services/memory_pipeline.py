import asyncio
import json
from typing import Any, Optional

from loguru import logger

from agentflow.services.memory.client import memory_client
from agentflow.settings import app_settings


class MemoryPipeline:
    """Async memory extraction pipeline.

    Uses Kafka when aiokafka and config are available; otherwise falls back to an
    in-process queue so chat streaming never waits for memory extraction.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1000)
        self._worker: Optional[asyncio.Task] = None
        self._producer = None
        self._topic = "agentflow-memory-events"

    async def start(self) -> None:
        cfg = app_settings.kafka or {}
        self._topic = cfg.get("memory_topic", self._topic)
        if cfg.get("enabled"):
            try:
                from aiokafka import AIOKafkaProducer  # type: ignore

                self._producer = AIOKafkaProducer(bootstrap_servers=cfg.get("bootstrap_servers", "localhost:9092"))
                await self._producer.start()
                logger.info("Kafka memory producer started")
            except Exception as err:
                logger.warning(f"Kafka unavailable, memory pipeline falls back to local queue: {err}")
                self._producer = None
        if self._worker is None:
            self._worker = asyncio.create_task(self._consume_local_queue())

    async def stop(self) -> None:
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None
        if self._producer:
            await self._producer.stop()
            self._producer = None

    async def publish(self, *, run_id: str, messages: list[dict[str, str]], user_id: str = "", agent_id: str = "") -> None:
        event = {
            "run_id": run_id,
            "user_id": user_id,
            "agent_id": agent_id,
            "messages": messages,
        }
        if self._producer is not None:
            try:
                await self._producer.send_and_wait(
                    self._topic,
                    json.dumps(event, ensure_ascii=False).encode("utf-8"),
                )
                return
            except Exception as err:
                logger.warning(f"Kafka memory publish failed, fallback to local queue: {err}")
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("memory pipeline queue is full; drop memory event")

    async def _consume_local_queue(self) -> None:
        while True:
            event = await self._queue.get()
            try:
                await memory_client.add(
                    messages=event["messages"],
                    user_id=event.get("user_id") or None,
                    agent_id=event.get("agent_id") or None,
                    run_id=event.get("run_id") or None,
                )
            except Exception as err:
                logger.error(f"memory pipeline consume failed: {err}")
            finally:
                self._queue.task_done()


memory_pipeline = MemoryPipeline()
