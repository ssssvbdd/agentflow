from datetime import datetime
from typing import Any, Optional

from sqlalchemy import func
from sqlmodel import select, update

from agentflow.database.models.published_app import (
    AppApiKeyTable,
    AppVersionTable,
    AuditLogTable,
    PublishedAppTable,
)
from agentflow.database.session import async_session_getter


class PublishedAppDao:
    @classmethod
    async def create(cls, app: PublishedAppTable) -> PublishedAppTable:
        async with async_session_getter() as session:
            session.add(app)
            await session.commit()
            await session.refresh(app)
            return app

    @classmethod
    async def create_with_version(
        cls,
        app: PublishedAppTable,
        version: AppVersionTable,
    ) -> tuple[PublishedAppTable, AppVersionTable]:
        async with async_session_getter() as session:
            session.add(app)
            session.add(version)
            await session.commit()
            await session.refresh(app)
            await session.refresh(version)
            return app, version

    @classmethod
    async def get_by_id(cls, app_id: str) -> Optional[PublishedAppTable]:
        async with async_session_getter() as session:
            return await session.get(PublishedAppTable, app_id)

    @classmethod
    async def get_by_slug(cls, slug: str) -> Optional[PublishedAppTable]:
        async with async_session_getter() as session:
            result = await session.exec(select(PublishedAppTable).where(PublishedAppTable.slug == slug))
            return result.first()

    @classmethod
    async def list_by_owner(cls, owner_id: str):
        async with async_session_getter() as session:
            result = await session.exec(
                select(PublishedAppTable)
                .where(PublishedAppTable.owner_id == owner_id)
                .order_by(PublishedAppTable.update_time.desc())
            )
            return result.all()

    @classmethod
    async def update_status(cls, app_id: str, status: str) -> None:
        async with async_session_getter() as session:
            await session.exec(update(PublishedAppTable).where(PublishedAppTable.app_id == app_id).values(status=status))
            await session.commit()

    @classmethod
    async def update(cls, app_id: str, values: dict[str, Any]) -> Optional[PublishedAppTable]:
        async with async_session_getter() as session:
            app = await session.get(PublishedAppTable, app_id)
            if app is None:
                return None
            for key, value in values.items():
                setattr(app, key, value)
            session.add(app)
            await session.commit()
            await session.refresh(app)
            return app


class AppVersionDao:
    @classmethod
    async def get_by_id(cls, version_id: str) -> Optional[AppVersionTable]:
        async with async_session_getter() as session:
            return await session.get(AppVersionTable, version_id)

    @classmethod
    async def get_latest(cls, app_id: str) -> Optional[AppVersionTable]:
        async with async_session_getter() as session:
            result = await session.exec(
                select(AppVersionTable)
                .where(AppVersionTable.app_id == app_id)
                .order_by(AppVersionTable.version.desc())
                .limit(1)
            )
            return result.first()

    @classmethod
    async def list_by_app(cls, app_id: str):
        async with async_session_getter() as session:
            result = await session.exec(
                select(AppVersionTable)
                .where(AppVersionTable.app_id == app_id)
                .order_by(AppVersionTable.version.desc())
            )
            return result.all()

    @classmethod
    async def create_next(
        cls,
        *,
        app_id: str,
        agent_id: str,
        config_snapshot: dict,
        config_hash: str,
        created_by: str,
    ) -> AppVersionTable:
        async with async_session_getter() as session:
            app = await session.get(PublishedAppTable, app_id, with_for_update=True)
            if app is None:
                raise ValueError("app not found")
            result = await session.exec(
                select(func.max(AppVersionTable.version)).where(AppVersionTable.app_id == app_id)
            )
            latest = result.one()
            version = AppVersionTable(
                app_id=app_id,
                agent_id=agent_id,
                version=int(latest or 0) + 1,
                config_snapshot=config_snapshot,
                config_hash=config_hash,
                created_by=created_by,
            )
            session.add(version)
            await session.commit()
            await session.refresh(version)
            return version


class AppApiKeyDao:
    @classmethod
    async def create(cls, key: AppApiKeyTable) -> AppApiKeyTable:
        async with async_session_getter() as session:
            session.add(key)
            await session.commit()
            await session.refresh(key)
            return key

    @classmethod
    async def get_by_prefix(cls, key_prefix: str) -> Optional[AppApiKeyTable]:
        async with async_session_getter() as session:
            result = await session.exec(
                select(AppApiKeyTable).where(
                    AppApiKeyTable.key_prefix == key_prefix,
                    AppApiKeyTable.enabled == True,  # noqa: E712
                )
            )
            return result.first()

    @classmethod
    async def get_by_id(cls, key_id: str) -> Optional[AppApiKeyTable]:
        async with async_session_getter() as session:
            return await session.get(AppApiKeyTable, key_id)

    @classmethod
    async def list_by_app(cls, app_id: str):
        async with async_session_getter() as session:
            result = await session.exec(
                select(AppApiKeyTable)
                .where(AppApiKeyTable.app_id == app_id)
                .order_by(AppApiKeyTable.create_time.desc())
            )
            return result.all()

    @classmethod
    async def disable(cls, key_id: str) -> bool:
        async with async_session_getter() as session:
            key = await session.get(AppApiKeyTable, key_id, with_for_update=True)
            if key is None:
                return False
            key.enabled = False
            session.add(key)
            await session.commit()
            return True

    @classmethod
    async def rotate(
        cls,
        old_key_id: str,
        replacement: AppApiKeyTable,
    ) -> AppApiKeyTable:
        async with async_session_getter() as session:
            old_key = await session.get(AppApiKeyTable, old_key_id, with_for_update=True)
            if old_key is None or not old_key.enabled:
                raise ValueError("api key not found or already revoked")
            old_key.enabled = False
            session.add(old_key)
            session.add(replacement)
            await session.commit()
            await session.refresh(replacement)
            return replacement

    @classmethod
    async def touch(cls, key_id: str) -> None:
        async with async_session_getter() as session:
            await session.exec(
                update(AppApiKeyTable)
                .where(AppApiKeyTable.key_id == key_id)
                .values(last_used_at=datetime.utcnow())
            )
            await session.commit()


class AuditLogDao:
    @classmethod
    async def create(cls, log: AuditLogTable) -> None:
        async with async_session_getter() as session:
            session.add(log)
            await session.commit()

    @classmethod
    async def list_by_owner(
        cls,
        owner_id: str,
        *,
        app_id: str = "",
        action: str = "",
        status: str = "",
        limit: int = 100,
        offset: int = 0,
    ):
        async with async_session_getter() as session:
            statement = select(AuditLogTable).where(AuditLogTable.owner_id == owner_id)
            if app_id:
                statement = statement.where(AuditLogTable.app_id == app_id)
            if action:
                statement = statement.where(AuditLogTable.action == action)
            if status:
                statement = statement.where(AuditLogTable.status == status)
            result = await session.exec(
                statement.order_by(AuditLogTable.create_time.desc()).offset(offset).limit(limit)
            )
            return result.all()
