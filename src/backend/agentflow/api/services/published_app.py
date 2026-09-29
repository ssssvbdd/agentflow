import asyncio
import hashlib
import hmac
import json
import re
import secrets
from typing import Optional

from agentflow.api.services.agent import AgentService
from agentflow.database.dao.published_app import (
    AppApiKeyDao,
    AppVersionDao,
    AuditLogDao,
    PublishedAppDao,
)
from agentflow.database.models.published_app import (
    AppApiKeyTable,
    AppVersionTable,
    AuditLogTable,
    PublishedAppTable,
)
from agentflow.database.models.user import AdminUser
from agentflow.schemas.published_app import PublishAppReq, UpdateAppReq


class PublishedAppError(ValueError):
    status_code = 400


class AppNotFoundError(PublishedAppError):
    status_code = 404


class AppPermissionError(PublishedAppError):
    status_code = 403


class AppConflictError(PublishedAppError):
    status_code = 409


def _hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def _snapshot_hash(snapshot: dict) -> str:
    canonical = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-").lower()
    return value or secrets.token_hex(4)


def _safe_key(key: AppApiKeyTable) -> dict:
    return key.to_dict(hide_fields=["key_hash"])


def _version_data(version: AppVersionTable, *, include_snapshot: bool = False) -> dict:
    hidden = [] if include_snapshot else ["config_snapshot"]
    return version.to_dict(hide_fields=hidden)


class PublishedAppService:
    ACTIVE = "active"
    PAUSED = "paused"

    @classmethod
    async def _agent_snapshot(cls, agent_id: str, owner_id: str) -> dict:
        await AgentService.verify_user_permission(agent_id, owner_id)
        snapshot = await AgentService.select_agent_by_id(agent_id)
        if not snapshot:
            raise AppNotFoundError("agent not found")
        return snapshot

    @classmethod
    async def publish(cls, req: PublishAppReq, owner_id: str) -> dict:
        snapshot = await cls._agent_snapshot(req.agent_id, owner_id)
        slug = _slugify(req.slug or req.name)
        if await PublishedAppDao.get_by_slug(slug):
            raise AppConflictError("app slug already exists")

        app = PublishedAppTable(
            agent_id=req.agent_id,
            owner_id=owner_id,
            name=req.name.strip(),
            slug=slug,
            description=req.description.strip(),
            api_enabled=req.api_enabled,
            web_enabled=req.web_enabled,
        )
        version = AppVersionTable(
            app_id=app.app_id,
            agent_id=req.agent_id,
            version=1,
            config_snapshot=snapshot,
            config_hash=_snapshot_hash(snapshot),
            created_by=owner_id,
        )
        created, created_version = await PublishedAppDao.create_with_version(app, version)
        await cls.audit(
            owner_id=owner_id,
            actor_id=owner_id,
            app_id=created.app_id,
            agent_id=req.agent_id,
            action="publish_app",
            detail={"slug": slug, "version": created_version.version},
        )
        data = created.to_dict()
        data["current_version"] = created_version.version
        data["current_version_id"] = created_version.version_id
        return data

    @classmethod
    async def list_apps(cls, owner_id: str) -> list[dict]:
        apps = list(await PublishedAppDao.list_by_owner(owner_id))
        versions = await asyncio.gather(*(AppVersionDao.get_latest(app.app_id) for app in apps))
        result = []
        for app, version in zip(apps, versions):
            data = app.to_dict()
            data["current_version"] = version.version if version else None
            data["current_version_id"] = version.version_id if version else None
            result.append(data)
        return result

    @classmethod
    async def get_app_detail(cls, app_id: str, owner_id: str) -> dict:
        app = await cls.get_owned_app(app_id, owner_id)
        keys, versions = await asyncio.gather(
            AppApiKeyDao.list_by_app(app_id),
            AppVersionDao.list_by_app(app_id),
        )
        data = app.to_dict()
        data["api_keys"] = [_safe_key(key) for key in keys]
        data["versions"] = [_version_data(version) for version in versions]
        if versions:
            data["runtime_snapshot"] = versions[0].config_snapshot
            data["current_version"] = versions[0].version
            data["current_version_id"] = versions[0].version_id
        return data

    @classmethod
    async def update_app(cls, app_id: str, owner_id: str, req: UpdateAppReq) -> dict:
        app = await cls.get_owned_app(app_id, owner_id)
        values = req.model_dump(exclude_none=True)
        if "name" in values:
            values["name"] = values["name"].strip()
        if "description" in values:
            values["description"] = values["description"].strip()
        if "slug" in values:
            slug = _slugify(values["slug"])
            existing = await PublishedAppDao.get_by_slug(slug)
            if existing and existing.app_id != app_id:
                raise AppConflictError("app slug already exists")
            values["slug"] = slug
        if not values:
            return app.to_dict()

        updated = await PublishedAppDao.update(app_id, values)
        if updated is None:
            raise AppNotFoundError("app not found")
        await cls.audit(
            owner_id=owner_id,
            actor_id=owner_id,
            app_id=app_id,
            agent_id=app.agent_id,
            action="update_app",
            detail={"fields": sorted(values)},
        )
        return updated.to_dict()

    @classmethod
    async def set_status(cls, app_id: str, owner_id: str, status: str) -> dict:
        app = await cls.get_owned_app(app_id, owner_id)
        if status not in {cls.ACTIVE, cls.PAUSED}:
            raise PublishedAppError("invalid app status")
        await PublishedAppDao.update_status(app.app_id, status)
        await cls.audit(
            owner_id=owner_id,
            actor_id=owner_id,
            app_id=app_id,
            agent_id=app.agent_id,
            action="resume_app" if status == cls.ACTIVE else "pause_app",
        )
        return {"app_id": app_id, "status": status}

    @classmethod
    async def pause(cls, app_id: str, owner_id: str) -> None:
        await cls.set_status(app_id, owner_id, cls.PAUSED)

    @classmethod
    async def republish(cls, app_id: str, owner_id: str) -> dict:
        app = await cls.get_owned_app(app_id, owner_id)
        snapshot = await cls._agent_snapshot(app.agent_id, owner_id)
        version = await AppVersionDao.create_next(
            app_id=app.app_id,
            agent_id=app.agent_id,
            config_snapshot=snapshot,
            config_hash=_snapshot_hash(snapshot),
            created_by=owner_id,
        )
        await cls.audit(
            owner_id=owner_id,
            actor_id=owner_id,
            app_id=app_id,
            agent_id=app.agent_id,
            action="republish_app",
            detail={"version": version.version, "config_hash": version.config_hash},
        )
        return _version_data(version, include_snapshot=True)

    @classmethod
    async def list_versions(cls, app_id: str, owner_id: str) -> list[dict]:
        await cls.get_owned_app(app_id, owner_id)
        versions = await AppVersionDao.list_by_app(app_id)
        return [_version_data(version) for version in versions]

    @classmethod
    async def rollback(cls, app_id: str, version_id: str, owner_id: str) -> dict:
        app = await cls.get_owned_app(app_id, owner_id)
        source = await AppVersionDao.get_by_id(version_id)
        if source is None or source.app_id != app_id:
            raise AppNotFoundError("app version not found")
        version = await AppVersionDao.create_next(
            app_id=app.app_id,
            agent_id=source.agent_id,
            config_snapshot=source.config_snapshot,
            config_hash=source.config_hash,
            created_by=owner_id,
        )
        await cls.audit(
            owner_id=owner_id,
            actor_id=owner_id,
            app_id=app_id,
            agent_id=app.agent_id,
            action="rollback_app",
            detail={"source_version": source.version, "new_version": version.version},
        )
        return _version_data(version, include_snapshot=True)

    @classmethod
    def _new_api_key(cls, app_id: str, owner_id: str, name: str) -> tuple[str, AppApiKeyTable]:
        raw = f"agc_{secrets.token_urlsafe(32)}"
        return raw, AppApiKeyTable(
            app_id=app_id,
            owner_id=owner_id,
            name=name.strip() or "default",
            key_prefix=raw[:12],
            key_hash=_hash_key(raw),
        )

    @classmethod
    async def create_api_key(cls, app_id: str, owner_id: str, name: str = "default") -> dict:
        app = await cls.get_owned_app(app_id, owner_id)
        raw, key = cls._new_api_key(app_id, owner_id, name)
        created = await AppApiKeyDao.create(key)
        await cls.audit(
            owner_id=owner_id,
            actor_id=owner_id,
            app_id=app_id,
            agent_id=app.agent_id,
            action="create_api_key",
            detail={"key_id": created.key_id, "key_prefix": created.key_prefix},
        )
        data = _safe_key(created)
        data["api_key"] = raw
        return data

    @classmethod
    async def list_api_keys(cls, app_id: str, owner_id: str) -> list[dict]:
        await cls.get_owned_app(app_id, owner_id)
        keys = await AppApiKeyDao.list_by_app(app_id)
        return [_safe_key(key) for key in keys]

    @classmethod
    async def revoke_api_key(cls, app_id: str, key_id: str, owner_id: str) -> None:
        app = await cls.get_owned_app(app_id, owner_id)
        key = await AppApiKeyDao.get_by_id(key_id)
        if key is None or key.app_id != app_id:
            raise AppNotFoundError("api key not found")
        if not key.enabled:
            return
        await AppApiKeyDao.disable(key_id)
        await cls.audit(
            owner_id=owner_id,
            actor_id=owner_id,
            app_id=app_id,
            agent_id=app.agent_id,
            action="revoke_api_key",
            detail={"key_id": key_id, "key_prefix": key.key_prefix},
        )

    @classmethod
    async def rotate_api_key(cls, app_id: str, key_id: str, owner_id: str) -> dict:
        app = await cls.get_owned_app(app_id, owner_id)
        old_key = await AppApiKeyDao.get_by_id(key_id)
        if old_key is None or old_key.app_id != app_id or not old_key.enabled:
            raise AppNotFoundError("api key not found or already revoked")
        raw, replacement = cls._new_api_key(app_id, owner_id, old_key.name)
        created = await AppApiKeyDao.rotate(key_id, replacement)
        await cls.audit(
            owner_id=owner_id,
            actor_id=owner_id,
            app_id=app_id,
            agent_id=app.agent_id,
            action="rotate_api_key",
            detail={
                "old_key_id": key_id,
                "new_key_id": created.key_id,
                "new_key_prefix": created.key_prefix,
            },
        )
        data = _safe_key(created)
        data["api_key"] = raw
        return data

    @classmethod
    async def authenticate_key(cls, api_key: str) -> tuple[PublishedAppTable, AppApiKeyTable]:
        prefix = api_key[:12]
        key = await AppApiKeyDao.get_by_prefix(prefix)
        if key is None or not hmac.compare_digest(key.key_hash, _hash_key(api_key)):
            raise PublishedAppError("invalid api key")
        app = await PublishedAppDao.get_by_id(key.app_id)
        if app is None or app.status != cls.ACTIVE or not app.api_enabled:
            raise PublishedAppError("app is not active")
        await AppApiKeyDao.touch(key.key_id)
        return app, key

    @classmethod
    async def resolve_runtime_config(cls, app: PublishedAppTable) -> tuple[dict, Optional[AppVersionTable]]:
        version = await AppVersionDao.get_latest(app.app_id)
        if version is not None:
            return version.config_snapshot, version
        fallback = await AgentService.select_agent_by_id(app.agent_id)
        if not fallback:
            raise AppNotFoundError("agent not found")
        return fallback, None

    @classmethod
    async def get_owned_app(cls, app_id: str, owner_id: str) -> PublishedAppTable:
        app = await PublishedAppDao.get_by_id(app_id)
        if app is None:
            raise AppNotFoundError("app not found")
        if owner_id != AdminUser and app.owner_id != owner_id:
            raise AppPermissionError("no permission")
        return app

    @classmethod
    async def list_audit_logs(
        cls,
        app_id: str,
        owner_id: str,
        *,
        action: str = "",
        status: str = "",
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        await cls.get_owned_app(app_id, owner_id)
        logs = await AuditLogDao.list_by_owner(
            owner_id,
            app_id=app_id,
            action=action,
            status=status,
            limit=max(1, min(limit, 200)),
            offset=max(0, offset),
        )
        result = []
        for log in logs:
            data = log.to_dict()
            try:
                data["detail"] = json.loads(log.detail or "{}")
            except json.JSONDecodeError:
                data["detail"] = {}
            result.append(data)
        return result

    @classmethod
    async def audit(
        cls,
        *,
        owner_id: str,
        actor_id: str,
        action: str,
        app_id: str = "",
        agent_id: str = "",
        trace_id: str = "",
        status: str = "success",
        ip: str = "",
        user_agent: str = "",
        detail: Optional[dict] = None,
    ) -> None:
        await AuditLogDao.create(
            AuditLogTable(
                owner_id=owner_id,
                actor_id=actor_id,
                action=action,
                app_id=app_id,
                agent_id=agent_id,
                trace_id=trace_id,
                status=status,
                ip=ip,
                user_agent=user_agent,
                detail=json.dumps(detail or {}, ensure_ascii=False),
            )
        )
