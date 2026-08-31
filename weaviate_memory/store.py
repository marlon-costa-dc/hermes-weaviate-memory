"""Weaviate storage layer: schema, tenants, writes and hybrid search.

Everything that talks to Weaviate lives here. The provider in ``__init__.py``
owns policy (when to recall, what to capture); this module owns mechanism.

Two authorization modes are supported and probed at runtime:

``admin_list``
    Weaviate's legacy mode — one API key with broad rights. The provider
    creates collections and tenants on demand.

``rbac``
    Weaviate >= 1.29 role-based access control. The API key maps to a DB user
    whose role may be scoped to ``Mem*`` collections and a single tenant. Schema
    creation is typically NOT granted to that user, so the store degrades
    gracefully: it uses what exists and reports precisely what is missing
    instead of failing with a bare 403.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)

# Property names shared by every memory collection. Kept flat and explicit:
# a nested schema would force clients (Dify, MCP, other systems) to understand
# this plugin's internals to read its data.
PROPERTY_NAMES = (
    "content",
    "domain",
    "kind",
    "source",
    "actor",
    "sensitivity",
    "session_id",
    "tags",
    "created_at",
)


class StoreError(RuntimeError):
    """Raised when Weaviate cannot serve a request."""


class PermissionDeniedError(StoreError):
    """Raised when RBAC denies an operation.

    Distinct from StoreError so callers can tell "you may not do this" from
    "this is broken" — the first is a configuration answer, the second an
    incident.
    """


def _is_forbidden(exc: Exception) -> bool:
    """True when an exception carries a 403/authorization signal."""
    text = f"{type(exc).__name__}: {exc}".lower()
    return "403" in text or "forbidden" in text or "authorization" in text or "rbac" in text


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class WeaviateStore:
    """Schema management, writes and search over a set of memory collections."""

    def __init__(self, config, dims: int) -> None:
        self._config = config
        self._dims = dims
        self._client = None
        self._auth_mode = ""
        self._ensured: set[str] = set()
        self._readonly_schema = False

    # -- connection ---------------------------------------------------------

    @property
    def client(self):
        if self._client is None:
            raise StoreError("weaviate-memory store is not connected")
        return self._client

    @property
    def auth_mode(self) -> str:
        """The authorization mode in effect: ``rbac`` or ``admin_list``."""
        return self._auth_mode

    @property
    def readonly_schema(self) -> bool:
        """True when this identity may not create collections/tenants."""
        return self._readonly_schema

    def connect(self) -> None:
        import weaviate
        from weaviate.classes.init import AdditionalConfig, Auth, Timeout

        cfg = self._config
        auth = Auth.api_key(cfg.api_key) if cfg.api_key else None

        if cfg.use_grpc:
            self._client = weaviate.connect_to_custom(
                http_host=cfg.http_host,
                http_port=cfg.http_port,
                http_secure=cfg.http_secure,
                grpc_host=cfg.resolved_grpc_host(),
                grpc_port=cfg.grpc_port,
                grpc_secure=cfg.grpc_secure,
                auth_credentials=auth,
                additional_config=AdditionalConfig(timeout=Timeout(init=10, query=30, insert=60)),
            )
        else:
            # No gRPC route (common behind a plain HTTP ingress). skip_init_checks
            # stops the client probing a gRPC port that was never exposed.
            self._client = weaviate.connect_to_custom(
                http_host=cfg.http_host,
                http_port=cfg.http_port,
                http_secure=cfg.http_secure,
                grpc_host=cfg.resolved_grpc_host(),
                grpc_port=cfg.grpc_port,
                grpc_secure=cfg.grpc_secure,
                auth_credentials=auth,
                skip_init_checks=True,
                additional_config=AdditionalConfig(timeout=Timeout(init=10, query=30, insert=60)),
            )

        self._auth_mode = self._detect_auth_mode()

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            finally:
                self._client = None

    def _detect_auth_mode(self) -> str:
        """Resolve the live authorization mode.

        Asking for roles is NOT a reliable probe: on a server with RBAC
        disabled, ``GET /v1/authz/roles`` answers ``200 []`` rather than an
        error (verified against Weaviate 1.30.0), so a successful call proves
        nothing. What does discriminate is the identity's own record: with
        RBAC off ``roles`` comes back empty, and with RBAC on every
        authenticated identity carries at least one role — a user with none
        could not have reached this code, since it could not read the schema.

        A 403 anywhere in the probe also means RBAC: the server is enforcing
        permissions, this identity simply may not introspect itself.
        """
        declared = (self._config.auth_mode or "auto").lower()
        if declared in ("rbac", "admin_list"):
            return declared
        try:
            user = self.client.users.get_my_user()
        except Exception as exc:
            if _is_forbidden(exc):
                return "rbac"
            logger.debug("auth-mode probe failed, assuming admin_list (%s)", exc)
            return "admin_list"

        roles = getattr(user, "roles", None) or {}
        return "rbac" if len(roles) > 0 else "admin_list"

    # -- schema -------------------------------------------------------------

    def ensure_collection(self, name: str) -> None:
        """Create the collection if absent; tolerate a read-only identity.

        Under RBAC the memory user usually holds data permissions but not
        ``create_collection``. That is a legitimate deployment, so a denial
        here is recorded and reported, never raised: the collection is
        expected to have been provisioned by an administrator.
        """
        if name in self._ensured:
            return

        try:
            if self.client.collections.exists(name):
                self._ensured.add(name)
                self._ensure_tenant(name)
                return
        except Exception as exc:
            if _is_forbidden(exc):
                self._readonly_schema = True
                self._ensured.add(name)
                return
            raise StoreError(f"cannot inspect collection {name!r}: {exc}") from exc

        from weaviate.classes.config import Configure, DataType, Property

        cfg = self._config
        properties = [
            Property(name="content", data_type=DataType.TEXT),
            Property(name="domain", data_type=DataType.TEXT),
            Property(name="kind", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="actor", data_type=DataType.TEXT),
            Property(name="sensitivity", data_type=DataType.TEXT),
            Property(name="session_id", data_type=DataType.TEXT),
            Property(name="tags", data_type=DataType.TEXT_ARRAY),
            Property(name="created_at", data_type=DataType.DATE),
        ]

        kwargs: dict[str, Any] = {
            "name": name,
            "description": f"Hermes agent memory — {name}",
            "properties": properties,
            # Vectors are computed by this plugin through Ollama and supplied
            # with every write, so the server needs no vectorizer module.
            "vector_config": Configure.Vectors.self_provided(),
        }
        if cfg.multi_tenancy:
            kwargs["multi_tenancy_config"] = Configure.multi_tenancy(
                enabled=True, auto_tenant_creation=False
            )
        if cfg.replication_factor > 1:
            kwargs["replication_config"] = Configure.replication(factor=cfg.replication_factor)

        try:
            self.client.collections.create(**kwargs)
            logger.info("weaviate-memory created collection %s", name)
        except Exception as exc:
            if _is_forbidden(exc):
                self._readonly_schema = True
                self._ensured.add(name)
                logger.warning(
                    "weaviate-memory may not create %s (RBAC). "
                    "Ask an administrator to provision it.",
                    name,
                )
                return
            # A concurrent agent may have won the race; that is success.
            if "already exists" in str(exc).lower():
                self._ensured.add(name)
                self._ensure_tenant(name)
                return
            raise StoreError(f"cannot create collection {name!r}: {exc}") from exc

        self._ensured.add(name)
        self._ensure_tenant(name)

    def _ensure_tenant(self, name: str) -> None:
        """Create this operator's tenant when multi-tenancy is on."""
        if not self._config.multi_tenancy:
            return
        from weaviate.classes.tenants import Tenant

        collection = self.client.collections.get(name)
        wanted = self._config.tenant
        try:
            existing = {t for t in collection.tenants.get()}
            if wanted in existing:
                return
            collection.tenants.create([Tenant(name=wanted)])
            logger.info("weaviate-memory created tenant %s on %s", wanted, name)
        except Exception as exc:
            if _is_forbidden(exc):
                self._readonly_schema = True
                logger.warning(
                    "weaviate-memory may not manage tenants on %s (RBAC); "
                    "tenant %r must exist already.",
                    name,
                    wanted,
                )
                return
            raise StoreError(f"cannot ensure tenant on {name!r}: {exc}") from exc

    def _handle(self, collection_name: str, tenant: str | None = None):
        """Return a collection handle, tenant-scoped when multi-tenancy is on."""
        collection = self.client.collections.get(collection_name)
        if self._config.multi_tenancy:
            return collection.with_tenant(tenant or self._config.tenant)
        return collection

    # -- writes -------------------------------------------------------------

    def add(
        self,
        collection_name: str,
        content: str,
        vector: Sequence[float],
        *,
        domain: str = "",
        kind: str = "observation",
        source: str = "tool",
        actor: str = "",
        sensitivity: str = "internal",
        session_id: str = "",
        tags: Sequence[str] = (),
        tenant: str | None = None,
    ) -> str:
        """Insert one memory. Returns the object UUID."""
        self.ensure_collection(collection_name)
        handle = self._handle(collection_name, tenant)
        properties = {
            "content": content,
            "domain": domain,
            "kind": kind,
            "source": source,
            "actor": actor,
            "sensitivity": sensitivity,
            "session_id": session_id,
            "tags": list(tags),
            "created_at": utcnow(),
        }
        try:
            uuid = handle.data.insert(properties=properties, vector=list(vector))
        except Exception as exc:
            if _is_forbidden(exc):
                raise PermissionDeniedError(
                    f"not allowed to write to {collection_name!r} "
                    f"(tenant {tenant or self._config.tenant!r})"
                ) from exc
            raise StoreError(f"insert into {collection_name!r} failed: {exc}") from exc
        return str(uuid)

    def delete(self, collection_name: str, uuid: str, *, tenant: str | None = None) -> bool:
        handle = self._handle(collection_name, tenant)
        try:
            return bool(handle.data.delete_by_id(uuid))
        except Exception as exc:
            if _is_forbidden(exc):
                raise PermissionDeniedError(
                    f"not allowed to delete from {collection_name!r}"
                ) from exc
            raise StoreError(f"delete from {collection_name!r} failed: {exc}") from exc

    # -- search -------------------------------------------------------------

    def search(
        self,
        collection_name: str,
        vector: Sequence[float],
        query: str,
        *,
        limit: int = 10,
        min_certainty: float = 0.0,
        alpha: float = 0.7,
        tenant: str | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid (vector + BM25) search within one collection and tenant.

        Hybrid is used rather than pure vector search because agent memories
        mix prose with literal tokens — error codes, file paths, flag names —
        that lexical matching finds and embeddings blur.
        """
        from weaviate.classes.query import MetadataQuery

        handle = self._handle(collection_name, tenant)
        try:
            response = handle.query.hybrid(
                query=query,
                vector=list(vector),
                alpha=alpha,
                limit=limit,
                return_metadata=MetadataQuery(score=True, explain_score=False),
                return_properties=list(PROPERTY_NAMES),
            )
        except Exception as exc:
            if _is_forbidden(exc):
                raise PermissionDeniedError(
                    f"not allowed to read {collection_name!r} "
                    f"(tenant {tenant or self._config.tenant!r})"
                ) from exc
            raise StoreError(f"search in {collection_name!r} failed: {exc}") from exc

        hits: list[dict[str, Any]] = []
        for obj in response.objects:
            props = dict(obj.properties or {})
            score = float(getattr(obj.metadata, "score", 0.0) or 0.0)
            hits.append(
                {
                    "id": str(obj.uuid),
                    "collection": collection_name,
                    "score": score,
                    "content": props.get("content", ""),
                    "domain": props.get("domain", ""),
                    "kind": props.get("kind", ""),
                    "source": props.get("source", ""),
                    "tags": props.get("tags") or [],
                }
            )
        return hits

    def near_vector(
        self,
        collection_name: str,
        vector: Sequence[float],
        *,
        limit: int = 10,
        min_certainty: float = 0.0,
        tenant: str | None = None,
    ) -> list[dict[str, Any]]:
        """Pure vector search, exposing Weaviate's cosine ``certainty``.

        Used by automatic recall, where a comparable-across-queries threshold
        matters. Hybrid scores are relative to a result set and cannot be
        thresholded the same way.
        """
        from weaviate.classes.query import MetadataQuery

        handle = self._handle(collection_name, tenant)
        try:
            response = handle.query.near_vector(
                near_vector=list(vector),
                limit=limit,
                certainty=min_certainty or None,
                return_metadata=MetadataQuery(certainty=True, distance=True),
                return_properties=list(PROPERTY_NAMES),
            )
        except Exception as exc:
            if _is_forbidden(exc):
                raise PermissionDeniedError(f"not allowed to read {collection_name!r}") from exc
            raise StoreError(f"near_vector in {collection_name!r} failed: {exc}") from exc

        hits: list[dict[str, Any]] = []
        for obj in response.objects:
            props = dict(obj.properties or {})
            hits.append(
                {
                    "id": str(obj.uuid),
                    "collection": collection_name,
                    "certainty": float(getattr(obj.metadata, "certainty", 0.0) or 0.0),
                    "content": props.get("content", ""),
                    "domain": props.get("domain", ""),
                    "kind": props.get("kind", ""),
                    "source": props.get("source", ""),
                    "tags": props.get("tags") or [],
                }
            )
        return hits

    # -- introspection ------------------------------------------------------

    def count(self, collection_name: str, *, tenant: str | None = None) -> int:
        handle = self._handle(collection_name, tenant)
        try:
            return int(handle.aggregate.over_all(total_count=True).total_count or 0)
        except Exception as exc:
            logger.debug("count on %s failed: %s", collection_name, exc)
            return 0

    def totals(self) -> dict[str, int]:
        """Per-collection object counts for this tenant."""
        result: dict[str, int] = {}
        for name in self._config.collections():
            if name in self._ensured or self._exists_quiet(name):
                result[name] = self.count(name)
        return result

    def _exists_quiet(self, name: str) -> bool:
        try:
            return bool(self.client.collections.exists(name))
        except Exception:
            return False
