"""Configuration resolution for the Weaviate memory provider.

Precedence, highest first:
  1. Environment variables (WEAVIATE_MEMORY_*) — secrets and overrides
  2. config.yaml under plugins.weaviate-memory — the canonical surface
  3. Built-in defaults

This mirrors Hermes' own rule: .env carries credentials, config.yaml carries
behaviour. An API key is a credential, so it is accepted only from the
environment — never written to config.yaml.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

DEFAULT_HTTP_HOST = "127.0.0.1"
DEFAULT_HTTP_PORT = 8080
DEFAULT_GRPC_PORT = 50051

# Collection per domain. The provider owns the "Mem*" prefix so an RBAC role
# can be scoped with a single `Mem*` collection filter (see docs/RBAC.md).
DEFAULT_COLLECTION_PREFIX = "Mem"
DEFAULT_DOMAINS = ("Episodic", "Semantic", "Procedural")


@dataclass
class Config:
    """Resolved provider settings."""

    # -- connection ---------------------------------------------------------
    http_host: str = DEFAULT_HTTP_HOST
    http_port: int = DEFAULT_HTTP_PORT
    http_secure: bool = False
    grpc_host: str = ""  # defaults to http_host when empty
    grpc_port: int = DEFAULT_GRPC_PORT
    grpc_secure: bool = False
    api_key: str = ""
    # The v4 client routes queries AND tenant operations over gRPC; only a few
    # REST calls remain. Disabling this is therefore not "REST-only mode" —
    # it is for deployments that expose gRPC on a non-default route, where
    # skip_init_checks stops the client probing the wrong port at connect time.
    # A cluster with no gRPC route at all cannot serve multi-tenancy.
    use_grpc: bool = True

    # -- data model ---------------------------------------------------------
    collection_prefix: str = DEFAULT_COLLECTION_PREFIX
    domains: tuple[str, ...] = DEFAULT_DOMAINS
    # Multi-tenancy is the isolation boundary between operators/teams. When
    # disabled the provider writes to the collection directly, which is what
    # a single-user install wants.
    multi_tenancy: bool = True
    tenant: str = ""  # resolved from the OS user when empty
    shared_tenant: str = "s_shared"
    replication_factor: int = 1

    # -- authorization mode -------------------------------------------------
    # "admin_list": one API key with broad rights (Weaviate's legacy mode).
    # "rbac": per-system DB user whose role is scoped to collections/tenants.
    # "auto": probe /v1/authz/roles once at init and pick the live mode.
    auth_mode: str = "auto"

    # -- embeddings ---------------------------------------------------------
    ollama_host: str = "http://127.0.0.1:11434"
    embed_model: str = "nomic-embed-text"

    # -- recall behaviour ---------------------------------------------------
    auto_recall: bool = True
    recall_limit: int = 5
    # Weaviate reports cosine `certainty` in [0,1]. Below ~0.6 nomic-embed-text
    # neighbours are mostly topical noise; injecting them dilutes attention.
    min_certainty: float = 0.6
    # Hybrid alpha: 1.0 = pure vector, 0.0 = pure BM25. 0.7 keeps semantic
    # recall dominant while still matching exact tokens (error codes, paths).
    hybrid_alpha: float = 0.7
    search_shared: bool = True

    auto_capture_turns: bool = False
    min_turn_chars: int = 80
    mirror_memory_tool: bool = True

    def resolved_grpc_host(self) -> str:
        return self.grpc_host or self.http_host

    def collection_for(self, domain: str) -> str:
        return f"{self.collection_prefix}{domain}"

    def collections(self) -> tuple[str, ...]:
        return tuple(self.collection_for(d) for d in self.domains)

    def as_dict(self) -> dict[str, Any]:
        return {
            "http_host": self.http_host,
            "http_port": self.http_port,
            "http_secure": self.http_secure,
            "grpc_host": self.grpc_host,
            "grpc_port": self.grpc_port,
            "grpc_secure": self.grpc_secure,
            "use_grpc": self.use_grpc,
            "collection_prefix": self.collection_prefix,
            "domains": list(self.domains),
            "multi_tenancy": self.multi_tenancy,
            "tenant": self.tenant,
            "shared_tenant": self.shared_tenant,
            "replication_factor": self.replication_factor,
            "auth_mode": self.auth_mode,
            "ollama_host": self.ollama_host,
            "embed_model": self.embed_model,
            "auto_recall": self.auto_recall,
            "recall_limit": self.recall_limit,
            "min_certainty": self.min_certainty,
            "hybrid_alpha": self.hybrid_alpha,
            "search_shared": self.search_shared,
            "auto_capture_turns": self.auto_capture_turns,
            "min_turn_chars": self.min_turn_chars,
            "mirror_memory_tool": self.mirror_memory_tool,
        }


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_tuple(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        return tuple(parts) or default
    if isinstance(value, (list, tuple)):
        parts = [str(p).strip() for p in value if str(p).strip()]
        return tuple(parts) or default
    return default


def default_tenant() -> str:
    """Tenant name for the current operator: ``u_<login>``.

    Prefixed so an RBAC role can be scoped to a single operator with the
    filter ``u_<login>`` while ``t_*`` stays available for team tenants.
    """
    import getpass

    try:
        login = getpass.getuser()
    except Exception:
        login = os.environ.get("USER") or "default"
    return f"u_{login}"


def load_config(cfg_get=None) -> Config:
    """Resolve configuration.

    ``cfg_get`` is Hermes' ``hermes_cli.config.cfg_get``, injected so this
    module stays importable (and testable) outside the agent process.
    """

    def _cfg(key: str, default: Any = None) -> Any:
        """Read one key, treating an explicit ``None`` as absent.

        Hermes' ``cfg_get`` returns ``None`` for a key that exists in the
        config tree but has no value, ignoring the default it was handed —
        so callers must not assume a non-None result.
        """
        if cfg_get is None:
            return default
        try:
            value = cfg_get(f"plugins.weaviate-memory.{key}", default)
        except Exception:
            return default
        return default if value is None else value

    env = os.environ.get
    tenant = env("WEAVIATE_MEMORY_TENANT") or _cfg("tenant", "") or default_tenant()

    return Config(
        http_host=env("WEAVIATE_MEMORY_HTTP_HOST") or _cfg("http_host", DEFAULT_HTTP_HOST),
        http_port=_as_int(env("WEAVIATE_MEMORY_HTTP_PORT") or _cfg("http_port"), DEFAULT_HTTP_PORT),
        http_secure=_as_bool(env("WEAVIATE_MEMORY_HTTP_SECURE") or _cfg("http_secure"), False),
        grpc_host=env("WEAVIATE_MEMORY_GRPC_HOST") or _cfg("grpc_host", ""),
        grpc_port=_as_int(env("WEAVIATE_MEMORY_GRPC_PORT") or _cfg("grpc_port"), DEFAULT_GRPC_PORT),
        grpc_secure=_as_bool(env("WEAVIATE_MEMORY_GRPC_SECURE") or _cfg("grpc_secure"), False),
        api_key=env("WEAVIATE_MEMORY_API_KEY") or "",
        use_grpc=_as_bool(env("WEAVIATE_MEMORY_USE_GRPC") or _cfg("use_grpc"), True),
        collection_prefix=_cfg("collection_prefix", DEFAULT_COLLECTION_PREFIX),
        domains=_as_tuple(_cfg("domains"), DEFAULT_DOMAINS),
        multi_tenancy=_as_bool(_cfg("multi_tenancy"), True),
        tenant=tenant,
        shared_tenant=_cfg("shared_tenant", "s_shared"),
        replication_factor=_as_int(_cfg("replication_factor"), 1),
        auth_mode=(env("WEAVIATE_MEMORY_AUTH_MODE") or _cfg("auth_mode", "auto")).strip().lower(),
        ollama_host=env("WEAVIATE_MEMORY_OLLAMA_HOST")
        or _cfg("ollama_host", "http://127.0.0.1:11434"),
        embed_model=env("WEAVIATE_MEMORY_EMBED_MODEL") or _cfg("embed_model", "nomic-embed-text"),
        auto_recall=_as_bool(_cfg("auto_recall"), True),
        recall_limit=_as_int(_cfg("recall_limit"), 5),
        min_certainty=_as_float(_cfg("min_certainty"), 0.6),
        hybrid_alpha=_as_float(_cfg("hybrid_alpha"), 0.7),
        search_shared=_as_bool(_cfg("search_shared"), True),
        auto_capture_turns=_as_bool(_cfg("auto_capture_turns"), False),
        min_turn_chars=_as_int(_cfg("min_turn_chars"), 80),
        mirror_memory_tool=_as_bool(_cfg("mirror_memory_tool"), True),
    )
