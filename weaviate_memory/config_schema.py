"""Declarative config surface for the Hermes desktop/dashboard UI.

Loaded **by path** by the web server, never imported as a package: the
provider's ``__init__.py`` pulls in the agent runtime, which must not load
into the web process. This module may only import from the host's
``plugins.memory.config_schema``.

The field keys mirror ``weaviate_memory.config.Config`` so the generic
read/write endpoints land values where ``load_config`` reads them.
"""

from __future__ import annotations

try:
    from plugins.memory.config_schema import (
        KIND_BOOL,
        KIND_NUMBER,
        KIND_SECRET,
        KIND_SELECT,
        KIND_TEXT,
        ProviderConfigSchema,
        ProviderField,
        ProviderFieldOption,
    )
except ImportError:  # outside the host (tests, tooling)
    KIND_TEXT, KIND_SELECT, KIND_SECRET = "text", "select", "secret"
    KIND_BOOL, KIND_NUMBER = "bool", "number"
    ProviderConfigSchema = ProviderField = ProviderFieldOption = None  # type: ignore[assignment]


if ProviderConfigSchema is not None:
    CONFIG_SCHEMA = ProviderConfigSchema(
        name="weaviate-memory",
        label="Weaviate",
        docs_url="https://github.com/marlon-costa-dc/hermes-weaviate-memory",
        fields=(
            ProviderField(
                key="api_key",
                label="API key",
                kind=KIND_SECRET,
                env_key="WEAVIATE_MEMORY_API_KEY",
                inline=True,
                group="Connection",
                description="Weaviate API key for this agent's identity.",
                info=(
                    "Under RBAC this key maps to a database user whose role is "
                    "scoped to the memory collections and one tenant. Stored in "
                    ".env, never in config.yaml."
                ),
            ),
            ProviderField(
                key="http_host",
                label="Host",
                default="127.0.0.1",
                inline=True,
                group="Connection",
                placeholder="weaviate.internal.example.com",
                description="Hostname, without a scheme.",
            ),
            ProviderField(
                key="http_port",
                label="HTTP port",
                kind=KIND_NUMBER,
                default="8080",
                group="Connection",
            ),
            ProviderField(
                key="http_secure",
                label="Use HTTPS",
                kind=KIND_BOOL,
                default="false",
                group="Connection",
            ),
            ProviderField(
                key="grpc_host",
                label="gRPC host",
                group="Connection",
                description="Defaults to the HTTP host when empty.",
                info=(
                    "The v4 client sends queries and tenant operations over "
                    "gRPC. A cluster without a gRPC route cannot serve "
                    "multi-tenancy."
                ),
            ),
            ProviderField(
                key="grpc_port",
                label="gRPC port",
                kind=KIND_NUMBER,
                default="50051",
                group="Connection",
            ),
            ProviderField(
                key="grpc_secure",
                label="gRPC over TLS",
                kind=KIND_BOOL,
                default="false",
                group="Connection",
            ),
            ProviderField(
                key="auth_mode",
                label="Authorization mode",
                kind=KIND_SELECT,
                default="auto",
                inline=True,
                group="Access",
                options=(
                    ProviderFieldOption(
                        "auto", "Detect automatically", "Ask the server which mode is live"
                    ),
                    ProviderFieldOption(
                        "admin_list", "Admin list", "One API key with broad rights"
                    ),
                    ProviderFieldOption(
                        "rbac", "RBAC", "Scoped role per identity (Weaviate 1.29+)"
                    ),
                ),
            ),
            ProviderField(
                key="multi_tenancy",
                label="Isolate by tenant",
                kind=KIND_BOOL,
                default="true",
                inline=True,
                group="Access",
                info=(
                    "Each operator or team gets its own tenant, which is a "
                    "separate shard. A query scoped to one tenant cannot "
                    "return another's objects."
                ),
            ),
            ProviderField(
                key="tenant",
                label="Tenant",
                group="Access",
                placeholder="u_<login>",
                description="Defaults to u_<login> when empty.",
            ),
            ProviderField(
                key="shared_tenant",
                label="Shared tenant",
                default="s_shared",
                group="Access",
                description="Organisation-wide knowledge, searched alongside your own.",
            ),
            ProviderField(
                key="collection_prefix",
                label="Collection prefix",
                default="Mem",
                group="Data model",
                info="Change this to run two independent memory sets on one cluster.",
            ),
            ProviderField(
                key="domains",
                label="Domains",
                default="Episodic,Semantic,Procedural",
                group="Data model",
                description="Comma-separated. One collection is created per domain.",
            ),
            ProviderField(
                key="replication_factor",
                label="Replication factor",
                kind=KIND_NUMBER,
                default="1",
                group="Data model",
                description="Set to your cluster's node count for redundancy.",
            ),
            ProviderField(
                key="embed_model",
                label="Embedding model",
                default="nomic-embed-text",
                inline=True,
                group="Embeddings",
                info=(
                    "Changing this changes the vector width and requires "
                    "re-embedding every stored memory."
                ),
            ),
            ProviderField(
                key="ollama_host",
                label="Ollama URL",
                default="http://127.0.0.1:11434",
                group="Embeddings",
            ),
            ProviderField(
                key="auto_recall",
                label="Recall automatically",
                kind=KIND_BOOL,
                default="true",
                inline=True,
                group="Recall",
                description="Inject relevant memories before each turn.",
            ),
            ProviderField(
                key="recall_limit",
                label="Memories per turn",
                kind=KIND_NUMBER,
                default="5",
                group="Recall",
            ),
            ProviderField(
                key="min_certainty",
                label="Certainty floor",
                kind=KIND_NUMBER,
                default="0.6",
                group="Recall",
                info=(
                    "Cosine certainty below which a match is treated as noise. "
                    "Applies to automatic recall only; an explicit search sees "
                    "everything."
                ),
            ),
            ProviderField(
                key="hybrid_alpha",
                label="Hybrid weight",
                kind=KIND_NUMBER,
                default="0.7",
                group="Recall",
                info="1.0 is pure vector search, 0.0 is pure keyword search.",
            ),
            ProviderField(
                key="search_shared",
                label="Also search shared tenant",
                kind=KIND_BOOL,
                default="true",
                group="Recall",
            ),
            ProviderField(
                key="auto_capture_turns",
                label="Capture every turn",
                kind=KIND_BOOL,
                default="false",
                group="Capture",
                description="Noisy: stores whole conversation turns, not curated facts.",
            ),
            ProviderField(
                key="mirror_memory_tool",
                label="Mirror built-in memory",
                kind=KIND_BOOL,
                default="true",
                group="Capture",
                info=(
                    "MEMORY.md has a character budget; Weaviate does not. "
                    "Mirroring keeps evicted entries retrievable."
                ),
            ),
        ),
    )
else:  # pragma: no cover - only outside the host
    CONFIG_SCHEMA = None
