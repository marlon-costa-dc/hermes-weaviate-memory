"""weaviate-memory — Hermes memory provider backed by Weaviate.

Layered memory for agents: a collection per domain, a tenant per operator or
team, and an authorization mode that follows the server (admin_list or RBAC).
Embeddings are produced locally through Ollama, so memory text never leaves
the machine except to the Weaviate cluster you point it at.

Config in $HERMES_HOME/config.yaml:

    memory:
      provider: weaviate-memory
    plugins:
      weaviate-memory:
        http_host: weaviate.int.example.com
        http_port: 443
        http_secure: true
        use_grpc: false
        multi_tenancy: true
        auth_mode: auto        # auto | admin_list | rbac
        embed_model: nomic-embed-text

The API key belongs in $HERMES_HOME/.env as WEAVIATE_MEMORY_API_KEY.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from agent.memory_provider import MemoryProvider, RecallStatus, is_trivial_prompt

from .config import Config, load_config
from .embeddings import EmbeddingError, OllamaEmbedder
from .store import PermissionDeniedError, StoreError, WeaviateStore

logger = logging.getLogger(__name__)

__version__ = "0.1.0"

__all__ = [
    "Config",
    "EmbeddingError",
    "OllamaEmbedder",
    "PermissionDeniedError",
    "StoreError",
    "WeaviateMemoryProvider",
    "WeaviateStore",
    "load_config",
]

REMEMBER_SCHEMA = {
    "name": "weaviate_remember",
    "description": (
        "Store a durable memory in Weaviate, filed under a domain. "
        "Use for facts worth recalling in a LATER session: user preferences, "
        "project conventions, environment quirks, decisions and their reasons.\n\n"
        "Write one self-contained statement per call — it will be retrieved "
        "without surrounding conversation. Prefer 'The operator runs CachyOS "
        "with systemd-boot' over 'he uses that distro'.\n\n"
        "Do NOT store: transient task state, secrets, or anything trivially "
        "re-derivable from the filesystem."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The memory, as one self-contained declarative statement.",
            },
            "domain": {
                "type": "string",
                "description": (
                    "Which memory domain to file this under. Episodic = what "
                    "happened; Semantic = stable facts; Procedural = how to do "
                    "something. Defaults to Semantic."
                ),
            },
            "kind": {
                "type": "string",
                "enum": ["fact", "preference", "observation"],
                "description": (
                    "fact = objective and durable; preference = how the user wants "
                    "things done; observation = something noticed, weaker claim."
                ),
            },
            "tags": {
                "type": "string",
                "description": "Optional comma-separated tags (e.g. 'kubernetes,weaviate').",
            },
        },
        "required": ["content"],
    },
}

RECALL_SCHEMA = {
    "name": "weaviate_recall",
    "description": (
        "Search stored memories by meaning. Combines vector similarity with "
        "keyword matching, so it finds paraphrases AND exact tokens like error "
        "codes or file paths.\n\n"
        "Call this before answering questions about the user's environment, "
        "preferences, or past decisions — recall injected automatically each "
        "turn is capped and may not cover the question at hand."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "domain": {
                "type": "string",
                "description": "Optional: restrict to one domain (Episodic/Semantic/Procedural).",
            },
            "limit": {"type": "integer", "description": "Max results (default 10)."},
        },
        "required": ["query"],
    },
}

FORGET_SCHEMA = {
    "name": "weaviate_forget",
    "description": (
        "Delete a memory by id, when it is wrong or obsolete. Get the id and "
        "its domain from weaviate_recall. Deletion is permanent."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "UUID from weaviate_recall."},
            "domain": {
                "type": "string",
                "description": "Domain the memory lives in (from weaviate_recall).",
            },
        },
        "required": ["memory_id", "domain"],
    },
}


class WeaviateMemoryProvider(MemoryProvider):
    """MemoryProvider implementation over Weaviate with per-domain collections."""

    def __init__(self, config: Config | None = None) -> None:
        self._config = config
        self._store: WeaviateStore | None = None
        self._embedder: OllamaEmbedder | None = None
        self._session_id = ""
        self._agent_identity = ""
        self._unavailable_reason = ""
        self._last_recall_count = 0
        # Prefetch runs on a background thread; the result is consumed on the
        # next turn. Guarded because the turn thread reads what it writes.
        self._prefetch_lock = threading.Lock()
        self._prefetched = ""

    # -- identity -----------------------------------------------------------

    @property
    def name(self) -> str:
        return "weaviate-memory"

    @property
    def config(self) -> Config:
        if self._config is None:
            try:
                from hermes_cli.config import cfg_get
            except ImportError:
                cfg_get = None
            self._config = load_config(cfg_get)
        return self._config

    # -- lifecycle ----------------------------------------------------------

    def is_available(self) -> bool:
        """Cheap readiness check: client importable, key present, embedder up.

        The host calls this before initialize() and documents it as
        no-network for the backend; the Ollama probe is a localhost GET with a
        5s timeout, which is the only way to know the embedder exists at all.
        """
        try:
            import weaviate  # noqa: F401
        except ImportError:
            self._unavailable_reason = (
                "weaviate-client is not installed — run: pip install 'weaviate-client>=4.22,<5'"
            )
            return False

        if not self.config.api_key:
            self._unavailable_reason = (
                "WEAVIATE_MEMORY_API_KEY is not set — add it to $HERMES_HOME/.env"
            )
            return False

        embedder = OllamaEmbedder(self.config.ollama_host, self.config.embed_model)
        if not embedder.is_available():
            self._unavailable_reason = (
                f"Ollama at {self.config.ollama_host} is not serving "
                f"{self.config.embed_model!r} — run: ollama pull {self.config.embed_model}"
            )
            return False

        self._embedder = embedder
        return True

    def unavailable_reason(self) -> str:
        return self._unavailable_reason

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._agent_identity = kwargs.get("agent_identity", "") or ""

        if self._embedder is None:
            self._embedder = OllamaEmbedder(self.config.ollama_host, self.config.embed_model)

        dims = self._embedder.dims
        if dims is None:
            # Unknown model: learn the width from one real call rather than
            # guessing, so a dimension mismatch surfaces immediately.
            dims = len(self._embedder.embed_one("dimension probe"))

        store = WeaviateStore(self.config, dims)
        store.connect()
        for collection in self.config.collections():
            store.ensure_collection(collection)
        self._store = store

        logger.info(
            "weaviate-memory ready: %s @ %s:%s (auth=%s, tenant=%s, %d dims)",
            ", ".join(self.config.collections()),
            self.config.http_host,
            self.config.http_port,
            store.auth_mode,
            self.config.tenant if self.config.multi_tenancy else "-",
            dims,
        )

    def shutdown(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None

    # -- prompt integration -------------------------------------------------

    def system_prompt_block(self) -> str:
        if self._store is None:
            return ""
        try:
            totals = self._store.totals()
        except Exception as exc:
            logger.debug("totals failed: %s", exc)
            return ""
        total = sum(totals.values())
        if not total:
            return ""
        breakdown = ", ".join(
            f"{n.replace(self.config.collection_prefix, '')}: {c}"
            for n, c in sorted(totals.items())
            if c
        )
        scope = f" (tenant {self.config.tenant})" if self.config.multi_tenancy else ""
        return (
            f"Persistent memory: {total} entries in Weaviate{scope} — {breakdown}. "
            f"Use weaviate_recall to search them and weaviate_remember to add "
            f"durable ones."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return recall context for this turn.

        Consumes whatever the background thread prepared after the previous
        turn. Returns quickly and never embeds on the turn thread.
        """
        if not self.config.auto_recall:
            return ""
        with self._prefetch_lock:
            text, self._prefetched = self._prefetched, ""
        return text

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Kick off background recall for the NEXT turn."""
        if not self.config.auto_recall or self._store is None:
            return
        if len(query.strip()) < 12 or is_trivial_prompt(query):
            # Host-owned classifier: "ok", "thanks", "vai" carry no retrievable
            # intent, and embedding them would return arbitrary neighbours.
            return

        def _work() -> None:
            try:
                hits = self._recall_across_domains(query, limit=self.config.recall_limit)
            except Exception as exc:
                logger.debug("background recall failed: %s", exc)
                return
            if not hits:
                with self._prefetch_lock:
                    self._prefetched = ""
                self._last_recall_count = 0
                return
            lines = [
                f"- [{h['collection'].replace(self.config.collection_prefix, '')}"
                f"/{h.get('kind') or 'memory'}] {h['content']}"
                for h in hits
            ]
            with self._prefetch_lock:
                self._prefetched = "Relevant memories (retrieved from Weaviate):\n" + "\n".join(
                    lines
                )
            self._last_recall_count = len(hits)

        threading.Thread(target=_work, daemon=True, name="weav-recall").start()

    def recall_status(self) -> RecallStatus | None:
        if not self._last_recall_count:
            return None
        count, self._last_recall_count = self._last_recall_count, 0
        return RecallStatus(provider_label="weaviate", count=count)

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        """Optionally capture the turn, then queue recall for the next one."""
        if self.config.auto_capture_turns and self._store is not None:
            text = f"User: {user_content.strip()}\nAssistant: {assistant_content.strip()}"
            if len(text) >= self.config.min_turn_chars:
                threading.Thread(
                    target=self._store_safe,
                    args=(text,),
                    kwargs={
                        "domain": "Episodic",
                        "kind": "observation",
                        "source": "turn",
                        "session_id": session_id,
                    },
                    daemon=True,
                    name="weav-capture",
                ).start()
        self.queue_prefetch(user_content, session_id=session_id)

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Mirror built-in MEMORY.md / USER.md writes into Weaviate.

        MEMORY.md has a hard character budget; Weaviate does not. Mirroring
        means a memory evicted from the file for space stays retrievable.
        """
        if not self.config.mirror_memory_tool or self._store is None:
            return
        if action not in ("add", "replace") or not content.strip():
            return
        threading.Thread(
            target=self._store_safe,
            args=(content,),
            kwargs={
                "domain": "Semantic",
                "kind": "preference" if target == "user" else "fact",
                "source": "memory_tool",
                "tags": [f"target:{target}", f"action:{action}"],
            },
            daemon=True,
            name="weav-mirror",
        ).start()

    # -- tools --------------------------------------------------------------

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [REMEMBER_SCHEMA, RECALL_SCHEMA, FORGET_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs) -> str:
        if self._store is None:
            return "Error: weaviate-memory is not initialized."
        try:
            if tool_name == "weaviate_remember":
                return self._tool_remember(args)
            if tool_name == "weaviate_recall":
                return self._tool_recall(args)
            if tool_name == "weaviate_forget":
                return self._tool_forget(args)
        except PermissionDeniedError as exc:
            return f"Error: permission denied — {exc}"
        except EmbeddingError as exc:
            return f"Error: embedding failed — {exc}"
        except StoreError as exc:
            return f"Error: storage failed — {exc}"
        except Exception as exc:  # surface, never silently succeed
            logger.exception("weaviate-memory tool %s failed", tool_name)
            return f"Error: {type(exc).__name__}: {exc}"
        return f"Error: unknown tool {tool_name!r}"

    def _resolve_domain(self, requested: str | None) -> str:
        """Map a requested domain onto a configured one, case-insensitively."""
        domains = self.config.domains
        if not requested:
            return "Semantic" if "Semantic" in domains else domains[0]
        wanted = requested.strip().lower()
        for domain in domains:
            if domain.lower() == wanted:
                return domain
        return "Semantic" if "Semantic" in domains else domains[0]

    def _tool_remember(self, args: dict[str, Any]) -> str:
        content = (args.get("content") or "").strip()
        if not content:
            return "Error: 'content' is required."
        domain = self._resolve_domain(args.get("domain"))
        tags = [t.strip() for t in (args.get("tags") or "").split(",") if t.strip()]
        uuid = self._store_now(
            content,
            domain=domain,
            kind=args.get("kind") or "observation",
            source="tool",
            tags=tags,
        )
        return f"Stored in {self.config.collection_for(domain)} as {uuid}."

    def _tool_recall(self, args: dict[str, Any]) -> str:
        query = (args.get("query") or "").strip()
        if not query:
            return "Error: 'query' is required."
        limit = int(args.get("limit") or 10)
        requested = args.get("domain")
        domains = [self._resolve_domain(requested)] if requested else list(self.config.domains)

        hits = self._recall_across_domains(
            query,
            limit=limit,
            domains=domains,
            # An explicit search is the agent looking for something specific:
            # drop the automatic-recall floor and let it see weak matches.
            min_certainty=0.0,
            hybrid=True,
        )
        if not hits:
            return f"No memories found for {query!r}."
        lines = [
            f"{h['id']} [{h['collection']}/{h.get('kind') or 'memory'}] "
            f"(score {h.get('score', h.get('certainty', 0.0)):.3f}) {h['content']}"
            for h in hits
        ]
        return f"{len(hits)} memories:\n" + "\n".join(lines)

    def _tool_forget(self, args: dict[str, Any]) -> str:
        memory_id = (args.get("memory_id") or "").strip()
        if not memory_id:
            return "Error: 'memory_id' is required."
        domain = self._resolve_domain(args.get("domain"))
        store = self._require_store()
        if store.delete(self.config.collection_for(domain), memory_id):
            return f"Deleted {memory_id} from {self.config.collection_for(domain)}."
        return f"No memory {memory_id} in {self.config.collection_for(domain)}."

    # -- internals ----------------------------------------------------------

    def _require_store(self) -> WeaviateStore:
        """Return the live store, or fail loudly.

        Every call path below runs only after initialize(), but a provider
        whose init failed must not silently no-op: a memory that was never
        written is worse than an error the operator can see.
        """
        if self._store is None or self._embedder is None:
            raise StoreError("weaviate-memory is not initialized")
        return self._store

    def _recall_across_domains(
        self,
        query: str,
        *,
        limit: int,
        domains: list[str] | None = None,
        min_certainty: float | None = None,
        hybrid: bool = False,
    ) -> list[dict[str, Any]]:
        """Search every domain, merge, and keep the strongest ``limit`` hits.

        Cross-domain by default because the caller asks a question, not a
        filing-cabinet drawer: a procedural memory can answer what looks like
        a semantic question.
        """
        store = self._require_store()
        assert self._embedder is not None
        vector = self._embedder.embed_one(query)
        floor = self.config.min_certainty if min_certainty is None else min_certainty
        targets = domains or list(self.config.domains)

        tenants: list[str | None] = [None]
        if self.config.multi_tenancy and self.config.search_shared:
            shared = self.config.shared_tenant
            if shared and shared != self.config.tenant:
                tenants.append(shared)

        merged: list[dict[str, Any]] = []
        for domain in targets:
            collection = self.config.collection_for(domain)
            for tenant in tenants:
                try:
                    if hybrid:
                        hits = store.search(
                            collection,
                            vector,
                            query,
                            limit=limit,
                            alpha=self.config.hybrid_alpha,
                            tenant=tenant,
                        )
                    else:
                        hits = store.near_vector(
                            collection,
                            vector,
                            limit=limit,
                            min_certainty=floor,
                            tenant=tenant,
                        )
                except PermissionDeniedError as exc:
                    # Under RBAC a scoped role legitimately cannot see every
                    # collection or tenant. Skip what is denied, keep the rest.
                    logger.debug("recall skipped %s/%s: %s", collection, tenant, exc)
                    continue
                except StoreError as exc:
                    logger.debug("recall failed on %s/%s: %s", collection, tenant, exc)
                    continue
                merged.extend(hits)

        merged.sort(key=lambda h: h.get("score", h.get("certainty", 0.0)), reverse=True)
        return merged[:limit]

    def _store_now(self, content: str, *, domain: str = "Semantic", **kwargs) -> str:
        store = self._require_store()
        assert self._embedder is not None
        vector = self._embedder.embed_one(content)
        return store.add(
            self.config.collection_for(domain),
            content,
            vector,
            domain=domain,
            actor=self._agent_identity or self.config.tenant,
            session_id=kwargs.pop("session_id", self._session_id),
            **kwargs,
        )

    def _store_safe(self, content: str, **kwargs) -> None:
        """Background write. Logs failures instead of raising into a thread."""
        try:
            self._store_now(content, **kwargs)
        except Exception as exc:
            logger.warning("weaviate-memory background write failed: %s", exc)

    # -- setup surface ------------------------------------------------------

    def get_config_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "api_key",
                "description": "Weaviate API key for this agent's identity",
                "secret": True,
                "required": True,
                "env_var": "WEAVIATE_MEMORY_API_KEY",
            },
            {
                "key": "http_host",
                "description": "Weaviate host (no scheme)",
                "default": "127.0.0.1",
                "required": True,
            },
            {
                "key": "http_port",
                "description": "Weaviate HTTP port",
                "type": "integer",
                "default": 8080,
                "minimum": 1,
                "maximum": 65535,
            },
            {
                "key": "http_secure",
                "description": "Use HTTPS for the REST API",
                "type": "boolean",
                "default": False,
            },
            {
                "key": "use_grpc",
                "description": "Verify the gRPC route at connect time (needed for tenants/queries)",
                "type": "boolean",
                "default": True,
            },
            {
                "key": "auth_mode",
                "description": "Authorization mode of the server",
                "default": "auto",
                "choices": ["auto", "admin_list", "rbac"],
            },
            {
                "key": "multi_tenancy",
                "description": "Isolate memories per operator/team via tenants",
                "type": "boolean",
                "default": True,
            },
            {
                "key": "tenant",
                "description": "Tenant name (defaults to u_<login>)",
                "default": "",
            },
            {
                "key": "collection_prefix",
                "description": "Prefix for memory collections",
                "default": "Mem",
            },
            {
                "key": "embed_model",
                "description": "Ollama embedding model",
                "default": "nomic-embed-text",
            },
            {
                "key": "ollama_host",
                "description": "Ollama base URL",
                "default": "http://127.0.0.1:11434",
            },
            {
                "key": "auto_recall",
                "description": "Inject relevant memories before each turn",
                "type": "boolean",
                "default": True,
            },
            {
                "key": "recall_limit",
                "description": "Max memories injected per turn",
                "type": "integer",
                "default": 5,
                "minimum": 1,
                "maximum": 50,
            },
            {
                "key": "min_certainty",
                "description": "Cosine certainty floor for automatic recall",
                "type": "number",
                "default": 0.6,
                "minimum": 0.0,
                "maximum": 1.0,
                "step": 0.05,
            },
            {
                "key": "hybrid_alpha",
                "description": "Hybrid weight: 1.0 vector-only, 0.0 keyword-only",
                "type": "number",
                "default": 0.7,
                "minimum": 0.0,
                "maximum": 1.0,
                "step": 0.05,
            },
            {
                "key": "auto_capture_turns",
                "description": "Also store every conversation turn (noisy)",
                "type": "boolean",
                "default": False,
            },
            {
                "key": "mirror_memory_tool",
                "description": "Mirror MEMORY.md / USER.md writes into Weaviate",
                "type": "boolean",
                "default": True,
            },
        ]

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        """Persist non-secret settings under plugins.weaviate-memory."""
        try:
            from hermes_cli.config import cfg_set
        except ImportError:
            logger.warning("hermes_cli.config unavailable; config not saved")
            return
        for key, value in values.items():
            if key == "api_key":  # secrets live in .env, never config.yaml
                continue
            try:
                cfg_set(f"plugins.weaviate-memory.{key}", value)
            except Exception as exc:
                logger.warning("could not save %s: %s", key, exc)


def register_memory_provider() -> WeaviateMemoryProvider:
    """Entry point used by Hermes' directory-based plugin loader."""
    return WeaviateMemoryProvider()
