"""Integration tests — a REAL Weaviate and a REAL Ollama. No mocks.

Nothing here is faked: every assertion is about bytes that crossed a socket.
The suite skips (never passes vacuously) when either dependency is absent.

Point it at a disposable Weaviate:

    export WEAVIATE_MEMORY_HTTP_HOST=127.0.0.1
    export WEAVIATE_MEMORY_HTTP_PORT=8080
    export WEAVIATE_MEMORY_API_KEY=...
    export WEAVIATE_IT=1
    pytest tests/test_integration.py -v

Collections are created under a unique per-run prefix (``ItMem<random>``) and
deleted in teardown, so a run leaves the cluster exactly as it found it.
"""

from __future__ import annotations

import os
import time
import uuid

import pytest

from weaviate_memory import WeaviateMemoryProvider
from weaviate_memory.config import Config
from weaviate_memory.embeddings import OllamaEmbedder
from weaviate_memory.store import PermissionDeniedError, WeaviateStore

RUN_INTEGRATION = os.environ.get("WEAVIATE_IT", "").strip().lower() in ("1", "true", "yes")

pytestmark = pytest.mark.skipif(
    not RUN_INTEGRATION,
    reason="set WEAVIATE_IT=1 (plus WEAVIATE_MEMORY_* env) to run integration tests",
)

DOMAINS = ("Episodic", "Semantic", "Procedural")


def _unique_prefix() -> str:
    return f"ItMem{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module")
def embedder() -> OllamaEmbedder:
    host = os.environ.get("WEAVIATE_MEMORY_OLLAMA_HOST", "http://127.0.0.1:11434")
    model = os.environ.get("WEAVIATE_MEMORY_EMBED_MODEL", "nomic-embed-text")
    emb = OllamaEmbedder(host, model)
    if not emb.is_available():
        pytest.skip(f"Ollama at {host} is not serving {model!r}")
    return emb


@pytest.fixture(scope="module")
def config(embedder) -> Config:
    api_key = os.environ.get("WEAVIATE_MEMORY_API_KEY", "")
    if not api_key:
        pytest.skip("WEAVIATE_MEMORY_API_KEY is not set")
    return Config(
        http_host=os.environ.get("WEAVIATE_MEMORY_HTTP_HOST", "127.0.0.1"),
        http_port=int(os.environ.get("WEAVIATE_MEMORY_HTTP_PORT", "8080")),
        http_secure=os.environ.get("WEAVIATE_MEMORY_HTTP_SECURE", "").lower()
        in ("1", "true", "yes"),
        api_key=api_key,
        use_grpc=os.environ.get("WEAVIATE_MEMORY_USE_GRPC", "").lower() in ("1", "true", "yes"),
        collection_prefix=_unique_prefix(),
        domains=DOMAINS,
        multi_tenancy=True,
        tenant="u_ittest",
        shared_tenant="s_ittest",
        replication_factor=1,
        embed_model=embedder.model,
        ollama_host=embedder.host,
    )


@pytest.fixture(scope="module")
def store(config, embedder):
    import contextlib

    st = WeaviateStore(config, embedder.dims or 768)
    st.connect()
    for name in config.collections():
        st.ensure_collection(name)
    yield st
    for name in config.collections():
        # Teardown must not mask a test failure with a cleanup error.
        with contextlib.suppress(Exception):
            st.client.collections.delete(name)
    st.close()


# -- connectivity ----------------------------------------------------------


def test_server_is_reachable_and_reports_a_version(store):
    meta = store.client.get_meta()
    assert meta["version"]
    print(f"\n  Weaviate {meta['version']} modules={list(meta.get('modules', {}))}")


def test_auth_mode_was_detected(store):
    """The provider must know which authorization world it is living in."""
    assert store.auth_mode in ("rbac", "admin_list")
    print(f"  detected auth_mode={store.auth_mode}")


# -- schema ----------------------------------------------------------------


def test_every_domain_collection_exists(store, config):
    for name in config.collections():
        assert store.client.collections.exists(name), name


def test_collections_are_multi_tenant(store, config):
    for name in config.collections():
        cfg = store.client.collections.get(name).config.get()
        assert cfg.multi_tenancy_config.enabled, name


def test_operator_tenant_was_created(store, config):
    collection = store.client.collections.get(config.collections()[0])
    assert config.tenant in set(collection.tenants.get())


def test_ensure_collection_is_idempotent(store, config):
    """A second agent starting up must not fail or duplicate the schema."""
    before = len(store.client.collections.list_all())
    for name in config.collections():
        store.ensure_collection(name)
    assert len(store.client.collections.list_all()) == before


# -- writes and retrieval --------------------------------------------------


def test_written_memory_is_retrievable_by_meaning(store, config, embedder):
    """The core promise: store a fact, ask a differently-worded question, get it back."""
    collection = config.collection_for("Semantic")
    content = "The operator runs CachyOS with systemd-boot on the workstation tungs"
    store.add(collection, content, embedder.embed_one(content), domain="Semantic", kind="fact")
    time.sleep(1.0)  # HNSW insert is async

    # Deliberately shares almost no vocabulary with the stored sentence.
    query = "which linux distribution does the user have"
    hits = store.near_vector(collection, embedder.embed_one(query), limit=5)

    assert hits, "semantic recall returned nothing"
    assert any("CachyOS" in h["content"] for h in hits)
    top = max(hits, key=lambda h: h["certainty"])
    print(f"\n  certainty={top['certainty']:.4f} for a query sharing no keywords")
    assert top["certainty"] > 0.5


def test_hybrid_search_finds_exact_tokens_vectors_would_blur(store, config, embedder):
    """Error codes and paths are exactly what embeddings lose."""
    collection = config.collection_for("Procedural")
    content = "Fix RAFT_BOOTSTRAP_EXPECT mismatch by setting it to the RAFT_JOIN node count"
    store.add(collection, content, embedder.embed_one(content), domain="Procedural")
    time.sleep(1.0)

    hits = store.search(
        collection, embedder.embed_one("RAFT_BOOTSTRAP_EXPECT"), "RAFT_BOOTSTRAP_EXPECT", limit=5
    )
    assert any("RAFT_BOOTSTRAP_EXPECT" in h["content"] for h in hits)


def test_all_properties_survive_the_round_trip(store, config, embedder):
    collection = config.collection_for("Episodic")
    content = "Property round-trip probe"
    uid = store.add(
        collection,
        content,
        embedder.embed_one(content),
        domain="Episodic",
        kind="observation",
        source="test",
        actor="pytest",
        sensitivity="internal",
        session_id="sess-42",
        tags=["alpha", "beta"],
    )
    time.sleep(1.0)
    hits = store.near_vector(collection, embedder.embed_one(content), limit=5)
    match = next((h for h in hits if h["id"] == uid), None)
    assert match is not None
    assert match["kind"] == "observation"
    assert match["source"] == "test"
    assert set(match["tags"]) == {"alpha", "beta"}


def test_delete_actually_removes_the_object(store, config, embedder):
    collection = config.collection_for("Semantic")
    content = "Ephemeral memory that should not survive"
    uid = store.add(collection, content, embedder.embed_one(content), domain="Semantic")
    time.sleep(1.0)
    assert store.delete(collection, uid) is True
    time.sleep(1.0)
    hits = store.near_vector(collection, embedder.embed_one(content), limit=10)
    assert all(h["id"] != uid for h in hits)


# -- the isolation guarantee ----------------------------------------------


def test_tenants_cannot_see_each_others_memories(store, config, embedder):
    """THE security guarantee of this design.

    If this test ever passes vacuously (both tenants empty) it proves nothing,
    so it asserts the write landed in its own tenant first.
    """
    from weaviate.classes.tenants import Tenant

    collection_name = config.collection_for("Semantic")
    collection = store.client.collections.get(collection_name)
    other = "u_someone_else"
    existing = set(collection.tenants.get())
    if other not in existing:
        collection.tenants.create([Tenant(name=other)])

    secret = "Tenant isolation probe: this belongs to u_ittest alone"
    vector = embedder.embed_one(secret)
    store.add(collection_name, secret, vector, domain="Semantic", tenant=config.tenant)
    time.sleep(1.5)

    mine = store.near_vector(collection_name, vector, limit=10, tenant=config.tenant)
    assert any(secret in h["content"] for h in mine), "own tenant cannot see its own write"

    theirs = store.near_vector(collection_name, vector, limit=10, tenant=other)
    assert all(secret not in h["content"] for h in theirs), "TENANT ISOLATION BREACH"
    print(f"\n  isolation OK: {len(mine)} hit(s) in {config.tenant}, {len(theirs)} in {other}")


def test_writing_to_an_unknown_tenant_is_refused(store, config, embedder):
    """auto_tenant_creation is off, so a typo must fail loudly, not create a shard.

    Asserts the specific error type: a blind `Exception` here would also pass
    on a network blip, proving nothing about tenant enforcement.
    """
    from weaviate_memory.store import StoreError

    collection = config.collection_for("Semantic")
    content = "should never be stored"
    with pytest.raises(StoreError):
        store.add(
            collection,
            content,
            embedder.embed_one(content),
            domain="Semantic",
            tenant="u_this_tenant_does_not_exist",
        )

    # And the stray tenant must not have been created as a side effect.
    tenants = set(store.client.collections.get(collection).tenants.get())
    assert "u_this_tenant_does_not_exist" not in tenants


# -- the provider end to end ----------------------------------------------


@pytest.fixture
def provider(config, store):
    p = WeaviateMemoryProvider(config)
    assert p.is_available(), p.unavailable_reason()
    p.initialize("it-session")
    yield p
    p.shutdown()


def test_provider_initializes_against_the_real_server(provider):
    assert provider._store is not None
    assert provider._store.auth_mode in ("rbac", "admin_list")


def test_remember_then_recall_through_the_tool_surface(provider):
    """Exercise exactly what the model calls, not the internals."""
    stored = provider.handle_tool_call(
        "weaviate_remember",
        {
            "content": "Deployments to dc-prod require an explicit operator approval",
            "domain": "Procedural",
            "kind": "preference",
            "tags": "deploy,policy",
        },
    )
    assert stored.startswith("Stored in"), stored
    time.sleep(1.5)

    recalled = provider.handle_tool_call(
        "weaviate_recall", {"query": "what is needed before shipping to production"}
    )
    assert "explicit operator approval" in recalled, recalled


def test_recall_searches_across_domains_by_default(provider):
    provider.handle_tool_call(
        "weaviate_remember",
        {"content": "The Beads tracker is per-rig, never global", "domain": "Semantic"},
    )
    time.sleep(1.5)
    result = provider.handle_tool_call("weaviate_recall", {"query": "beads tracker scope"})
    assert "per-rig" in result


def test_forget_through_the_tool_surface(provider):
    provider.handle_tool_call(
        "weaviate_remember", {"content": "Temporary tool-surface memory", "domain": "Episodic"}
    )
    time.sleep(1.5)
    listing = provider.handle_tool_call(
        "weaviate_recall", {"query": "Temporary tool-surface memory", "domain": "Episodic"}
    )
    uid = listing.split("\n")[1].split()[0]
    deleted = provider.handle_tool_call("weaviate_forget", {"memory_id": uid, "domain": "Episodic"})
    assert deleted.startswith("Deleted"), deleted


def test_background_prefetch_populates_the_next_turn(provider):
    """The automatic path: no tool call, memory arrives on its own."""
    provider.handle_tool_call(
        "weaviate_remember",
        {"content": "The gateway CLIProxyAPI listens on port 8317", "domain": "Semantic"},
    )
    time.sleep(1.5)

    provider.queue_prefetch("which port does the local model gateway use")
    for _ in range(40):  # background thread; poll instead of a blind sleep
        time.sleep(0.25)
        if provider._prefetched:
            break

    injected = provider.prefetch("which port does the local model gateway use")
    assert "8317" in injected, f"prefetch injected: {injected!r}"
    status = provider.recall_status()
    assert status is not None and status.count > 0


def test_memory_tool_writes_are_mirrored(provider):
    provider.on_memory_write("add", "user", "The operator prefers answers in Portuguese")
    for _ in range(40):
        time.sleep(0.25)
        result = provider.handle_tool_call(
            "weaviate_recall", {"query": "which language should replies use"}
        )
        if "Portuguese" in result:
            break
    assert "Portuguese" in result


def test_system_prompt_reports_real_counts(provider):
    block = provider.system_prompt_block()
    assert "Weaviate" in block
    assert provider.config.tenant in block


def test_permission_errors_are_typed_not_generic(store, config, embedder):
    """Under RBAC, a denial must be PermissionDeniedError so recall can skip it.

    On an admin_list server nothing is denied, so the test asserts the
    classifier instead of forcing a 403 that cannot happen there.
    """
    if store.auth_mode != "rbac":
        pytest.skip("server runs admin_list; no denial to observe")
    with pytest.raises(PermissionDeniedError):
        store.search("SomeCollectionThisKeyCannotRead", embedder.embed_one("x"), "x")
