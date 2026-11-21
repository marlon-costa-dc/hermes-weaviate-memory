"""Unit tests — no network, no Weaviate, no Ollama.

These cover the pure logic: configuration precedence, collection naming,
domain resolution, and the authorization-signal classifier that decides
whether a failure is "you may not" or "this is broken".
"""

from __future__ import annotations

import os

import pytest

from weaviate_memory import WeaviateMemoryProvider
from weaviate_memory.config import Config, default_tenant, load_config
from weaviate_memory.embeddings import KNOWN_DIMS, OllamaEmbedder
from weaviate_memory.store import _is_forbidden


class TestConfig:
    def test_defaults_are_single_user_friendly(self):
        cfg = Config()
        assert cfg.http_host == "127.0.0.1"
        assert cfg.http_port == 8080
        assert cfg.auth_mode == "auto"
        assert cfg.embed_model == "nomic-embed-text"

    def test_env_beats_config_yaml(self, monkeypatch):
        monkeypatch.setenv("WEAVIATE_MEMORY_HTTP_HOST", "from-env")

        def cfg_get(key, default=None):
            return "from-yaml" if key.endswith("http_host") else default

        assert load_config(cfg_get).http_host == "from-env"

    def test_config_yaml_beats_defaults(self, monkeypatch):
        monkeypatch.delenv("WEAVIATE_MEMORY_HTTP_HOST", raising=False)

        def cfg_get(key, default=None):
            return "from-yaml" if key.endswith("http_host") else default

        assert load_config(cfg_get).http_host == "from-yaml"

    def test_api_key_is_env_only(self, monkeypatch):
        """A credential must never be readable from config.yaml."""
        monkeypatch.delenv("WEAVIATE_MEMORY_API_KEY", raising=False)

        def cfg_get(key, default=None):
            return "leaked-from-yaml" if key.endswith("api_key") else default

        assert load_config(cfg_get).api_key == ""

    def test_grpc_host_falls_back_to_http_host(self):
        cfg = Config(http_host="weaviate.example.com", grpc_host="")
        assert cfg.resolved_grpc_host() == "weaviate.example.com"

    def test_explicit_grpc_host_wins(self):
        cfg = Config(http_host="a.example.com", grpc_host="b.example.com")
        assert cfg.resolved_grpc_host() == "b.example.com"

    def test_collection_naming_is_prefix_plus_domain(self):
        cfg = Config(collection_prefix="Mem", domains=("Episodic", "Semantic"))
        assert cfg.collection_for("Episodic") == "MemEpisodic"
        assert cfg.collections() == ("MemEpisodic", "MemSemantic")

    def test_prefix_is_configurable_for_shared_clusters(self):
        """Two Hermes installs on one cluster must not collide."""
        cfg = Config(collection_prefix="TeamMem", domains=("Semantic",))
        assert cfg.collections() == ("TeamMemSemantic",)

    def test_domains_accept_comma_string_from_yaml(self, monkeypatch):
        monkeypatch.delenv("WEAVIATE_MEMORY_TENANT", raising=False)

        def cfg_get(key, default=None):
            return "Alpha, Beta" if key.endswith("domains") else default

        assert load_config(cfg_get).domains == ("Alpha", "Beta")

    def test_bad_numbers_fall_back_instead_of_crashing(self):
        def cfg_get(key, default=None):
            return "not-a-number" if key.endswith("recall_limit") else default

        assert load_config(cfg_get).recall_limit == 5

    def test_cfg_get_returning_none_does_not_crash(self, monkeypatch):
        """Hermes' cfg_get returns None for a valueless key, ignoring the default.

        Regression guard: loading through Hermes' real plugin loader raised
        ``AttributeError: 'NoneType' object has no attribute 'strip'`` because
        load_config trusted cfg_get to honour its default. A test double that
        echoes the default cannot catch this — this one returns None for
        everything, exactly as the host does.
        """
        for var in (
            "WEAVIATE_MEMORY_HTTP_HOST",
            "WEAVIATE_MEMORY_AUTH_MODE",
            "WEAVIATE_MEMORY_EMBED_MODEL",
            "WEAVIATE_MEMORY_OLLAMA_HOST",
            "WEAVIATE_MEMORY_TENANT",
        ):
            monkeypatch.delenv(var, raising=False)

        cfg = load_config(lambda key, default=None: None)

        assert cfg.auth_mode == "auto"
        assert cfg.http_host == "127.0.0.1"
        assert cfg.embed_model == "nomic-embed-text"
        assert cfg.collection_prefix == "Mem"
        assert cfg.domains == ("Episodic", "Semantic", "Procedural")
        assert cfg.tenant.startswith("u_")
        assert cfg.shared_tenant == "s_shared"

    def test_every_string_field_survives_a_none_returning_host(self):
        """No string field may end up None — each one is later .strip()ed or joined."""
        cfg = load_config(lambda key, default=None: None)
        for name in (
            "http_host",
            "grpc_host",
            "auth_mode",
            "tenant",
            "shared_tenant",
            "collection_prefix",
            "embed_model",
            "ollama_host",
            "api_key",
        ):
            assert isinstance(getattr(cfg, name), str), name

    def test_default_tenant_is_prefixed(self):
        tenant = default_tenant()
        assert tenant.startswith("u_")
        assert len(tenant) > 2

    def test_tenant_override_from_env(self, monkeypatch):
        monkeypatch.setenv("WEAVIATE_MEMORY_TENANT", "t_gruponos")
        assert load_config(None).tenant == "t_gruponos"

    def test_as_dict_round_trips_every_field(self):
        cfg = Config()
        data = cfg.as_dict()
        for key in ("http_host", "auth_mode", "multi_tenancy", "hybrid_alpha"):
            assert key in data


class TestForbiddenClassifier:
    """A 403 must be told apart from a real failure.

    Under RBAC a scoped role legitimately cannot see every collection; recall
    skips those and keeps going. A misclassified 500 would hide an outage.
    """

    @pytest.mark.parametrize(
        "message",
        [
            "UnexpectedStatusCodeError: status code: 403",
            "forbidden: user has no permission",
            "authorization error on collection MemSemantic",
            "rbac: missing permission read_data",
        ],
    )
    def test_authorization_failures_are_recognised(self, message):
        assert _is_forbidden(Exception(message))

    @pytest.mark.parametrize(
        "message",
        [
            "connection refused",
            "status code: 500 internal server error",
            "timeout waiting for response",
            "shard is read-only: disk pressure",
        ],
    )
    def test_operational_failures_are_not_permission_errors(self, message):
        assert not _is_forbidden(Exception(message))


class TestEmbedder:
    def test_known_model_dims_are_preloaded(self):
        assert OllamaEmbedder(model="nomic-embed-text").dims == 768
        assert OllamaEmbedder(model="all-minilm").dims == 384

    def test_unknown_model_learns_width_later(self):
        assert OllamaEmbedder(model="some-future-model").dims is None

    def test_known_dims_table_is_consistent(self):
        for model, dims in KNOWN_DIMS.items():
            assert dims > 0, model
            assert OllamaEmbedder(model=model).dims == dims

    def test_host_trailing_slash_is_normalised(self):
        assert OllamaEmbedder(host="http://x:11434/").host == "http://x:11434"


class TestProviderLogic:
    def test_name_is_stable(self):
        """The name is the config key operators type — it is an API."""
        assert WeaviateMemoryProvider(Config()).name == "weaviate-memory"

    def test_domain_resolution_is_case_insensitive(self):
        provider = WeaviateMemoryProvider(Config(domains=("Episodic", "Semantic")))
        assert provider._resolve_domain("episodic") == "Episodic"
        assert provider._resolve_domain("EPISODIC") == "Episodic"

    def test_unknown_domain_falls_back_to_semantic(self):
        provider = WeaviateMemoryProvider(Config(domains=("Episodic", "Semantic")))
        assert provider._resolve_domain("nonsense") == "Semantic"

    def test_missing_domain_falls_back_to_semantic(self):
        provider = WeaviateMemoryProvider(Config(domains=("Episodic", "Semantic")))
        assert provider._resolve_domain(None) == "Semantic"

    def test_fallback_uses_first_domain_when_semantic_absent(self):
        provider = WeaviateMemoryProvider(Config(domains=("Alpha", "Beta")))
        assert provider._resolve_domain("nope") == "Alpha"

    def test_tools_are_uninitialized_safe(self):
        """Calling a tool before initialize() must report, not crash."""
        provider = WeaviateMemoryProvider(Config())
        result = provider.handle_tool_call("weaviate_recall", {"query": "x"})
        assert "not initialized" in result

    def test_unknown_model_without_key_is_unavailable(self, monkeypatch):
        monkeypatch.delenv("WEAVIATE_MEMORY_API_KEY", raising=False)
        provider = WeaviateMemoryProvider(Config(api_key=""))
        assert provider.is_available() is False
        assert "WEAVIATE_MEMORY_API_KEY" in provider.unavailable_reason()

    def test_recall_status_is_consumed_once(self):
        """A stale count must never be reported on a later turn."""
        provider = WeaviateMemoryProvider(Config())
        provider._last_recall_count = 3
        first = provider.recall_status()
        assert first is not None and first.count == 3
        assert provider.recall_status() is None

    def test_prefetch_is_consumed_once(self):
        provider = WeaviateMemoryProvider(Config())
        provider._prefetched = "memories"
        assert provider.prefetch("q") == "memories"
        assert provider.prefetch("q") == ""

    def test_prefetch_returns_nothing_when_auto_recall_off(self):
        provider = WeaviateMemoryProvider(Config(auto_recall=False))
        provider._prefetched = "memories"
        assert provider.prefetch("q") == ""


class TestToolSchemas:
    def test_three_tools_are_exposed(self):
        names = {t["name"] for t in WeaviateMemoryProvider(Config()).get_tool_schemas()}
        assert names == {"weaviate_remember", "weaviate_recall", "weaviate_forget"}

    def test_schemas_are_openai_function_shaped(self):
        for schema in WeaviateMemoryProvider(Config()).get_tool_schemas():
            assert schema["parameters"]["type"] == "object"
            assert "properties" in schema["parameters"]
            assert schema["description"].strip()

    def test_required_fields_exist_in_properties(self):
        """A required field the model cannot see is a broken schema."""
        for schema in WeaviateMemoryProvider(Config()).get_tool_schemas():
            params = schema["parameters"]
            for field in params.get("required", []):
                assert field in params["properties"], f"{schema['name']}.{field}"


class TestAuthModeDetection:
    """The auth-mode probe must not be fooled by a permissive endpoint.

    Regression guard: an earlier version probed ``roles.list_all()`` and
    concluded "rbac" whenever the call succeeded. Measured against a real
    Weaviate 1.30.0 with RBAC DISABLED, that endpoint answers ``200 []`` —
    so the probe reported rbac on an admin_list server, and recall would have
    silently swallowed genuine failures as "permission denied".
    """

    class _FakeUser:
        def __init__(self, roles):
            self.user_id = "someone"
            self.roles = roles

    def _store_with_user(self, user, *, auth_mode="auto", raises=None):
        from weaviate_memory.store import WeaviateStore

        store = WeaviateStore(Config(auth_mode=auth_mode), 768)

        class FakeUsers:
            @staticmethod
            def get_my_user():
                if raises is not None:
                    raise raises
                return user

        class FakeClient:
            users = FakeUsers()

        store._client = FakeClient()
        return store

    def test_empty_roles_means_admin_list(self):
        store = self._store_with_user(self._FakeUser({}))
        assert store._detect_auth_mode() == "admin_list"

    def test_none_roles_means_admin_list(self):
        store = self._store_with_user(self._FakeUser(None))
        assert store._detect_auth_mode() == "admin_list"

    def test_assigned_roles_mean_rbac(self):
        store = self._store_with_user(self._FakeUser({"kb_reader": object()}))
        assert store._detect_auth_mode() == "rbac"

    def test_forbidden_probe_means_rbac(self):
        """A 403 proves the server enforces permissions."""
        store = self._store_with_user(None, raises=Exception("status code: 403 forbidden"))
        assert store._detect_auth_mode() == "rbac"

    def test_broken_probe_degrades_to_admin_list(self):
        store = self._store_with_user(None, raises=Exception("connection refused"))
        assert store._detect_auth_mode() == "admin_list"

    @pytest.mark.parametrize("declared", ["rbac", "admin_list"])
    def test_explicit_config_skips_the_probe_entirely(self, declared):
        """An operator override must never be second-guessed by a probe."""

        def explode():
            raise AssertionError("probe must not run when auth_mode is explicit")

        from weaviate_memory.store import WeaviateStore

        store = WeaviateStore(Config(auth_mode=declared), 768)
        store._client = None  # any probe would raise StoreError
        assert store._detect_auth_mode() == declared


class TestConfigSchema:
    def test_api_key_is_marked_secret_with_env_var(self):
        schema = WeaviateMemoryProvider(Config()).get_config_schema()
        api_key = next(f for f in schema if f["key"] == "api_key")
        assert api_key["secret"] is True
        assert api_key["env_var"] == "WEAVIATE_MEMORY_API_KEY"

    def test_auth_mode_offers_both_modes(self):
        schema = WeaviateMemoryProvider(Config()).get_config_schema()
        auth = next(f for f in schema if f["key"] == "auth_mode")
        assert set(auth["choices"]) == {"auto", "admin_list", "rbac"}

    def test_numeric_fields_declare_bounds(self):
        schema = WeaviateMemoryProvider(Config()).get_config_schema()
        for field in schema:
            if field.get("type") in ("integer", "number"):
                assert "minimum" in field and "maximum" in field, field["key"]

    def test_save_config_never_writes_the_key(self):
        """save_config must drop secrets even if handed one.

        The fake ``hermes_cli.config`` is installed and removed around the
        call: leaving it in ``sys.modules`` would poison every later import
        of the real module in the same session (it lacks ``cfg_get``), which
        silently broke an unrelated test file.
        """
        import sys
        import types

        written = {}
        provider = WeaviateMemoryProvider(Config())

        fake = types.ModuleType("hermes_cli.config")
        fake.cfg_set = lambda key, value: written.__setitem__(key, value)
        parent = types.ModuleType("hermes_cli")
        parent.config = fake

        had_parent = "hermes_cli" in sys.modules
        prev_parent = sys.modules.get("hermes_cli")
        prev_config = sys.modules.get("hermes_cli.config")

        sys.modules["hermes_cli"] = parent
        sys.modules["hermes_cli.config"] = fake
        try:
            provider.save_config({"api_key": "SECRET", "http_host": "h"}, os.getcwd())
        finally:
            if had_parent:
                sys.modules["hermes_cli"] = prev_parent
            else:
                sys.modules.pop("hermes_cli", None)
            if prev_config is not None:
                sys.modules["hermes_cli.config"] = prev_config
            else:
                sys.modules.pop("hermes_cli.config", None)

        assert not any("api_key" in key for key in written)
        assert written.get("plugins.weaviate-memory.http_host") == "h"

    def test_version_is_exposed(self):
        import weaviate_memory

        assert weaviate_memory.__version__
