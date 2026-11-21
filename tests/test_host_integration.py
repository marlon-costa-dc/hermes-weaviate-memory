"""Host-integration tests: the extension points Hermes discovers by convention.

These assert the *wiring*, not the behaviour — that Hermes' loader can find
this plugin, its config panel, and its CLI. Convention-based discovery fails
silently when a name is wrong, so each convention gets an explicit test.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from conftest import HERMES_AVAILABLE

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = REPO_ROOT / "weaviate_memory"
PROVIDER_NAME = "weaviate-memory"


def _load_by_path(name: str, path: Path):
    """Load a module from a path, exactly as the host does."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestPluginDiscovery:
    """Hermes text-scans __init__.py before importing anything."""

    def test_init_carries_a_discovery_marker(self):
        source = (PACKAGE_DIR / "__init__.py").read_text(encoding="utf-8")
        assert "register_memory_provider" in source or "MemoryProvider" in source

    def test_plugin_yaml_name_matches_the_directory_convention(self):
        import yaml

        meta = yaml.safe_load((REPO_ROOT / "plugin.yaml").read_text(encoding="utf-8"))
        assert meta["name"] == PROVIDER_NAME
        assert meta["description"].strip()

    def test_plugin_yaml_declares_its_dependency_and_secret(self):
        import yaml

        meta = yaml.safe_load((REPO_ROOT / "plugin.yaml").read_text(encoding="utf-8"))
        assert any("weaviate-client" in dep for dep in meta["pip_dependencies"])
        assert "WEAVIATE_MEMORY_API_KEY" in meta["requires_env"]

    def test_entry_point_group_is_the_one_hermes_scans(self):
        """A typo here means pip installs silently do not register."""
        content = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert 'entry-points."hermes_agent.memory_providers"' in content
        assert f'{PROVIDER_NAME} = "weaviate_memory"' in content


class TestCliRegistration:
    """``discover_plugin_cli_commands`` resolves the handler by name."""

    @pytest.fixture
    def cli_module(self):
        return _load_by_path("weaviate_memory_cli_undertest", PACKAGE_DIR / "cli.py")

    def test_register_cli_exists(self, cli_module):
        assert callable(cli_module.register_cli)

    def test_hyphenated_handler_name_is_bound(self, cli_module):
        """THE reason this test file exists.

        The host looks the handler up as ``getattr(mod, f"{name}_command")``.
        Our provider name contains a hyphen, so ``weaviate-memory_command`` is
        not a valid Python identifier and cannot be written as a ``def`` — it
        has to be bound through ``globals()``. Get this wrong and the CLI
        silently never appears, with no error anywhere.
        """
        attr = f"{PROVIDER_NAME}_command"
        assert not attr.isidentifier(), "precondition: the name must be hyphenated"
        assert hasattr(cli_module, attr), f"host would find no handler at {attr!r}"
        assert callable(getattr(cli_module, attr))

    def test_both_handler_spellings_are_the_same_function(self, cli_module):
        assert getattr(cli_module, f"{PROVIDER_NAME}_command") is (
            cli_module.weaviate_memory_command
        )

    def test_register_cli_builds_the_documented_subcommands(self, cli_module):
        import argparse

        parser = argparse.ArgumentParser(prog="hermes weaviate-memory")
        cli_module.register_cli(parser)

        for command in ("status", "doctor", "search", "tenants"):
            argv = [command, "q"] if command == "search" else [command]
            args = parser.parse_args(argv)
            assert args.weaviate_memory_command == command

    def test_dispatch_without_a_subcommand_does_not_crash(self, cli_module):
        import argparse

        assert cli_module._dispatch(argparse.Namespace()) == 1

    def test_cli_imports_without_the_weaviate_sdk_at_module_scope(self):
        """The host imports cli.py at argparse time, before deps are guaranteed."""
        source = (PACKAGE_DIR / "cli.py").read_text(encoding="utf-8")
        module_level = [
            line
            for line in source.splitlines()
            if line.startswith(("import ", "from ")) and "weaviate" in line
        ]
        assert not module_level, f"weaviate imported at module scope: {module_level}"


class TestConfigSchemaPanel:
    """The dashboard loads config_schema.py by path, never as a package."""

    @pytest.fixture
    def schema_module(self):
        return _load_by_path("weaviate_memory_schema_undertest", PACKAGE_DIR / "config_schema.py")

    def test_loads_outside_the_host_without_the_agent_runtime(self, schema_module):
        """It must import cleanly in the web process, which has no agent runtime."""
        assert hasattr(schema_module, "CONFIG_SCHEMA")

    @pytest.mark.skipif(not HERMES_AVAILABLE, reason="needs the real host schema classes")
    def test_declares_a_usable_panel(self, schema_module):
        schema = schema_module.CONFIG_SCHEMA
        assert schema is not None
        assert schema.name == PROVIDER_NAME
        assert schema.fields
        assert schema.inline_fields(), "an empty compact panel is a broken panel"

    @pytest.mark.skipif(not HERMES_AVAILABLE, reason="needs the real host schema classes")
    def test_the_api_key_is_the_only_secret_and_routes_to_env(self, schema_module):
        secrets = [f for f in schema_module.CONFIG_SCHEMA.fields if f.is_secret]
        assert [f.key for f in secrets] == ["api_key"]
        assert secrets[0].env_key == "WEAVIATE_MEMORY_API_KEY"

    @pytest.mark.skipif(not HERMES_AVAILABLE, reason="needs the real host schema classes")
    def test_every_field_maps_to_a_real_config_attribute(self, schema_module):
        """A field the loader never reads is a control that silently does nothing."""
        from weaviate_memory.config import Config

        config_fields = set(Config().as_dict()) | {"api_key"}
        for field in schema_module.CONFIG_SCHEMA.fields:
            assert field.key in config_fields, f"{field.key} is not read by load_config"

    @pytest.mark.skipif(not HERMES_AVAILABLE, reason="needs the real host schema classes")
    def test_select_options_match_what_the_code_accepts(self, schema_module):
        auth = next(f for f in schema_module.CONFIG_SCHEMA.fields if f.key == "auth_mode")
        assert auth.allowed_values() == {"auto", "admin_list", "rbac"}
