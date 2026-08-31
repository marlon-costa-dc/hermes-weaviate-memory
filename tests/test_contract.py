"""Contract tests — the provider must satisfy Hermes' REAL MemoryProvider ABC.

These skip when only the stub from conftest is available. That is deliberate:
a stub can only prove the provider matches the stub. Run them on a machine
with Hermes installed (or set HERMES_AGENT_PATH) to prove the real contract.
"""

from __future__ import annotations

import inspect

import pytest
from conftest import HERMES_AVAILABLE

from weaviate_memory import WeaviateMemoryProvider
from weaviate_memory.config import Config

pytestmark = pytest.mark.skipif(
    not HERMES_AVAILABLE,
    reason="real Hermes not installed; set HERMES_AGENT_PATH to run contract tests",
)


@pytest.fixture
def provider():
    return WeaviateMemoryProvider(Config())


def test_is_a_memory_provider(provider):
    from agent.memory_provider import MemoryProvider

    assert isinstance(provider, MemoryProvider)


def test_no_abstract_methods_left_unimplemented(provider):
    """Instantiation would already fail, but the message here is clearer."""
    assert not getattr(type(provider), "__abstractmethods__", set())


@pytest.mark.parametrize(
    "method",
    [
        "is_available",
        "initialize",
        "get_tool_schemas",
        "handle_tool_call",
        "prefetch",
        "queue_prefetch",
        "sync_turn",
        "shutdown",
        "system_prompt_block",
        "unavailable_reason",
        "recall_status",
        "on_memory_write",
        "get_config_schema",
        "save_config",
    ],
)
def test_signature_matches_the_host_abc(provider, method):
    """A host-side signature change must fail here, not at runtime.

    The host calls these with keyword arguments; a drifted signature would
    raise TypeError mid-conversation, after the agent already believed its
    memory was working.
    """
    from agent.memory_provider import MemoryProvider

    base = getattr(MemoryProvider, method, None)
    if base is None:
        pytest.skip(f"host ABC has no {method}")

    ours = getattr(type(provider), method)
    base_params = inspect.signature(base).parameters
    our_params = inspect.signature(ours).parameters

    for name, param in base_params.items():
        if name == "self":
            continue
        if param.kind in (param.VAR_KEYWORD, param.VAR_POSITIONAL):
            continue
        assert name in our_params, f"{method}() is missing parameter {name!r}"


def test_recall_status_returns_the_hosts_type(provider):
    from agent.memory_provider import RecallStatus

    provider._last_recall_count = 2
    status = provider.recall_status()
    assert isinstance(status, RecallStatus)
    assert status.count == 2


def test_trivial_prompt_gate_is_the_hosts_own(provider):
    """Recall must skip the same prompts the host considers trivial.

    Importing the host's classifier rather than reimplementing it is what
    keeps the two from drifting apart.
    """
    from agent.memory_provider import is_trivial_prompt

    assert is_trivial_prompt("ok")
    assert is_trivial_prompt("thanks!")
    assert not is_trivial_prompt("what distro does the operator run")


def test_provider_is_discoverable_by_the_plugin_loader():
    """Hermes' loader text-scans __init__.py for these markers."""
    from pathlib import Path

    source = (Path(__file__).resolve().parent.parent / "weaviate_memory" / "__init__.py").read_text(
        encoding="utf-8"
    )
    assert "register_memory_provider" in source or "MemoryProvider" in source


def test_tool_schemas_are_accepted_shape(provider):
    """Schemas go straight into the model's tool list; malformed ones break the turn."""
    for schema in provider.get_tool_schemas():
        assert isinstance(schema["name"], str) and schema["name"]
        assert isinstance(schema["description"], str) and schema["description"]
        assert isinstance(schema["parameters"], dict)
