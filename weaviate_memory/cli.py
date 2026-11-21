"""``hermes weaviate-memory`` — operational CLI for the active provider.

Hermes discovers this file through ``discover_plugin_cli_commands()`` and only
for the provider named in ``memory.provider``, so these subcommands appear
exactly when they are useful.

Two host details shape this module:

1. The handler is looked up as ``getattr(cli_mod, f"{provider_name}_command")``.
   Our provider name contains a hyphen, so ``weaviate-memory_command`` is not a
   valid Python identifier and cannot be written as a ``def``. It is bound
   into the module namespace explicitly at the bottom of this file.

2. This module is imported at argparse time, before any provider is loaded, so
   it must stay cheap: no ``weaviate`` import at module scope.
"""

from __future__ import annotations

import os
import sys


def register_cli(subparser) -> None:
    """Build the ``hermes weaviate-memory`` subcommand tree."""
    subs = subparser.add_subparsers(dest="weaviate_memory_command")

    subs.add_parser("status", help="Show configuration and live connection status")

    subs.add_parser("doctor", help="Diagnose connectivity, auth mode and schema")

    search = subs.add_parser("search", help="Search memories from the shell")
    search.add_argument("query", help="What to search for")
    search.add_argument("--domain", default="", help="Restrict to one domain")
    search.add_argument("--limit", type=int, default=10)

    tenants = subs.add_parser("tenants", help="List tenants on the memory collections")
    tenants.add_argument("--domain", default="", help="Restrict to one domain")


def _load():
    """Import the provider package regardless of how it was installed."""
    try:
        from weaviate_memory import WeaviateMemoryProvider, load_config
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from __init__ import WeaviateMemoryProvider  # type: ignore[no-redef]
        from config import load_config  # type: ignore[no-redef]

    try:
        from hermes_cli.config import cfg_get
    except ImportError:
        cfg_get = None
    return WeaviateMemoryProvider, load_config(cfg_get)


def _cmd_status(_args) -> int:
    provider_cls, config = _load()
    print(
        f"\n  endpoint    {config.http_host}:{config.http_port}"
        f" ({'https' if config.http_secure else 'http'})"
    )
    print(f"  grpc        {config.resolved_grpc_host()}:{config.grpc_port}")
    print(f"  auth mode   {config.auth_mode}")
    print(f"  tenant      {config.tenant if config.multi_tenancy else '(multi-tenancy off)'}")
    print(f"  collections {', '.join(config.collections())}")
    print(f"  embeddings  {config.embed_model} via {config.ollama_host}")
    print(f"  api key     {'set' if config.api_key else 'NOT SET'}")

    provider = provider_cls(config)
    if not provider.is_available():
        print(f"\n  unavailable: {provider.unavailable_reason()}\n")
        return 1

    provider.initialize("cli-status")
    try:
        totals = provider._store.totals()
        detected = provider._store.auth_mode
        print(f"\n  connected. server authorization: {detected}")
        if provider._store.readonly_schema:
            print("  note: this identity cannot create collections/tenants (RBAC).")
        if totals:
            for name, count in sorted(totals.items()):
                print(f"    {name}: {count}")
        else:
            print("    (no memories yet)")
        print()
    finally:
        provider.shutdown()
    return 0


def _cmd_doctor(_args) -> int:
    """Check each dependency in turn and report the first real blocker."""
    provider_cls, config = _load()
    failures = 0

    try:
        import weaviate  # noqa: F401

        print("  [ok]   weaviate-client importable")
    except ImportError:
        print("  [FAIL] weaviate-client missing — pip install 'weaviate-client>=4.22,<5'")
        return 1

    from weaviate_memory.embeddings import OllamaEmbedder

    embedder = OllamaEmbedder(config.ollama_host, config.embed_model)
    if embedder.is_available():
        print(f"  [ok]   ollama serving {config.embed_model}")
    else:
        print(f"  [FAIL] ollama at {config.ollama_host} is not serving {config.embed_model}")
        print(f"         run: ollama pull {config.embed_model}")
        failures += 1

    if config.api_key:
        print("  [ok]   API key present")
    else:
        print("  [FAIL] WEAVIATE_MEMORY_API_KEY is not set")
        failures += 1

    if failures:
        print()
        return 1

    provider = provider_cls(config)
    provider.initialize("cli-doctor")
    try:
        store = provider._store
        meta = store.client.get_meta()
        print(f"  [ok]   connected to Weaviate {meta['version']}")
        print(f"  [ok]   authorization mode: {store.auth_mode}")

        missing = [n for n in config.collections() if not store._exists_quiet(n)]
        if missing:
            print(f"  [warn] collections not provisioned: {', '.join(missing)}")
            if store.readonly_schema:
                print("         this identity may not create them — ask an administrator")
        else:
            print(f"  [ok]   all {len(config.collections())} collections present")

        vector = provider._embedder.embed_one("doctor probe")
        print(f"  [ok]   embeddings working ({len(vector)} dimensions)")
    finally:
        provider.shutdown()
    print()
    return 0


def _cmd_search(args) -> int:
    provider_cls, config = _load()
    provider = provider_cls(config)
    if not provider.is_available():
        print(f"  unavailable: {provider.unavailable_reason()}")
        return 1
    provider.initialize("cli-search")
    try:
        payload = {"query": args.query, "limit": args.limit}
        if args.domain:
            payload["domain"] = args.domain
        print(provider.handle_tool_call("weaviate_recall", payload))
    finally:
        provider.shutdown()
    return 0


def _cmd_tenants(args) -> int:
    provider_cls, config = _load()
    provider = provider_cls(config)
    if not provider.is_available():
        print(f"  unavailable: {provider.unavailable_reason()}")
        return 1
    provider.initialize("cli-tenants")
    try:
        store = provider._store
        targets = [config.collection_for(args.domain)] if args.domain else config.collections()
        for name in targets:
            if not store._exists_quiet(name):
                print(f"  {name}: (not provisioned)")
                continue
            try:
                tenants = sorted(store.client.collections.get(name).tenants.get())
                print(f"  {name}: {', '.join(tenants) if tenants else '(none)'}")
            except Exception as exc:
                print(f"  {name}: cannot list tenants — {exc}")
    finally:
        provider.shutdown()
    return 0


_HANDLERS = {
    "status": _cmd_status,
    "doctor": _cmd_doctor,
    "search": _cmd_search,
    "tenants": _cmd_tenants,
}


def _dispatch(args) -> int:
    command = getattr(args, "weaviate_memory_command", None)
    if not command:
        print("\n  usage: hermes weaviate-memory {status,doctor,search,tenants}\n")
        return 1
    handler = _HANDLERS.get(command)
    if handler is None:
        print(f"  unknown subcommand: {command}")
        return 1
    return handler(args)


# The host resolves the handler as ``getattr(module, f"{provider}_command")``.
# "weaviate-memory_command" cannot be written as a `def` (the hyphen makes it
# an invalid identifier), so it is bound here instead.
globals()["weaviate-memory_command"] = _dispatch

# Also exposed under a valid identifier, for direct imports and testing.
weaviate_memory_command = _dispatch
