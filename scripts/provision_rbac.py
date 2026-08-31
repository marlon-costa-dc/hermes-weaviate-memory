#!/usr/bin/env python3
"""Provision collections, tenants, roles and users for one operator.

Dry-run by default: it prints the plan and changes nothing. Pass --apply to
execute.

Requires a root-capable key in WEAVIATE_ADMIN_API_KEY — deliberately a
different variable from the agent's own WEAVIATE_MEMORY_API_KEY, so an agent
credential can never be used to provision infrastructure.

    export WEAVIATE_ADMIN_API_KEY=...
    python scripts/provision_rbac.py --host weaviate.example.com --user alice
    python scripts/provision_rbac.py --host weaviate.example.com --user alice --apply

Idempotent: existing collections, tenants and roles are left alone.
"""

from __future__ import annotations

import argparse
import os
import sys

DEFAULT_DOMAINS = ("Episodic", "Semantic", "Procedural")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--host", default=os.environ.get("WEAVIATE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("WEAVIATE_PORT", "8080")))
    parser.add_argument("--secure", action="store_true", help="use HTTPS/TLS")
    parser.add_argument("--grpc-host", default="", help="defaults to --host")
    parser.add_argument("--grpc-port", type=int, default=50051)
    parser.add_argument("--user", required=True, help="operator login, e.g. alice")
    parser.add_argument("--prefix", default="Mem", help="collection prefix (default: Mem)")
    parser.add_argument("--domains", default=",".join(DEFAULT_DOMAINS))
    parser.add_argument("--replication", type=int, default=1)
    parser.add_argument("--shared-tenant", default="s_shared")
    parser.add_argument("--apply", action="store_true", help="execute (default: dry run)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    admin_key = os.environ.get("WEAVIATE_ADMIN_API_KEY", "")
    if not admin_key:
        print("error: WEAVIATE_ADMIN_API_KEY is not set", file=sys.stderr)
        return 2

    import weaviate
    from weaviate.classes.config import Configure, DataType, Property
    from weaviate.classes.init import Auth
    from weaviate.classes.rbac import Permissions
    from weaviate.classes.tenants import Tenant

    domains = tuple(d.strip() for d in args.domains.split(",") if d.strip())
    collections = [f"{args.prefix}{d}" for d in domains]
    tenant = f"u_{args.user}"
    role = f"agent_memory_{args.user}"
    db_user = f"hermes-{args.user}"
    mode = "APPLY" if args.apply else "DRY RUN"

    print(f"[{mode}] target {args.host}:{args.port}")
    print(f"  collections : {', '.join(collections)}")
    print(f"  tenants     : {tenant}, {args.shared_tenant}")
    print(f"  role        : {role} (scoped to {args.prefix}* / {tenant})")
    print(f"  db user     : {db_user}")
    if not args.apply:
        print("\nnothing changed. re-run with --apply to execute.")
        return 0

    client = weaviate.connect_to_custom(
        http_host=args.host,
        http_port=args.port,
        http_secure=args.secure,
        grpc_host=args.grpc_host or args.host,
        grpc_port=args.grpc_port,
        grpc_secure=args.secure,
        auth_credentials=Auth.api_key(admin_key),
    )

    try:
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

        for name in collections:
            if client.collections.exists(name):
                print(f"  = collection {name} already exists")
            else:
                kwargs = {
                    "name": name,
                    "description": "Hermes agent memory",
                    "properties": properties,
                    "vector_config": Configure.Vectors.self_provided(),
                    "multi_tenancy_config": Configure.multi_tenancy(
                        enabled=True, auto_tenant_creation=False
                    ),
                }
                if args.replication > 1:
                    kwargs["replication_config"] = Configure.replication(factor=args.replication)
                client.collections.create(**kwargs)
                print(f"  + created collection {name}")

            collection = client.collections.get(name)
            existing = set(collection.tenants.get())
            for wanted in (tenant, args.shared_tenant):
                if wanted in existing:
                    print(f"  = tenant {wanted} already on {name}")
                else:
                    collection.tenants.create([Tenant(name=wanted)])
                    print(f"  + created tenant {wanted} on {name}")

        pattern = f"{args.prefix}*"
        permissions = [
            Permissions.collections(collection=pattern, read_config=True),
            Permissions.tenants(
                collection=pattern, tenant=tenant, create=True, read=True, update=True
            ),
            Permissions.data(
                collection=pattern,
                tenant=tenant,
                create=True,
                read=True,
                update=True,
                delete=True,
            ),
            # Read-only on the shared tenant: an operator may consult
            # organisation knowledge but not rewrite it.
            Permissions.data(collection=pattern, tenant=args.shared_tenant, read=True),
            Permissions.tenants(collection=pattern, tenant=args.shared_tenant, read=True),
        ]

        if client.roles.exists(role):
            print(f"  = role {role} already exists")
        else:
            client.roles.create(role_name=role, permissions=permissions)
            print(f"  + created role {role}")

        try:
            api_key = client.users.db.create(user_id=db_user)
            print(f"  + created db user {db_user}")
            print(f"\n  API KEY for {db_user} (store it now, it is not shown again):")
            print(f"  {api_key}")
        except Exception as exc:
            if "already exists" in str(exc).lower():
                print(f"  = db user {db_user} already exists (rotate the key if needed)")
            else:
                raise

        client.users.db.assign_roles(user_id=db_user, role_names=[role])
        print(f"  + assigned {role} to {db_user}")

        print("\ndone. Configure the agent with:")
        print("  WEAVIATE_MEMORY_API_KEY=<the key above>")
        print(f"  plugins.weaviate-memory.tenant: {tenant}")
        print(f"  plugins.weaviate-memory.collection_prefix: {args.prefix}")
    finally:
        client.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
