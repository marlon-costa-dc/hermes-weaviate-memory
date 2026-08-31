# RBAC: layered access for shared Weaviate

This plugin runs in two authorization worlds. `admin_list` is where most
clusters start: one API key, broad rights. RBAC is where a shared cluster has
to end up.

The plugin detects which one is live and adapts — you can migrate from one to
the other without touching agent configuration.

## Why the mode is probed, not assumed

On Weaviate with RBAC **disabled**, `GET /v1/authz/roles` answers `200 []`
rather than an error. Verified against 1.30.0:

```console
$ curl -H "Authorization: Bearer $KEY" http://weaviate/v1/authz/roles
[]
$ curl -H "Authorization: Bearer $KEY" http://weaviate/v1/users/own-info
{"groups":null,"roles":null,"username":"dify@example.com"}
```

So "listing roles succeeded" does not mean RBAC is on. The identity's own
record does discriminate: `roles` is empty under `admin_list`, and under RBAC
any identity that can read the schema necessarily holds at least one role.

Getting this wrong is not cosmetic. If the plugin believes RBAC is active on an
`admin_list` server, it treats failures as permission denials and skips them
during recall — turning a real outage into silent, empty memory.

Set `auth_mode: rbac` or `auth_mode: admin_list` explicitly to skip the probe.

## Server configuration

RBAC is enabled server-side; the plugin cannot turn it on.

```yaml
# values.yaml for the Weaviate Helm chart
authentication:
  anonymous_access:
    enabled: false
  apikey:
    enabled: true
    users:
      - platform-admin
authorization:
  rbac:
    enabled: true
    root_users:
      - platform-admin
  admin_list:
    enabled: false        # mutually exclusive with RBAC
env:
  AUTHENTICATION_DB_USERS_ENABLED: "true"   # runtime user management, 1.30+
```

> On Weaviate ≤ 1.30.6 and ≤ 1.31.0, enabling RBAC **without** a valid root
> user prevents startup. Update the root key before rolling the change out.

## Roles

Names below use `Mem*` because that is this plugin's default
`collection_prefix`. Change one and change the other.

### A single operator's memory

Scoped to one tenant. This is the role an agent identity should hold.

```python
from weaviate.classes.rbac import Permissions

client.roles.create(
    role_name="agent_memory_alice",
    permissions=[
        Permissions.collections(collection="Mem*", read_config=True),
        Permissions.tenants(
            collection="Mem*",
            tenant="u_alice",
            create=True,
            read=True,
            update=True,
        ),
        Permissions.data(
            collection="Mem*",
            tenant="u_alice",
            create=True,
            read=True,
            update=True,
            delete=True,
        ),
    ],
)
```

Note what is absent: no `create_collection`. The agent uses collections an
administrator provisioned. The plugin expects this — it records that its schema
access is read-only and reports what is missing rather than failing.

### Read-only access to shared knowledge

For MCP clients and consumers that must not write.

```python
client.roles.create(
    role_name="kb_reader",
    permissions=[
        Permissions.collections(collection="Mem*", read_config=True),
        Permissions.data(collection="Mem*", tenant="s_shared", read=True),
        Permissions.tenants(collection="Mem*", tenant="s_shared", read=True),
    ],
)
```

### Provisioning automation

```python
client.roles.create(
    role_name="tenant_manager",
    permissions=[
        Permissions.collections(
            collection="Mem*",
            create_collection=True,
            read_config=True,
            update_config=True,
        ),
        Permissions.tenants(
            collection="Mem*",
            tenant="*",
            create=True,
            read=True,
            update=True,
            delete=True,
        ),
    ],
)
```

**Never** grant role-management permissions to an agent identity. A role that
can assign roles can escalate itself.

## Users

One DB user per system, each with its own rotatable key:

```python
client.users.db.create(user_id="hermes-alice")
client.users.db.assign_roles(user_id="hermes-alice", role_names=["agent_memory_alice"])
```

Store the returned key in your secret manager and hand it to the agent as
`WEAVIATE_MEMORY_API_KEY`. Rotation is `client.users.db.rotate_key()` followed
by updating the secret — no schema or data changes.

## Provisioning script

`scripts/provision_rbac.py` creates the collections, tenants, roles and users
for one operator. Dry-run by default:

```bash
python scripts/provision_rbac.py --user alice          # shows the plan
python scripts/provision_rbac.py --user alice --apply  # executes it
```

It needs a root-capable key in `WEAVIATE_ADMIN_API_KEY` — deliberately a
different variable from the agent's own key, so an agent credential can never
be used to provision.

## Two things RBAC does not do

**Collection permissions are not tenant permissions.** Being allowed to create
a collection named `MemSemantic` does not allow creating tenants inside it.
Grant both.

**`False` means "unset", not "denied".** If one role grants read and another
sets it `False`, the user can read. There is no deny rule — model access by
granting narrowly, never by trying to subtract.
