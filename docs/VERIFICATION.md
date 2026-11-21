# Verification log

Claims in this repository's docs are backed by measurements, not by reading
prose. This file records what was run, against what, and what came back.

The `tests/_probe_*.py` scripts are ad-hoc probes, not part of the test suite.
They target a disposable local Weaviate and are kept so anyone can reproduce
the findings rather than take them on trust.

## The lab

```bash
docker run -d --name wv-lab --network host --env-file lab.env \
  cr.weaviate.io/semitechnologies/weaviate:1.39.0 \
  --host 0.0.0.0 --port 18090 --scheme http
```

`lab.env`:

```
AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=false
AUTHENTICATION_APIKEY_ENABLED=true
AUTHENTICATION_APIKEY_ALLOWED_KEYS=root-key,reader-key,agent-key
AUTHENTICATION_APIKEY_USERS=root-user,reader-user,agent-user
AUTHENTICATION_DB_USERS_ENABLED=true
AUTHORIZATION_RBAC_ENABLED=true
AUTHORIZATION_RBAC_ROOT_USERS=root-user
MCP_SERVER_ENABLED=true
MCP_SERVER_WRITE_ACCESS_ENABLED=false
ENABLE_MODULES=text2vec-ollama
DEFAULT_VECTORIZER_MODULE=none
PERSISTENCE_DATA_PATH=/var/lib/weaviate
DISABLE_TELEMETRY=true
```

Use `--env-file`, not repeated `-e` flags: some shells and agent runtimes
scrub the subprocess environment, and a silently dropped `ENABLE_MODULES`
looks exactly like a module that failed to load.

## 1. Self-provided vectors cannot serve string-only search

`tests/_probe_vectorizer.py`, Weaviate 1.39.0:

| Collection | Query form | Result |
|---|---|---|
| `vectorizer: none` | `hybrid(query=str)` | **fails** — `VectorFromInput was called without vectorizer` |
| `vectorizer: none` | `bm25(query=str)` | works |
| `vectorizer: none` | `near_vector(client vector)` | works |
| `text2vec-ollama` | insert without a vector | works |
| `text2vec-ollama` | `hybrid(query=str)` | works |
| `text2vec-ollama` | `near_text(query=str)` | works |

The same failure was first observed on Weaviate **1.30.0**, so it is not a
1.39 regression.

**Consequence.** This plugin always sends a vector, so it is unaffected. Any
consumer that sends only a string — the MCP server, and most other systems —
gets keyword search and nothing more, unless the collection has a server-side
vectorizer.

## 2. The MCP server inherits that limitation

`tests/_probe_mcp.py` speaks raw MCP (Streamable HTTP + JSON-RPC) to
`/v1/mcp`, exactly as an MCP client would.

Tools exposed with `MCP_SERVER_WRITE_ACCESS_ENABLED=false` — three, all
read-only:

```
weaviate-collections-get-config   Retrieves collection configuration(s)
weaviate-query-hybrid             Performs hybrid search (vector + keyword)
weaviate-tenants-list             Lists the tenants of a collection
```

Calling `weaviate-query-hybrid`:

```
on a vectorizer:none collection
  → tool execution failed: failed to execute hybrid search:
    VectorFromInput was called without vectorizer on class McpByo

on a text2vec-ollama collection
  → {"results":[{"_additional":{"score":0.0125,...},"content":"The oper...
```

**Consequence.** A server-side vectorizer is a prerequisite for useful MCP
search, not an optimisation.

## 3. RBAC denials are typed, and the mode probe is honest

With RBAC enabled and a `mem_only` role granting read on `Mem*` only:

- `reader-key` reading `MemSemantic` → allowed
- `reader-key` reading `SecretStuff` → `PermissionDeniedError`

`test_permission_errors_are_typed_not_generic` covers this. It requires
`WEAVIATE_IT_RESTRICTED_KEY` and `WEAVIATE_IT_FORBIDDEN_COLLECTION`, and skips
without them — an earlier version used the *root* key and passed for the wrong
reason (root gets "collection not found", never a denial). A test that cannot
fail proves nothing.

Auth-mode detection, measured on both servers:

| Server | `/v1/users/own-info` → `roles` | Detected |
|---|---|---|
| 1.30.0, RBAC off | `null` | `admin_list` |
| 1.39.0, RBAC on | populated (`root`) | `rbac` |

Note `GET /v1/authz/roles` returns `200 []` when RBAC is **off** — which is
why the probe reads the identity's own record instead. A successful call to
that endpoint proves nothing.

## 4. Version facts, read from source

The docs contradict themselves on several points; these were settled by
reading Weaviate's Go source at the release tags.

`usecases/config/authorization.go` — identical at v1.30.0 and v1.39.0:

```go
if a.AdminList.Enabled && a.Rbac.Enabled {
    return fmt.Errorf("cannot enable adminlist and rbac at the same time")
}
```

The pod refuses to start. Migration between the two is an atomic cut.

`usecases/config/environment.go` — both spellings accepted, at both tags:

```go
if entcfg.Enabled(os.Getenv("AUTHORIZATION_ENABLE_RBAC")) ||
   entcfg.Enabled(os.Getenv("AUTHORIZATION_RBAC_ENABLED")) {
```

`rbacconf.Config.Validate()`:

- v1.30.0 → `"at least one root user is required"` (RBAC without a root user
  will not boot)
- v1.39.0 → `return nil` (boots without one)

`MCP_SERVER_*` is absent from the 1.30.0 source and present in 1.39.0.

## 5. gRPC is required for multi-tenancy

With the gRPC port unreachable, tenant creation fails:

```
StoreError: cannot ensure tenant on '...': failed to connect to all addresses;
last error: UNKNOWN: ipv4:127.0.0.1:50051: Connection refused
```

An HTTP-only ingress cannot serve multi-tenancy. The v4 client routes queries
*and* tenant operations over gRPC. The docs state the client needs gRPC but do
not enumerate tenant operations as gRPC-only; this was measured.

## Suite status

Against the 1.39.0 lab with RBAC enabled: **107 passed, 0 skipped**.
Against a 1.30.0 cluster with `admin_list`: 106 passed, 1 skipped (the RBAC
denial test, correctly — there are no denials to observe on that server).
