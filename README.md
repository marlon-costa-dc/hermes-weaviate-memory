# hermes-weaviate-memory

Layered [Weaviate](https://weaviate.io) memory for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

A collection per **domain**, a tenant per **user or team**, and an authorization
mode that follows your server — `admin_list` today, RBAC when you turn it on.
Embeddings are computed locally through [Ollama](https://ollama.com), so memory
text only ever reaches the Weaviate cluster you point it at.

Works alongside Weaviate's built-in MCP server (Weaviate ≥ 1.38), so the same
data is reachable both as automatic agent memory and as an on-demand tool from
any MCP client. See [docs/MCP.md](docs/MCP.md).

## Why this exists

Hermes' built-in memory is a pair of markdown files with a hard character
budget. That works for one operator on one machine. It does not work when:

- several people or teams share one knowledge base but must not read each
  other's notes;
- memories belong to different **domains** (what happened, what is true, how to
  do something) that deserve separate retrieval;
- the vector store is shared infrastructure that other systems (Dify, RAG
  pipelines, internal tools) also write to;
- access has to be auditable and least-privilege.

Weaviate already solves the storage half. This plugin supplies the other half:
the memory contract Hermes expects, mapped onto Weaviate's isolation model.

## The four layers

| Layer | Weaviate primitive | Example |
|---|---|---|
| **Domain** | collection | `MemEpisodic`, `MemSemantic`, `MemProcedural` |
| **Identity** | tenant | `u_alice`, `t_platform`, `s_shared` |
| **Permission** | RBAC role | `agent_memory_alice` scoped to `Mem*` + `u_alice` |
| **Credential** | DB user + API key | one key per system, rotatable |

Domains are searched together by default — a procedural memory often answers
what looks like a semantic question — but each is a separate collection, so an
RBAC role can grant access to one and not another.

Tenants are the hard isolation boundary. Each is a separate shard; a query
scoped to one tenant physically cannot return another's objects. This is
[covered by a test](tests/test_integration.py) that asserts both directions:
the write is visible to its owner *and* invisible to everyone else.

## Requirements

- Hermes Agent
- Weaviate 1.25+ (RBAC needs 1.29+, the built-in MCP server needs 1.38+)
- Ollama with an embedding model (`ollama pull nomic-embed-text`)
- A **gRPC route** to Weaviate. The v4 client sends queries *and* tenant
  operations over gRPC — a cluster exposing only HTTP cannot serve
  multi-tenancy.

## Install

```bash
git clone https://github.com/marlon-costa-dc/hermes-weaviate-memory
cd hermes-weaviate-memory
./scripts/install.sh
```

Or as a package, using the entry point:

```bash
pip install hermes-weaviate-memory
```

Then:

```bash
echo 'WEAVIATE_MEMORY_API_KEY=your-key' >> ~/.hermes/.env
hermes memory setup      # pick weaviate-memory
hermes memory status
```

## Configuration

Behaviour goes in `~/.hermes/config.yaml`; the API key goes in `~/.hermes/.env`
and is never written to config.

```yaml
memory:
  provider: weaviate-memory

plugins:
  weaviate-memory:
    http_host: weaviate.internal.example.com
    http_port: 443
    http_secure: true
    grpc_host: weaviate-grpc.internal.example.com
    grpc_port: 443
    grpc_secure: true

    auth_mode: auto            # auto | admin_list | rbac
    multi_tenancy: true
    tenant: ""                 # defaults to u_<login>
    shared_tenant: s_shared
    collection_prefix: Mem
    domains: [Episodic, Semantic, Procedural]
    replication_factor: 3

    embed_model: nomic-embed-text
    ollama_host: http://127.0.0.1:11434

    auto_recall: true
    recall_limit: 5
    min_certainty: 0.6
    hybrid_alpha: 0.7          # 1.0 vector-only, 0.0 keyword-only
    auto_capture_turns: false
    mirror_memory_tool: true
```

Every key can be overridden by an environment variable
(`WEAVIATE_MEMORY_HTTP_HOST`, `WEAVIATE_MEMORY_TENANT`, …), which is what the
integration suite uses.

### Authorization modes

`auth_mode: auto` probes the server and adapts. The probe reads the identity's
own record rather than listing roles — on a server with RBAC **disabled**,
`GET /v1/authz/roles` returns `200 []` rather than an error, so a successful
call proves nothing. Measured against Weaviate 1.30.0.

- **admin_list** — one API key with broad rights. The plugin creates its
  collections and tenants on demand. Fine for a single operator.
- **rbac** — the key maps to a DB user with a scoped role. Schema creation is
  usually *not* granted, so the plugin uses what exists, records that its
  schema access is read-only, and reports precisely what an administrator must
  provision. Denials are typed (`PermissionDeniedError`) and skipped during
  recall instead of aborting the turn.

[docs/RBAC.md](docs/RBAC.md) has the roles and a provisioning script.

## Tools exposed to the model

| Tool | Purpose |
|---|---|
| `weaviate_remember` | Store one durable memory under a domain |
| `weaviate_recall` | Hybrid (vector + keyword) search across domains |
| `weaviate_forget` | Delete a memory by id |

Beyond the tools, the provider injects relevant memories **automatically**
before each turn (background prefetch, so the turn thread never blocks on an
embedding) and can mirror Hermes' built-in `MEMORY.md` writes into Weaviate —
useful because that file has a character budget and Weaviate does not.

## Hybrid search

Recall uses Weaviate's hybrid search rather than pure vector similarity. Agent
memories mix prose with literal tokens — error codes, file paths, flag names —
and embeddings blur exactly those. `hybrid_alpha` tunes the balance; the
default 0.7 keeps semantic recall dominant while still matching
`RAFT_BOOTSTRAP_EXPECT` verbatim.

Automatic recall uses pure vector search instead, because its `min_certainty`
floor must be comparable across queries; hybrid scores are relative to a
result set and cannot be thresholded that way.

## Testing

```bash
pytest tests/test_unit.py tests/test_contract.py    # no network
```

Integration tests run against a **real** Weaviate and a **real** Ollama —
nothing is mocked, and they skip rather than pass when either is missing:

```bash
export WEAVIATE_IT=1
export WEAVIATE_MEMORY_HTTP_HOST=127.0.0.1
export WEAVIATE_MEMORY_HTTP_PORT=8080
export WEAVIATE_MEMORY_API_KEY=...
pytest tests/test_integration.py -v -s
```

Each run uses a unique collection prefix and deletes it in teardown, so it
leaves the cluster as it found it.

`test_contract.py` asserts the provider satisfies Hermes' **real**
`MemoryProvider` ABC, and skips when only the bundled stub is available — a
stub can only prove the provider matches the stub.

## License

MIT
