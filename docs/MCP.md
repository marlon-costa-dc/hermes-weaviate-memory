# MCP alongside this plugin

Weaviate ships a Model Context Protocol server **inside the database binary**
(preview in 1.37.1, GA in 1.38). It listens on the same port as the REST API at
`/v1/mcp`, authenticates with the same API keys, and honours RBAC.

> The standalone `weaviate/mcp-server-weaviate` repository is **deprecated**.
> There is nothing separate to install or run.

## Why use both

They answer different questions, and the overlap is the point: the same
memories, two access paths.

| | This plugin | MCP server |
|---|---|---|
| Retrieval | Automatic, every turn | Only when the model calls a tool |
| Writes | Automatic (turns, mirrored memory) | Explicit, if write access is on |
| Scope | The agent's own memory domains | Any collection the key can read |
| Clients | Hermes | Any MCP client — Claude Code, Cursor, VS Code |

The plugin is what makes an agent *remember without being asked*. MCP is what
lets a human or another tool go look things up. Use the plugin for continuous
memory and MCP for exploration and cross-system reads.

## Enable it on the server

Weaviate ≥ 1.38, Helm chart ≥ 17.8:

```yaml
mcp:
  enabled: true
  writeAccessEnabled: false   # keep writes with the plugin's scoped identity
```

Or directly: `MCP_SERVER_ENABLED=true`.

Leaving `writeAccessEnabled` off is the safer default. The plugin already
writes through an identity scoped to one tenant; an MCP client with write
access and a broader key can reach further than you intend.

## Register it with Hermes

Use a **separate, read-only key** — not the agent's memory key:

```bash
hermes mcp add weaviate --url https://weaviate.internal.example.com/v1/mcp
```

Resulting `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  weaviate:
    url: https://weaviate.internal.example.com/v1/mcp
    enabled: true
    headers:
      Authorization: Bearer ${WEAVIATE_MCP_API_KEY}
```

With `WEAVIATE_MCP_API_KEY` in `~/.hermes/.env`, never inline.

Give that key the `kb_reader` role from [RBAC.md](RBAC.md). Then even a
confused agent cannot write through MCP, and it cannot read another operator's
tenant.

## Other MCP clients

Streamable HTTP, so the config is nearly identical everywhere.

Claude Code / Cursor (`.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "weaviate": {
      "type": "streamable-http",
      "url": "https://weaviate.internal.example.com/v1/mcp",
      "headers": { "Authorization": "Bearer YOUR_READONLY_KEY" }
    }
  }
}
```

VS Code (`.vscode/mcp.json`) uses the same object under `"servers"`.

## Describing this plugin's collections to MCP clients

The MCP server lets you override tool descriptions, which is how you teach a
client what your domains mean:

```yaml
mcp:
  enabled: true
  config:
    tools:
      weaviate-collections-get-config:
        description: >-
          Agent memory collections. MemEpisodic = what happened,
          MemSemantic = durable facts, MemProcedural = how to do things.
      weaviate-tenants-list:
        description: >-
          Tenants: s_shared is organisation-wide, u_<login> is one
          operator's private memory, t_<team> belongs to a team.
```

Without this, an MCP client sees `MemSemantic` and has to guess.

## If MCP is not available

Below Weaviate 1.37.1 there is no MCP server, and this plugin does not need
one — it talks to the REST and gRPC APIs directly. MCP is additive.

> The docs are inconsistent about the version: the MCP server page is badged
> "Added in v1.38" while the env-var reference says the `MCP_SERVER_*`
> variables arrived in v1.37.1 and became runtime-configurable in v1.38. The
> variables are absent from the 1.30.0 source and present in 1.39.0.

## The vectorizer requirement — read this before enabling MCP

The MCP search tools take a **query string**, not a vector. On a collection
created with `vectorizer: none` (this plugin's default, and what
`self_provided()` produces) Weaviate has no way to embed that string.

Measured through the MCP protocol itself against Weaviate 1.39.0 —
`tools/call` on `weaviate-query-hybrid`:

```
collection with vectorizer: none
  → tool execution failed: failed to execute hybrid search:
    VectorFromInput was called without vectorizer on class McpByo

collection with text2vec-ollama
  → {"results":[{"_additional":{"score":0.0125,...},"content":"The oper...
```

BM25 on the same BYO collection works. So with self-provided vectors, **MCP is
limited to keyword search** — the semantic half is unavailable, which is
usually the entire reason for reaching for a vector database. Full log:
[VERIFICATION.md](VERIFICATION.md).

### Tools actually exposed

With `MCP_SERVER_WRITE_ACCESS_ENABLED=false`, three read-only tools (listed
via `tools/list` on 1.39.0):

| Tool | Purpose |
|---|---|
| `weaviate-collections-get-config` | Retrieves collection configuration(s) |
| `weaviate-query-hybrid` | Hybrid search (vector + keyword) in a collection |
| `weaviate-tenants-list` | Lists a collection's tenants |

### Making MCP semantic: a server-side vectorizer

Give the server its own embedder and it can vectorize incoming queries. With
Ollama reachable from the cluster:

```yaml
# Weaviate values.yaml
env:
  ENABLE_MODULES: "backup-s3,text2vec-ollama"
```

```python
from weaviate.classes.config import Configure

client.collections.create(
    name="MemSemantic",
    vector_config=Configure.Vectors.text2vec_ollama(
        api_endpoint="http://ollama.ollama.svc.cluster.local:11434",
        model="nomic-embed-text",
    ),
    ...
)
```

Then every consumer shares one vector space:

| Consumer | How it vectorizes | Semantic search |
|---|---|---|
| MCP clients | server-side | yes |
| Other systems (Dify, pipelines) | server-side | yes |
| This plugin | sends its own vector | yes |

Point the plugin's `embed_model` at the **same model** the server uses.
Different models produce incompatible vector spaces, and mixing them silently
degrades recall rather than raising an error.

`nomic-embed-text` is 274 MB and runs on CPU — no GPU node required.
