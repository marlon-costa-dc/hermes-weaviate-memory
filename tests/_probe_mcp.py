"""Probe: does the Weaviate MCP server's hybrid tool work on BYO-vector collections?

Speaks raw MCP (Streamable HTTP + JSON-RPC) against the local lab container.
This is the decisive test for the plan: the MCP tool takes a query STRING, so
a collection with no server-side vectorizer should fail, and one with
text2vec-ollama should succeed.
"""

from __future__ import annotations

import contextlib
import json
import re
import time
import urllib.request

import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.init import Auth

BASE = "http://127.0.0.1:18090"
KEY = "root-key"
FACT = "The operator runs CachyOS with systemd-boot on the workstation tungs"
QUERY = "which linux distribution does the user have"


def embed(text: str) -> list[float]:
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/embed",
        data=json.dumps({"model": "nomic-embed-text", "input": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)["embeddings"][0]


class MCP:
    """Minimal MCP client: initialize, then call tools."""

    def __init__(self, base: str, key: str) -> None:
        self.url = f"{base}/v1/mcp"
        self.key = key
        self.session = ""

    def _post(self, payload: dict) -> tuple[dict | None, dict]:
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.session:
            headers["Mcp-Session-Id"] = self.session
        request = urllib.request.Request(
            self.url, data=json.dumps(payload).encode(), headers=headers
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode()
            received = dict(response.headers)
        match = re.search(r"\{.*\}", body, re.S)
        return (json.loads(match.group(0)) if match else None), received

    def connect(self) -> None:
        _, headers = self._post(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "probe", "version": "1"},
                },
            }
        )
        for name, value in headers.items():
            if name.lower() == "mcp-session-id":
                self.session = value
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def call(self, tool: str, arguments: dict) -> dict:
        response, _ = self._post(
            {
                "jsonrpc": "2.0",
                "id": 99,
                "method": "tools/call",
                "params": {"name": tool, "arguments": arguments},
            }
        )
        return response or {}


client = weaviate.connect_to_custom(
    http_host="127.0.0.1",
    http_port=18090,
    http_secure=False,
    grpc_host="127.0.0.1",
    grpc_port=50051,
    grpc_secure=False,
    auth_credentials=Auth.api_key(KEY),
)

PROPS = [Property(name="content", data_type=DataType.TEXT)]

try:
    for name in ("McpByo", "McpOllama"):
        if client.collections.exists(name):
            client.collections.delete(name)

    client.collections.create(
        name="McpByo", properties=PROPS, vector_config=Configure.Vectors.self_provided()
    )
    client.collections.get("McpByo").data.insert(properties={"content": FACT}, vector=embed(FACT))

    client.collections.create(
        name="McpOllama",
        properties=PROPS,
        vector_config=Configure.Vectors.text2vec_ollama(
            api_endpoint="http://127.0.0.1:11434", model="nomic-embed-text"
        ),
    )
    client.collections.get("McpOllama").data.insert(properties={"content": FACT})
    time.sleep(2.5)

    mcp = MCP(BASE, KEY)
    mcp.connect()
    print(f"MCP session established: {bool(mcp.session)}\n")

    for collection in ("McpByo", "McpOllama"):
        label = "vectorizer: none" if collection == "McpByo" else "text2vec-ollama"
        print("=" * 68)
        print(f"weaviate-query-hybrid on {collection}  ({label})")
        print("=" * 68)
        result = mcp.call(
            "weaviate-query-hybrid",
            {"collection_name": collection, "query": QUERY, "limit": 3},
        )
        payload = result.get("result", {})
        is_error = payload.get("isError")
        text = ""
        for block in payload.get("content", []):
            text += block.get("text", "")
        if result.get("error"):
            print(f"  JSON-RPC error: {str(result['error'])[:150]}")
        elif is_error:
            print(f"  TOOL ERROR: {text[:200]}")
        else:
            print(f"  OK: {text[:220]}")
        print()
finally:
    for name in ("McpByo", "McpOllama"):
        with contextlib.suppress(Exception):
            client.collections.delete(name)
    client.close()
