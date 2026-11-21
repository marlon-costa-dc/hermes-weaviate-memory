"""Probe: does a server-side vectorizer make string-only search work?

This is the claim the plan rests on. Measured, not assumed:
  - vectorizer: none  -> hybrid(query=str) fails, bm25 works
  - text2vec-ollama   -> hybrid(query=str) works, no client vector needed
"""

from __future__ import annotations

import contextlib
import json
import time
import urllib.request

import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.init import Auth

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


client = weaviate.connect_to_custom(
    http_host="127.0.0.1",
    http_port=18090,
    http_secure=False,
    grpc_host="127.0.0.1",
    grpc_port=50051,
    grpc_secure=False,
    auth_credentials=Auth.api_key("root-key"),
)

PROPS = [Property(name="content", data_type=DataType.TEXT)]

try:
    # ---- A: self-provided vectors (what the plugin does today) ----
    for name in ("ProbeByo", "ProbeOllama"):
        if client.collections.exists(name):
            client.collections.delete(name)

    client.collections.create(
        name="ProbeByo", properties=PROPS, vector_config=Configure.Vectors.self_provided()
    )
    byo = client.collections.get("ProbeByo")
    byo.data.insert(properties={"content": FACT}, vector=embed(FACT))
    time.sleep(1.5)

    print("=" * 68)
    print("A. vectorizer: none  (self-provided vectors)")
    print("=" * 68)
    print("  hybrid(query=str)  — what the MCP server does:")
    try:
        result = byo.query.hybrid(query=QUERY, limit=3)
        print(f"    worked, {len(result.objects)} hit(s)")
    except Exception as exc:
        print(f"    FAILED: {str(exc)[:120]}")

    print("  bm25(query=str):")
    try:
        result = byo.query.bm25(query="CachyOS", limit=3)
        print(f"    worked, {len(result.objects)} hit(s)")
    except Exception as exc:
        print(f"    FAILED: {str(exc)[:90]}")

    print("  near_vector(client-supplied) — what the plugin does:")
    try:
        result = byo.query.near_vector(near_vector=embed(QUERY), limit=3)
        print(f"    worked, {len(result.objects)} hit(s)")
    except Exception as exc:
        print(f"    FAILED: {str(exc)[:90]}")

    # ---- B: server-side vectorizer ----
    print()
    print("=" * 68)
    print("B. text2vec-ollama  (server vectorizes)")
    print("=" * 68)
    client.collections.create(
        name="ProbeOllama",
        properties=PROPS,
        vector_config=Configure.Vectors.text2vec_ollama(
            api_endpoint="http://127.0.0.1:11434",
            model="nomic-embed-text",
        ),
    )
    ollama = client.collections.get("ProbeOllama")
    # NOTE: no vector passed — the server must compute it.
    ollama.data.insert(properties={"content": FACT})
    time.sleep(2.0)

    print("  insert without a client vector:")
    count = ollama.aggregate.over_all(total_count=True).total_count
    print(f"    stored, {count} object(s)")

    print("  hybrid(query=str)  — what the MCP server does:")
    try:
        result = ollama.query.hybrid(query=QUERY, limit=3)
        print(f"    worked, {len(result.objects)} hit(s)")
        for obj in result.objects:
            print(f"      -> {obj.properties['content'][:58]}")
    except Exception as exc:
        print(f"    FAILED: {str(exc)[:120]}")

    print("  near_text(query=str):")
    try:
        result = ollama.query.near_text(query=QUERY, limit=3)
        print(f"    worked, {len(result.objects)} hit(s)")
    except Exception as exc:
        print(f"    FAILED: {str(exc)[:90]}")
finally:
    for name in ("ProbeByo", "ProbeOllama"):
        with contextlib.suppress(Exception):
            client.collections.delete(name)
    client.close()
