#!/usr/bin/env bash
# Install weaviate-memory into a Hermes installation.
#
# Copies the package into $HERMES_HOME/plugins/ and installs its one
# dependency into the Hermes virtualenv — the plugin is imported by the agent
# process, so the dependency must live where that process can see it.
set -euo pipefail

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
PLUGIN_DIR="$HERMES_HOME/plugins/weaviate-memory"
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -d "$HERMES_HOME" ]]; then
  echo "error: no Hermes installation at $HERMES_HOME" >&2
  echo "set HERMES_HOME if it lives elsewhere." >&2
  exit 1
fi

echo "Installing weaviate-memory into $PLUGIN_DIR"
mkdir -p "$PLUGIN_DIR"
cp "$SOURCE_DIR"/weaviate_memory/*.py "$PLUGIN_DIR/"
cp "$SOURCE_DIR/plugin.yaml" "$PLUGIN_DIR/"

# The Hermes venv, if this is a standard install.
VENV_PY="$HERMES_HOME/hermes-agent/venv/bin/python"
if [[ -x "$VENV_PY" ]]; then
  echo "Installing weaviate-client into the Hermes venv"
  "$VENV_PY" -m pip install --quiet 'weaviate-client>=4.22,<5'
else
  echo "note: Hermes venv not found at $VENV_PY" >&2
  echo "      install the dependency yourself: pip install 'weaviate-client>=4.22,<5'" >&2
fi

cat <<'EOF'

Installed. Next:

  1. Put your API key in $HERMES_HOME/.env:
       WEAVIATE_MEMORY_API_KEY=your-key

  2. Point the plugin at your cluster in $HERMES_HOME/config.yaml:
       plugins:
         weaviate-memory:
           http_host: weaviate.example.com
           http_port: 443
           http_secure: true
           grpc_host: weaviate-grpc.example.com
           grpc_port: 443
           grpc_secure: true

  3. Activate it:
       hermes memory setup     # choose weaviate-memory
       hermes memory status

An embedding model is required:
       ollama pull nomic-embed-text
EOF
