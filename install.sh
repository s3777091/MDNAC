#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRIPT_DIR/.uv-cache}"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if ! command -v curl >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    APT_GET=(apt-get)
    if [ "$(id -u)" -ne 0 ]; then
      if ! command -v sudo >/dev/null 2>&1; then
        echo "error: curl is missing and sudo is not available for apt-get." >&2
        exit 1
      fi
      APT_GET=(sudo apt-get)
    fi
    "${APT_GET[@]}" update
    "${APT_GET[@]}" install -y curl ca-certificates
  else
    echo "error: curl is required to install uv automatically." >&2
    echo "Install curl, then rerun this script." >&2
    exit 1
  fi
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv was not found after installation." >&2
  exit 1
fi

uv python install 3.11
uv sync --frozen --python 3.11 "$@"

echo
echo "Environment is ready."
echo "Run the RefSeq compiler with:"
echo "bash cmd/build_refseq_profile_text.sh data/raw/refseq_bacteria_protein -o data/compiled/refseq_bacteria_protein/fungi/package_1 --vocab-size 256 --instruction-min-proteins 5 --workers 8 --skip tokenizer_map.json"
