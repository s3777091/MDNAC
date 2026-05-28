#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRIPT_DIR/.uv-cache}"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

log() {
  printf '\n==> %s\n' "$*"
}

warn() {
  printf 'warning: %s\n' "$*" >&2
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

have() {
  command -v "$1" >/dev/null 2>&1
}

ensure_project_root() {
  [ -f "pyproject.toml" ] || die "pyproject.toml not found. Run this script from the repository root."
  [ -f "uv.lock" ] || die "uv.lock not found. Commit or generate uv.lock before installing."
}

ensure_download_tool() {
  if have curl || have wget; then
    return
  fi

  if ! have apt-get; then
    die "curl or wget is required to install uv automatically."
  fi

  log "Installing curl and CA certificates"
  local -a apt_get=(apt-get)
  if [ "$(id -u)" -ne 0 ]; then
    have sudo || die "curl is missing and sudo is not available for apt-get."
    apt_get=(sudo apt-get)
  fi

  "${apt_get[@]}" update
  "${apt_get[@]}" install -y curl ca-certificates
}

ensure_uv() {
  if have uv; then
    return
  fi

  ensure_download_tool
  log "Installing uv"
  if have curl; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  else
    wget -qO- https://astral.sh/uv/install.sh | sh
  fi

  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  have uv || die "uv was not found after installation. Add ~/.local/bin to PATH and rerun."
}

ensure_python() {
  if uv python find "$PYTHON_VERSION" >/dev/null 2>&1; then
    return
  fi

  log "Installing Python $PYTHON_VERSION through uv"
  uv python install "$PYTHON_VERSION"
}

prepare_local_files() {
  log "Preparing local runtime directories"
  mkdir -p \
    data/raw/refseq_bacteria_protein \
    data/compiled/refseq_bacteria_protein \
    data/compiled/refseq_bacteria_profile_pretrain \
    data/cache/protein_train_parts \
    data/checkpoints/protein_from_scratch \
    libs/data/models/catalog \
    libs/data/models/datasets \
    libs/data/models/trash \
    libs/data/models/sessions

  if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    warn ".env was not created automatically. Copy .env.example only when you need MinIO or NCBI credentials."
  fi
}

sync_environment() {
  log "Syncing Python environment from uv.lock"
  uv sync --frozen --python "$PYTHON_VERSION" "$@"
}

verify_install() {
  log "Verifying installed package"
  uv run --no-sync python -c "from libs.data.config import DataConfig; print('ok: microbial-dna-compiler environment is importable')"
}

print_next_steps() {
  cat <<'EOF'

Environment is ready.

Common commands:
  uv run python -m unittest discover -s tests -p "test_*.py"
  bash cmd/build_refseq_profile_text.sh data/raw/refseq_bacteria_protein -o data/compiled/refseq_bacteria_protein --vocab-size 512 --instruction-min-proteins 10 --workers 0
  bash cmd/build_profile_pretrain_from_instruction_jsonl.sh data/compiled/refseq_bacteria_protein/instruction.jsonl -o data/compiled/refseq_bacteria_profile_pretrain

Notes:
  - Put raw RefSeq archives under data/raw/refseq_bacteria_protein before compiling.
  - Keep real MinIO and NCBI credentials in .env or environment variables.
  - Pass extra uv sync options to this script if needed.
EOF
}

main() {
  ensure_project_root
  ensure_uv
  ensure_python
  sync_environment "$@"
  prepare_local_files
  verify_install
  print_next_steps
}

main "$@"
