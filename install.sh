#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_VARIANT="${TORCH_VARIANT:-cu126}"
RECREATE=0
SKIP_VERIFY=0
SKIP_KERNEL=0

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRIPT_DIR/.uv-cache}"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# --- Parse arguments ---
EXTRA_SYNC_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --torch)
      TORCH_VARIANT="$2"
      shift 2
      ;;
    --python)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --recreate)
      RECREATE=1
      shift
      ;;
    --skip-verify)
      SKIP_VERIFY=1
      shift
      ;;
    --skip-kernel)
      SKIP_KERNEL=1
      shift
      ;;
    *)
      EXTRA_SYNC_ARGS+=("$1")
      shift
      ;;
  esac
done

# Validate torch variant
case "$TORCH_VARIANT" in
  cu126|cu128|cpu|auto|none) ;;
  *) die "Invalid --torch value: $TORCH_VARIANT. Must be one of: cu126, cu128, cpu, auto, none" ;;
esac

log() {
  printf '\n==> %s\n' "$*"
}

warn() {
  printf 'WARNING: %s\n' "$*" >&2
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
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
  uv sync --frozen --python "$PYTHON_VERSION" "${EXTRA_SYNC_ARGS[@]+"${EXTRA_SYNC_ARGS[@]}"}"
}

install_torch() {
  # Resolve venv python explicitly
  local venv_python="$SCRIPT_DIR/.venv/bin/python"
  if [ ! -x "$venv_python" ]; then
    die ".venv/bin/python not found after uv sync."
  fi
  echo "venv python: $venv_python"

  # Ensure pyvenv.cfg exists (uv 0.11+ may not create it for managed envs)
  local pyvenv_cfg="$SCRIPT_DIR/.venv/pyvenv.cfg"
  if [ ! -f "$pyvenv_cfg" ]; then
    log "Creating pyvenv.cfg (missing after uv sync)"
    local py_home
    py_home=$("$venv_python" -c "import sys, os; print(os.path.dirname(getattr(sys, '_base_executable', sys.executable)))")
    local py_ver
    py_ver=$("$venv_python" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
    cat > "$pyvenv_cfg" <<PYCFG
home = $py_home
include-system-site-packages = false
version = $py_ver
PYCFG
  fi

  # Point uv pip at our project venv
  export VIRTUAL_ENV="$SCRIPT_DIR/.venv"

  if [ "$TORCH_VARIANT" = "none" ]; then
    log "Skipping PyTorch (--torch none)"
    return
  fi

  if [ "$TORCH_VARIANT" = "auto" ]; then
    log "Keeping PyTorch from uv.lock (--torch auto)"
    return
  fi

  local index_url=""
  case "$TORCH_VARIANT" in
    cpu)   index_url="https://download.pytorch.org/whl/cpu" ;;
    cu126) index_url="https://download.pytorch.org/whl/cu126" ;;
    cu128) index_url="https://download.pytorch.org/whl/cu128" ;;
  esac

  # Use 'uv pip' instead of 'python -m pip' because uv-managed Pythons
  # set PEP 668 EXTERNALLY-MANAGED markers that block direct pip usage.
  log "Installing PyTorch $TORCH_VARIANT from $index_url"
  uv pip install --reinstall --index-url "$index_url" torch torchvision torchaudio
}

verify_install() {
  local venv_python="$SCRIPT_DIR/.venv/bin/python"

  if [ "$SKIP_VERIFY" -eq 1 ]; then
    return
  fi

  log "Verifying Python"
  "$venv_python" -c "import sys; print(f'Python {sys.version}')"

  if [ "$TORCH_VARIANT" = "cu126" ] || [ "$TORCH_VARIANT" = "cu128" ]; then
    log "Checking NVIDIA driver"
    if have nvidia-smi; then
      nvidia-smi
    else
      warn "nvidia-smi not found. CUDA verification may fail."
    fi
  fi

  if [ "$TORCH_VARIANT" != "none" ]; then
    log "Verifying PyTorch"
    "$venv_python" -c "
import torch
print(f'torch {torch.__version__}')
print(f'CUDA compiled: {torch.version.cuda}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
"
    if [ "$TORCH_VARIANT" = "cu126" ] || [ "$TORCH_VARIANT" = "cu128" ]; then
      log "Verifying CUDA tensor allocation"
      "$venv_python" -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available but variant is $TORCH_VARIANT'
x = torch.randn(256, 256, device='cuda')
print(f'CUDA tensor OK on {x.device}')
" || warn "GPU tensor test failed. Check nvidia-smi and driver version."
    fi
  fi

  log "Verifying project import"
  "$venv_python" -c "from libs.data.config import DataConfig; print('OK: libs.data.config importable')"
}

install_jupyter_kernel() {
  if [ "$SKIP_KERNEL" -eq 1 ]; then
    return
  fi

  local venv_python="$SCRIPT_DIR/.venv/bin/python"
  log "Installing ipykernel and registering Jupyter kernel"
  uv pip install ipykernel 2>/dev/null || true
  "$venv_python" -m ipykernel install --user --name "microbial-dna-compiler" --display-name "Microbial DNA Compiler (uv)"
}

persist_path() {
  if ! grep -q '.local/bin' ~/.bashrc 2>/dev/null; then
    log "Persisting PATH in ~/.bashrc"
    echo 'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
  fi
}

print_done() {
  cat <<EOF


==================================================================
  DONE. Torch=$TORCH_VARIANT  Python=$PYTHON_VERSION
==================================================================

GPU test:
  uv run python -c "import torch; print(torch.cuda.is_available())"

Run tests:
  uv run python -m pytest tests/

EOF
}

main() {
  ensure_project_root

  if [ "$RECREATE" -eq 1 ]; then
    log "Removing .venv"
    rm -rf .venv
  fi

  ensure_uv
  ensure_python
  sync_environment
  install_torch
  prepare_local_files
  verify_install
  install_jupyter_kernel
  persist_path
  print_done
}

main
