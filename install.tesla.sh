#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_VARIANT="${TORCH_VARIANT:-auto}"
TORCH_MIN_VERSION="${TORCH_MIN_VERSION:-2.11}"
RECREATE=0
SKIP_VERIFY=0
SKIP_KERNEL=0
TORCH_SYNC_PROTECTED=0

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
  cu126|cu128|cu130|cpu|auto|none) ;;
  *) die "Invalid --torch value: $TORCH_VARIANT. Must be one of: cu126, cu128, cu130, cpu, auto, none" ;;
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
  local -a sync_args=(--frozen --python "$PYTHON_VERSION")
  local venv_python="$SCRIPT_DIR/.venv/bin/python"

  if [ "$TORCH_VARIANT" = "none" ]; then
    TORCH_SYNC_PROTECTED=1
    log "PyTorch sync disabled (--torch none)"
  elif [ "$RECREATE" -eq 0 ] && [ -x "$venv_python" ]; then
    if [ "$TORCH_VARIANT" = "auto" ] && torch_auto_install_usable "$venv_python"; then
      TORCH_SYNC_PROTECTED=1
      TORCH_VARIANT=$(torch_current_variant "$venv_python")
      log "Existing PyTorch is usable ($TORCH_VARIANT); uv sync will leave it untouched"
    elif [ "$TORCH_VARIANT" != "auto" ] && torch_install_matches "$venv_python" "$TORCH_VARIANT"; then
      TORCH_SYNC_PROTECTED=1
      log "Existing PyTorch matches $TORCH_VARIANT; uv sync will leave it untouched"
    else
      log "PyTorch will be handled after uv sync; uv sync will skip PyTorch packages"
    fi
  else
    log "PyTorch will be handled after uv sync; uv sync will skip PyTorch packages"
  fi

  sync_args+=(--inexact --no-install-package torch --no-install-package torchvision --no-install-package torchaudio)

  log "Syncing Python environment from uv.lock"
  uv sync "${sync_args[@]}" "${EXTRA_SYNC_ARGS[@]+"${EXTRA_SYNC_ARGS[@]}"}"
}

detect_cuda_variant() {
  if ! have nvidia-smi; then
    echo "cpu"
    return
  fi

  if is_tesla_v100_gpu; then
    warn "Tesla V100 / compute capability 7.0 detected. Selecting cu126."
    echo "cu126"
    return
  fi

  local output
  output=$(nvidia-smi 2>&1) || { echo "cpu"; return; }
  local cuda_ver
  cuda_ver=$(echo "$output" | grep -oP 'CUDA Version:\s+\K[0-9]+\.[0-9]+' || true)
  if [ -z "$cuda_ver" ]; then
    echo "cpu"
    return
  fi
  local major minor
  major=$(echo "$cuda_ver" | cut -d. -f1)
  minor=$(echo "$cuda_ver" | cut -d. -f2)
  echo "Detected CUDA driver version: $cuda_ver" >&2
  if [ "$major" -ge 13 ]; then
    warn "CUDA driver $cuda_ver detected. Tesla-safe auto mode selects cu126 instead of blindly selecting cu130."
    echo "cu126"
  elif [ "$major" -eq 12 ] && [ "$minor" -ge 8 ]; then
    echo "cu128"
  elif [ "$major" -eq 12 ] && [ "$minor" -ge 6 ]; then
    echo "cu126"
  else
    warn "CUDA driver $cuda_ver is older than 12.6. Selecting cpu variant."
    echo "cpu"
  fi
}

has_nvidia_driver() {
  have nvidia-smi && nvidia-smi >/dev/null 2>&1
}

is_tesla_v100_gpu() {
  if ! have nvidia-smi; then
    return 1
  fi

  local query
  query=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits 2>/dev/null || true)
  if printf '%s\n' "$query" | grep -Eiq '(^|,| )Tesla V100|(^|,| )V100|(^|,| )7\.0($|,| )'; then
    return 0
  fi

  query=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)
  printf '%s\n' "$query" | grep -Eiq 'Tesla V100|V100'
}

ensure_tesla_torch_variant() {
  if ! is_tesla_v100_gpu; then
    return
  fi

  if [ "$TORCH_VARIANT" = "auto" ]; then
    TORCH_VARIANT="cu126"
    log "Tesla V100 detected; forcing PyTorch variant: $TORCH_VARIANT"
    return
  fi

  if [ "$TORCH_VARIANT" != "cu126" ] && [ "$TORCH_VARIANT" != "none" ]; then
    die "Tesla V100 detected. Use --torch cu126 or --torch none, not --torch $TORCH_VARIANT."
  fi
}

torch_current_variant() {
  local venv_python="$1"

  "$venv_python" -c '
try:
    import torch
except Exception:
    raise SystemExit(1)

compiled_cuda = torch.version.cuda
if compiled_cuda is None:
    print("cpu")
else:
    parts = compiled_cuda.split(".")
    major = parts[0]
    minor = parts[1] if len(parts) > 1 else "0"
    print(f"cu{major}{minor}")
'
}

torch_auto_install_usable() {
  local venv_python="$1"
  local has_nvidia=0

  if has_nvidia_driver; then
    has_nvidia=1
  fi

  "$venv_python" -c '
import re
import sys

has_nvidia = sys.argv[1] == "1"
min_version = sys.argv[2]

try:
    import torch
except Exception as exc:
    print(f"PyTorch import failed: {exc}")
    raise SystemExit(1)


def version_tuple(value):
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", value)
    if not match:
        return ()
    return tuple(int(part or 0) for part in match.groups())


torch_version = torch.__version__.split("+", 1)[0]
compiled_cuda = torch.version.cuda
cuda_available = torch.cuda.is_available()
print(f"Found torch {torch.__version__} (CUDA compiled: {compiled_cuda}, CUDA available: {cuda_available})")

if version_tuple(torch_version) < version_tuple(min_version):
    print(f"PyTorch {torch_version} is older than required {min_version}.")
    raise SystemExit(1)

if has_nvidia and (compiled_cuda is None or not cuda_available):
    print("NVIDIA driver found, but installed PyTorch is not CUDA-usable.")
    raise SystemExit(1)

if has_nvidia:
    try:
        x = torch.randint(0, 256, (2, 512), device="cuda")
        torch.cuda.synchronize()
        del x
    except Exception as exc:
        print(f"CUDA tensor allocation failed: {exc}")
        raise SystemExit(1)

raise SystemExit(0)
' "$has_nvidia" "$TORCH_MIN_VERSION"
}

torch_install_matches() {
  local venv_python="$1"
  local requested_variant="$2"

  "$venv_python" -c '
import re
import sys

requested_variant = sys.argv[1]
min_version = sys.argv[2]

try:
    import torch
except Exception as exc:
    print(f"PyTorch import failed: {exc}")
    raise SystemExit(1)


def version_tuple(value):
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", value)
    if not match:
        return ()
    return tuple(int(part or 0) for part in match.groups())


torch_version = torch.__version__.split("+", 1)[0]
compiled_cuda = torch.version.cuda
cuda_available = torch.cuda.is_available()
print(f"Found torch {torch.__version__} (CUDA compiled: {compiled_cuda}, CUDA available: {cuda_available})")

if version_tuple(torch_version) < version_tuple(min_version):
    print(f"PyTorch {torch_version} is older than required {min_version}.")
    raise SystemExit(1)

if requested_variant == "cpu":
    if compiled_cuda is None:
        raise SystemExit(0)
    print("Installed PyTorch is a CUDA build, but CPU variant was requested.")
    raise SystemExit(1)

expected_cuda = {"cu126": "12.6", "cu128": "12.8", "cu130": "13.0"}[requested_variant]
if compiled_cuda and compiled_cuda.startswith(expected_cuda):
    try:
        x = torch.randint(0, 256, (2, 512), device="cuda")
        torch.cuda.synchronize()
        del x
    except Exception as exc:
        print(f"CUDA tensor allocation failed: {exc}")
        raise SystemExit(1)
    raise SystemExit(0)

print(f"Installed PyTorch CUDA build does not match requested {requested_variant}.")
raise SystemExit(1)
' "$requested_variant" "$TORCH_MIN_VERSION"
}

install_torch() {
  # Resolve venv python explicitly
  local venv_python="$SCRIPT_DIR/.venv/bin/python"
  if [ ! -x "$venv_python" ]; then
    die ".venv/bin/python not found after uv sync."
  fi
  echo "venv python: $venv_python"

  if [ "$TORCH_SYNC_PROTECTED" -eq 1 ]; then
    log "Keeping existing PyTorch; skipping install"
    return
  fi

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

  # Resolve "auto" to concrete variant
  if [ "$TORCH_VARIANT" = "auto" ]; then
    log "Checking existing PyTorch before selecting an install variant"
    if torch_auto_install_usable "$venv_python"; then
      TORCH_VARIANT=$(torch_current_variant "$venv_python")
      log "Keeping existing PyTorch variant: $TORCH_VARIANT"
      return
    fi

    log "Auto-detecting PyTorch variant from nvidia-smi"
    TORCH_VARIANT=$(detect_cuda_variant)
    log "Selected PyTorch variant: $TORCH_VARIANT"
  fi

  if [ "$TORCH_VARIANT" = "none" ]; then
    log "Skipping PyTorch (--torch none)"
    return
  fi

  if torch_install_matches "$venv_python" "$TORCH_VARIANT"; then
    log "PyTorch already satisfies >=$TORCH_MIN_VERSION and variant $TORCH_VARIANT; skipping install"
    return
  fi

  local index_url=""
  case "$TORCH_VARIANT" in
    cpu)   index_url="https://download.pytorch.org/whl/cpu" ;;
    cu126) index_url="https://download.pytorch.org/whl/cu126" ;;
    cu128) index_url="https://download.pytorch.org/whl/cu128" ;;
    cu130) index_url="https://download.pytorch.org/whl/cu130" ;;
  esac

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

  case "$TORCH_VARIANT" in
    cu*)
      log "Checking NVIDIA driver"
      if have nvidia-smi; then
        nvidia-smi
      else
        die "nvidia-smi not found but CUDA PyTorch variant $TORCH_VARIANT was selected."
      fi
      ;;
  esac

  if [ "$TORCH_VARIANT" != "none" ]; then
    log "Verifying PyTorch"
    "$venv_python" -c "
import torch
print(f'torch {torch.__version__}')
print(f'CUDA compiled: {torch.version.cuda}')
print(f'CUDA available: {torch.cuda.is_available()}')
print('Muon available:', hasattr(torch.optim, 'Muon'))
if torch.cuda.is_available():
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'GPU capability: {torch.cuda.get_device_capability(0)}')
"
    case "$TORCH_VARIANT" in
      cu*)
        log "Verifying CUDA tensor allocation"
        "$venv_python" -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available but variant is $TORCH_VARIANT'
x = torch.randint(0, 256, (2, 512), device='cuda')
torch.cuda.synchronize()
print(f'CUDA tensor OK: {tuple(x.shape)} {x.device}')
"
        ;;
    esac
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
  uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); x=torch.randint(0,256,(2,512),device='cuda'); print('CUDA OK:', x.shape, x.device); print('Muon available:', hasattr(torch.optim, 'Muon'))"

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
  ensure_tesla_torch_variant
  sync_environment
  install_torch
  prepare_local_files
  verify_install
  install_jupyter_kernel
  persist_path
  print_done
}

main
