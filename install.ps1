#Requires -Version 5.1
<#
.SYNOPSIS
    Windows installer for Microbial DNA Compiler (with PyTorch).
.DESCRIPTION
    Sets up Python 3.11 via uv, syncs the project environment, installs PyTorch
    with the selected variant (CUDA 12.6, CUDA 12.8, CPU, or none), creates local
    data directories, and verifies the installation.
.PARAMETER Python
    Python version to use. Default: 3.11
.PARAMETER Torch
    PyTorch variant: cu126, cu128, cpu, auto, none. Default: cu126
.PARAMETER Recreate
    Remove .venv before install.
.PARAMETER SkipVerify
    Skip verification steps.
.PARAMETER SkipKernel
    Skip Jupyter kernel registration.
.EXAMPLE
    .\install.ps1
    .\install.ps1 -Torch cu126
    .\install.ps1 -Torch cpu -Recreate
    .\install.ps1 -Torch none -SkipKernel
#>

param(
    [string]$Python = $(if ($env:PYTHON_VERSION) { $env:PYTHON_VERSION } else { "3.11" }),
    [ValidateSet("cu126", "cu128", "cpu", "auto", "none")]
    [string]$Torch = $(if ($env:TORCH_VARIANT) { $env:TORCH_VARIANT } else { "cu126" }),
    [switch]$Recreate,
    [switch]$SkipVerify,
    [switch]$SkipKernel
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Definition

Push-Location $SCRIPT_DIR
try {

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function Log($msg) {
    Write-Host "`n==> $msg" -ForegroundColor Cyan
}

function Warn($msg) {
    Write-Warning $msg
}

function Die($msg) {
    Write-Host "ERROR: $msg" -ForegroundColor Red
    exit 1
}

function Have($cmd) {
    return [bool](Get-Command $cmd -ErrorAction SilentlyContinue)
}

# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

function Ensure-ProjectRoot {
    if (-not (Test-Path "pyproject.toml")) {
        Die "pyproject.toml not found. Run this script from the repository root."
    }
    if (-not (Test-Path "uv.lock")) {
        Die "uv.lock not found."
    }
}

function Ensure-Uv {
    if (Have "uv") {
        Log "Using uv: $(uv --version)"
        return
    }

    Log "Installing uv via official installer"
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression

    # Refresh PATH for the current session
    $uvPath = Join-Path $env:USERPROFILE ".local\bin"
    $cargoPath = Join-Path $env:USERPROFILE ".cargo\bin"
    foreach ($p in @($uvPath, $cargoPath)) {
        if ((Test-Path $p) -and ($env:PATH -notlike "*$p*")) {
            $env:PATH = "$p;$env:PATH"
        }
    }

    if (-not (Have "uv")) {
        Die "uv was not found after installation. Close and reopen the terminal, or add ~/.local/bin to PATH."
    }
    Log "uv installed: $(uv --version)"
}

function Ensure-Python {
    $found = uv python find $Python 2>$null
    if ($LASTEXITCODE -eq 0 -and $found) {
        Log "Python $Python is available: $found"
        return
    }

    Log "Installing Python $Python through uv"
    uv python install $Python
    if ($LASTEXITCODE -ne 0) { Die "Failed to install Python $Python" }
}

function Remove-VenvIfRequested {
    if ($Recreate) {
        Log "Removing existing .venv"
        if (Test-Path ".venv") {
            Remove-Item -Recurse -Force ".venv"
        }
    }
}

function Sync-Environment {
    Log "Syncing environment from uv.lock"

    & uv sync --frozen --python $Python
    if ($LASTEXITCODE -ne 0) { Die "uv sync failed" }
}

function Ensure-Pip {
    Log "Ensuring pip exists"

    & uv run --no-sync python -m ensurepip --upgrade 2>$null
    & uv run --no-sync python -m pip install --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) { Warn "pip upgrade had issues, continuing..." }
}

function Install-Torch {
    if ($Torch -eq "none") {
        Log "Skipping PyTorch install (--Torch none)"
        return
    }

    if ($Torch -eq "auto") {
        Log "Keeping PyTorch from uv.lock (--Torch auto)"
        return
    }

    if ($Torch -eq "cpu") {
        Log "Installing PyTorch CPU build"
        & uv run --no-sync python -m pip install `
            --reinstall --upgrade `
            --index-url https://download.pytorch.org/whl/cpu `
            torch torchvision torchaudio
        if ($LASTEXITCODE -ne 0) { Die "PyTorch CPU install failed" }
        return
    }

    if ($Torch -eq "cu126") {
        Log "Installing PyTorch CUDA 12.6 build"
        & uv run --no-sync python -m pip install `
            --reinstall --upgrade `
            --index-url https://download.pytorch.org/whl/cu126 `
            torch torchvision torchaudio
        if ($LASTEXITCODE -ne 0) { Die "PyTorch CUDA 12.6 install failed" }
        return
    }

    if ($Torch -eq "cu128") {
        Log "Installing PyTorch CUDA 12.8 build"
        & uv run --no-sync python -m pip install `
            --reinstall --upgrade `
            --index-url https://download.pytorch.org/whl/cu128 `
            torch torchvision torchaudio
        if ($LASTEXITCODE -ne 0) { Die "PyTorch CUDA 12.8 install failed" }
        return
    }
}

function Prepare-LocalDirs {
    Log "Preparing local runtime directories"

    $dirs = @(
        "data\raw\refseq_bacteria_protein",
        "data\compiled\refseq_bacteria_protein",
        "data\compiled\refseq_bacteria_profile_pretrain",
        "data\cache\protein_train_parts",
        "data\checkpoints\protein_from_scratch",
        "libs\data\models\catalog",
        "libs\data\models\datasets",
        "libs\data\models\trash",
        "libs\data\models\sessions"
    )

    foreach ($d in $dirs) {
        if (-not (Test-Path $d)) {
            New-Item -ItemType Directory -Path $d -Force | Out-Null
        }
    }

    if (-not (Test-Path ".env") -and (Test-Path ".env.example")) {
        Warn ".env was not created automatically. Copy .env.example only if you need MinIO or NCBI credentials."
    }
}

function Verify-Python {
    Log "Verifying Python"
    & uv run --no-sync python -c "import sys; print('python:', sys.version); print('executable:', sys.executable)"
}

function Verify-Driver {
    if ($Torch -ne "cu126" -and $Torch -ne "cu128") {
        return
    }

    Log "Checking NVIDIA driver"
    if (Have "nvidia-smi") {
        & nvidia-smi
    } else {
        Warn "nvidia-smi not found. CUDA may not work until NVIDIA driver is installed."
    }
}

function Verify-Torch {
    if ($Torch -eq "none") {
        return
    }

    Log "Verifying PyTorch"
    & uv run --no-sync python -c @"
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('cuda version:', torch.version.cuda)
print('gpu count:', torch.cuda.device_count())
print('gpu name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')
"@

    if ($Torch -eq "cu126" -or $Torch -eq "cu128") {
        & uv run --no-sync python -c @"
import torch
assert torch.cuda.is_available(), 'CUDA GPU is not available'
x = torch.randn(256, 256, device='cuda')
y = x @ x
torch.cuda.synchronize()
print('gpu test:', y.shape, y.device)
"@
        if ($LASTEXITCODE -ne 0) {
            Warn "GPU test failed. Check your NVIDIA driver and CUDA installation."
        }
    }
}

function Verify-Project {
    Log "Verifying project import"
    & uv run --no-sync python -c "from libs.data.config import DataConfig; print('ok: libs.data.config.DataConfig importable')"
}

function Verify-Install {
    if ($SkipVerify) {
        Warn "Skipping verification"
        return
    }

    Verify-Python
    Verify-Driver
    Verify-Torch
    Verify-Project
}

function Install-JupyterKernel {
    if ($SkipKernel) {
        Warn "Skipping Jupyter kernel"
        return
    }

    Log "Installing Jupyter kernel"
    & uv run --no-sync python -m pip install --upgrade ipykernel
    & uv run --no-sync python -m ipykernel install `
        --user `
        --name "microbial-dna-compiler" `
        --display-name "Microbial DNA Compiler (uv GPU)"
}

function Print-NextSteps {
    Write-Host @"

===================================================================
  Environment is ready.  Torch variant: $Torch
===================================================================

GPU test:
  uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

Jupyter kernel:
  Microbial DNA Compiler (uv GPU)

Common commands:
  uv run python -m pytest tests/
  python cmd\build_refseq_profile_text.py data\raw\refseq_bacteria_protein -o data\compiled\refseq_bacteria_protein --vocab-size 512 --instruction-min-proteins 10 --workers 0

Recommended install:
  .\install.ps1 -Recreate -Torch cu126
"@ -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

Log "Microbial DNA Compiler installer (Windows)"
Log "Python=$Python  Torch=$Torch  Recreate=$Recreate"

Ensure-ProjectRoot
Ensure-Uv
Ensure-Python
Remove-VenvIfRequested
Sync-Environment
Ensure-Pip
Install-Torch
Prepare-LocalDirs
Verify-Install
Install-JupyterKernel
Print-NextSteps

} finally {
    Pop-Location
}
