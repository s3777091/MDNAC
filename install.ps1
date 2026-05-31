#Requires -Version 5.1
param(
    [string]$Python = "3.11",
    [ValidateSet("cu126","cu128","cpu","auto","none")]
    [string]$Torch = "cu126",
    [switch]$Recreate,
    [switch]$SkipVerify,
    [switch]$SkipKernel
)

$ErrorActionPreference = "Stop"
if ($env:PYTHON_VERSION) { $Python = $env:PYTHON_VERSION }
if ($env:TORCH_VARIANT) { $Torch = $env:TORCH_VARIANT }

# Deactivate any active venv immediately
if ($env:VIRTUAL_ENV) {
    $deact = Join-Path $env:VIRTUAL_ENV 'Scripts\deactivate.ps1'
    if (Test-Path $deact) { & $deact }
    $env:VIRTUAL_ENV = $null
}

# Remove .venv\Scripts from PATH so uv resolves from system
$cleanPath = ($env:PATH -split ';') | Where-Object { $_ -notlike '*\.venv\*' }
$env:PATH = $cleanPath -join ';'

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptRoot

function Log {
    param([string]$m)
    Write-Host ''
    Write-Host ('==> ' + $m) -ForegroundColor Cyan
}

function Warn {
    param([string]$m)
    Write-Host ('WARNING: ' + $m) -ForegroundColor Yellow
}

function Die {
    param([string]$m)
    Write-Host ('ERROR: ' + $m) -ForegroundColor Red
    exit 1
}

# --- Check project root ---
Log 'Checking project root'
if (-not (Test-Path 'pyproject.toml')) { Die 'pyproject.toml not found.' }
if (-not (Test-Path 'uv.lock')) { Die 'uv.lock not found.' }

# --- Ensure uv ---
Log 'Checking uv'
$uvCmd = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvCmd) {
    Log 'Installing uv'
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $p1 = Join-Path $env:USERPROFILE '.local\bin'
    $p2 = Join-Path $env:USERPROFILE '.cargo\bin'
    if (Test-Path $p1) { $env:PATH = $p1 + ';' + $env:PATH }
    if (Test-Path $p2) { $env:PATH = $p2 + ';' + $env:PATH }
    $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uvCmd) { Die 'uv not found after install. Reopen terminal.' }
}

# Store absolute path to uv so it survives venv deletion
$UV = $uvCmd.Source
if (-not $UV) { $UV = (Get-Command uv).Source }
Write-Host ('uv path: ' + $UV)
& $UV --version

# --- Ensure Python ---
Log ('Checking Python ' + $Python)
$pyFound = & $UV python find $Python 2>$null
if ($LASTEXITCODE -ne 0) {
    Log ('Installing Python ' + $Python)
    & $UV python install $Python
    if ($LASTEXITCODE -ne 0) { Die 'Python install failed' }
}

# --- Recreate venv ---
if ($Recreate) {
    Log 'Removing .venv'
    if (Test-Path '.venv') {
        # Deactivate if active
        if ($env:VIRTUAL_ENV) { & deactivate 2>$null }
        # Kill any python processes from this venv
        $venvPath = (Resolve-Path '.venv').Path
        Get-Process python*, pip* -ErrorAction SilentlyContinue | Where-Object {
            $_.Path -and $_.Path.StartsWith($venvPath)
        } | Stop-Process -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
        # Use cmd rmdir which handles locked files better
        & cmd /c "rmdir /s /q .venv" 2>$null
        if (Test-Path '.venv') {
            Remove-Item -Recurse -Force '.venv' -ErrorAction SilentlyContinue
        }
    }
}

# --- Sync environment ---
Log 'Syncing environment from uv.lock'
& $UV sync --frozen --python $Python
if ($LASTEXITCODE -ne 0) { Die 'uv sync failed' }

# --- Resolve venv Python path explicitly ---
$VenvPython = Join-Path $ScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $VenvPython)) { Die '.venv\Scripts\python.exe not found after uv sync.' }
Write-Host ('venv python: ' + $VenvPython)

# --- Ensure pip ---
Log 'Ensuring pip'
$ErrorActionPreference = "Continue"
& $VenvPython -m ensurepip --upgrade 2>$null
& $VenvPython -m pip install --upgrade pip setuptools wheel 2>$null
$ErrorActionPreference = "Stop"

# --- Verify pip works ---
$pipVer = & $VenvPython -m pip --version 2>&1
if ($LASTEXITCODE -ne 0) { Die ('pip is broken: ' + $pipVer) }
Write-Host ('pip: ' + $pipVer)

# --- Install PyTorch ---
if ($Torch -eq 'none') {
    Log 'Skipping PyTorch (--Torch none)'
}
elseif ($Torch -eq 'auto') {
    Log 'Keeping PyTorch from uv.lock (--Torch auto)'
}
else {
    $indexUrl = ''
    if ($Torch -eq 'cpu') { $indexUrl = 'https://download.pytorch.org/whl/cpu' }
    if ($Torch -eq 'cu126') { $indexUrl = 'https://download.pytorch.org/whl/cu126' }
    if ($Torch -eq 'cu128') { $indexUrl = 'https://download.pytorch.org/whl/cu128' }

    Log ('Installing PyTorch ' + $Torch + ' from ' + $indexUrl)
    & $VenvPython -m pip install --force-reinstall --index-url $indexUrl torch torchvision torchaudio
    if ($LASTEXITCODE -ne 0) { Die 'PyTorch install failed' }
}

# --- Local directories ---
Log 'Creating local directories'
$dirs = @(
    'data\raw\refseq_bacteria_protein',
    'data\compiled\refseq_bacteria_protein',
    'data\compiled\refseq_bacteria_profile_pretrain',
    'data\cache\protein_train_parts',
    'data\checkpoints\protein_from_scratch',
    'libs\data\models\catalog',
    'libs\data\models\datasets',
    'libs\data\models\trash',
    'libs\data\models\sessions'
)
foreach ($d in $dirs) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
}

$hasEnv = Test-Path '.env'
$hasExample = Test-Path '.env.example'
if ((-not $hasEnv) -and $hasExample) {
    Warn '.env not found. Copy .env.example if you need MinIO/NCBI credentials.'
}

# --- Verify ---
if (-not $SkipVerify) {
    Log 'Verifying Python'
    & $VenvPython -c "import sys; print(f'Python {sys.version}')"
    if ($LASTEXITCODE -ne 0) { Die 'Python verification failed' }

    & $VenvPython -c "import pip; print(f'pip {pip.__version__}')"

    if (($Torch -eq 'cu126') -or ($Torch -eq 'cu128')) {
        Log 'Checking NVIDIA driver'
        $nvsmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
        if ($nvsmi) { & nvidia-smi }
        else { Warn 'nvidia-smi not found. CUDA verification may fail.' }
    }

    if ($Torch -ne 'none') {
        Log 'Verifying PyTorch'
        & $VenvPython -c @"
import torch
print(f'torch {torch.__version__}')
print(f'CUDA compiled: {torch.version.cuda}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
"@
        if ($LASTEXITCODE -ne 0) { Die 'PyTorch import verification failed' }

        if (($Torch -eq 'cu126') -or ($Torch -eq 'cu128')) {
            Log 'Verifying CUDA tensor allocation'
            & $VenvPython -c @"
import torch
assert torch.cuda.is_available(), 'CUDA not available but variant is $Torch'
x = torch.randn(256, 256, device='cuda')
print(f'CUDA tensor OK on {x.device}')
"@
            if ($LASTEXITCODE -ne 0) { Warn 'GPU tensor test failed. Check nvidia-smi and driver version.' }
        }
    }

    Log 'Verifying project import'
    & $VenvPython -c "from libs.data.config import DataConfig; print('OK: libs.data.config importable')"
    if ($LASTEXITCODE -ne 0) { Die 'Project import verification failed' }
}

# --- Jupyter kernel ---
if (-not $SkipKernel) {
    Log 'Installing Jupyter kernel'
    & $VenvPython -m pip install --upgrade ipykernel 2>$null
    & $VenvPython -m ipykernel install --user --name microbial-dna-compiler --display-name 'Microbial DNA Compiler (uv GPU)'
}

# --- Done ---
Write-Host ''
$line = '=================================================================='
Write-Host $line -ForegroundColor Green
Write-Host ('  DONE. Torch=' + $Torch + '  Python=' + $Python) -ForegroundColor Green
Write-Host $line -ForegroundColor Green
Write-Host ''
Write-Host 'GPU test:'
Write-Host '  uv run python -c "import torch; print(torch.cuda.is_available())"'
Write-Host ''
Write-Host 'Run tests:'
Write-Host '  uv run python -m pytest tests/'
Write-Host ''
