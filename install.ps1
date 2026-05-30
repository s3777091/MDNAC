#Requires -Version 5.1
param(
    [string]$Python = "3.11",
    [ValidateSet("cu126","cu128","cpu","auto","none")]
    [string]$Torch = "cu126",
    [switch]$Recreate,
    [switch]$SkipVerify,
    [switch]$SkipKernel
)

$ErrorActionPreference = "Continue"
if ($env:PYTHON_VERSION) { $Python = $env:PYTHON_VERSION }
if ($env:TORCH_VARIANT) { $Torch = $env:TORCH_VARIANT }

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
& uv --version

# --- Ensure Python ---
Log ('Checking Python ' + $Python)
$pyFound = & uv python find $Python 2>$null
if ($LASTEXITCODE -ne 0) {
    Log ('Installing Python ' + $Python)
    & uv python install $Python
    if ($LASTEXITCODE -ne 0) { Die 'Python install failed' }
}

# --- Recreate venv ---
if ($Recreate) {
    Log 'Removing .venv'
    if (Test-Path '.venv') { Remove-Item -Recurse -Force '.venv' }
}

# --- Sync environment ---
Log 'Syncing environment from uv.lock'
& uv sync --frozen --python $Python
if ($LASTEXITCODE -ne 0) { Die 'uv sync failed' }

# --- Ensure pip ---
Log 'Ensuring pip'
& uv run --no-sync python -m ensurepip --upgrade 2>$null
& uv run --no-sync python -m pip install --upgrade pip setuptools wheel 2>$null

# --- Install PyTorch ---
if ($Torch -eq 'none') {
    Log 'Skipping PyTorch'
}
elseif ($Torch -eq 'auto') {
    Log 'Keeping PyTorch from uv.lock'
}
else {
    $indexUrl = ''
    if ($Torch -eq 'cpu') { $indexUrl = 'https://download.pytorch.org/whl/cpu' }
    if ($Torch -eq 'cu126') { $indexUrl = 'https://download.pytorch.org/whl/cu126' }
    if ($Torch -eq 'cu128') { $indexUrl = 'https://download.pytorch.org/whl/cu128' }

    Log ('Installing PyTorch ' + $Torch + ' from ' + $indexUrl)
    & uv run --no-sync python -m pip install --reinstall --upgrade --index-url $indexUrl torch torchvision torchaudio
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
    & uv run --no-sync python -c 'import sys; print(sys.version)'

    if (($Torch -eq 'cu126') -or ($Torch -eq 'cu128')) {
        Log 'Checking NVIDIA driver'
        $nvsmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
        if ($nvsmi) { & nvidia-smi }
        else { Warn 'nvidia-smi not found' }
    }

    if ($Torch -ne 'none') {
        Log 'Verifying PyTorch'
        & uv run --no-sync python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'

        if (($Torch -eq 'cu126') -or ($Torch -eq 'cu128')) {
            & uv run --no-sync python -c 'import torch; assert torch.cuda.is_available(); x=torch.randn(256,256,device=chr(99)+chr(117)+chr(100)+chr(97)); print(x.device)'
            if ($LASTEXITCODE -ne 0) { Warn 'GPU test failed' }
        }
    }

    Log 'Verifying project import'
    & uv run --no-sync python -c 'from libs.data.config import DataConfig; print(DataConfig)'
}

# --- Jupyter kernel ---
if (-not $SkipKernel) {
    Log 'Installing Jupyter kernel'
    & uv run --no-sync python -m pip install --upgrade ipykernel 2>$null
    & uv run --no-sync python -m ipykernel install --user --name microbial-dna-compiler --display-name 'Microbial DNA Compiler (uv GPU)'
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
