@echo off
setlocal
if not defined UV_CACHE_DIR set "UV_CACHE_DIR=%~dp0..\.uv-cache"
if defined PYTHONPATH (
    set "PYTHONPATH=%~dp0..;%PYTHONPATH%"
) else (
    set "PYTHONPATH=%~dp0.."
)
uv run --no-sync python "%~dp0downsample_instruction_jsonl.py" %*
