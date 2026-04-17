@echo off
setlocal
if not defined UV_CACHE_DIR set "UV_CACHE_DIR=%~dp0..\.uv-cache"
uv run --no-sync python "%~dp0dedupe_refseq_profile_text.py" %*
