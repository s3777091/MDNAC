# ONNX Export

This tool exports an MDNAC protein checkpoint (`.pt`) to ONNX for the standalone API.

Run from `api/`:

```powershell
uv sync --extra export
uv run mdnac-export-onnx --checkpoint ..\data\checkpoints\your_run\checkpoint_best.pt
```

By default it writes:

```text
api/data/model/<run_name>.onnx
api/data/model/<run_name>.json
```

The sidecar JSON includes:

- `model_config`
- embedded `tokenizer_map`
- ONNX input/output names
- export settings
- verification result

Useful options:

- `--output`: write to a specific `.onnx` path
- `--seq-len`: choose dummy export sequence length
- `--static-shapes`: disable dynamic batch/sequence axes
- `--skip-verify`: skip ONNX Runtime comparison
- `--opset`: choose ONNX opset
