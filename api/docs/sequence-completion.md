# Sequence Completion API

This worker completes protein sequences with the exported MDNAC ONNX model.

## Local dev

```powershell
cd api
uv sync --extra local
uv run mdnac-api --env local
```

The local environment reads `api/config.yaml` and loads exactly one ONNX model from:

```text
api/data/model
```

Generate:

```powershell
curl -X POST http://127.0.0.1:8000/generate `
  -H "Content-Type: application/json" `
  -d "{\"prompt\":\"MPEPTIDE\",\"max_new_tokens\":64}"
```

## Export PyTorch checkpoint to ONNX

Run this from the `api` directory:

```powershell
uv sync --extra export
uv run mdnac-export-onnx --checkpoint ..\data\checkpoints\your_run\checkpoint_best.pt
```

Default output:

```text
api/data/model/<run_name>.onnx
api/data/model/<run_name>.json
```

The JSON sidecar embeds the tokenizer map, so inference can tokenize prompts without importing
the training project.

## RunPod production

Production config is in the `production` environment of `api/config.yaml`.

The default model path is:

```text
/runpod-volume/mdnac/model
```

Put the exported `.onnx` and sidecar `.json` there, then deploy the sequence completion worker:

```powershell
cd api
uv sync --extra production
flash run
flash deploy
```

`runpod_app.py` exposes:

- `GET /health`
- `GET /ready`
- `POST /generate`
