# ONNX Export

Script trong thu muc nay export training checkpoint `.pt` cua repo sang ONNX.
Exporter chi support checkpoint Qwen3.5 cua repo nay.

## Cai dependency

```powershell
uv sync --extra onnx
```

`onnx` la bat buoc de export. `onnxruntime` duoc cai cung extra nay de script co the verify output sau khi export.

## Export checkpoint

```powershell
uv run python tools\onnx\export_checkpoint.py `
  --checkpoint data\checkpoints\demo_full_real_ava_tiny\checkpoint_best.pt
```

Mac dinh script se:

- doc `model_config` tu checkpoint
- export sang `tools\onnx\exports\<ten_run>.onnx`
- ghi them file sidecar `.json` chua metadata de inference co the load lai tokenizer/model family
- neu co `onnxruntime`, so output ONNX voi PyTorch tren dummy input

ONNX graph hien chi co 1 input:

- `input_ids`

Co the override kich thuoc dummy input:

```powershell
uv run python tools\onnx\export_checkpoint.py `
  --checkpoint data\checkpoints\demo_full_real_ava_tiny\checkpoint_best.pt `
  --seq-len 32
```

## Tuy chon hay dung

- `--output`: doi duong dan file `.onnx`
- `--static-shapes`: export shape co dinh thay vi dynamic axes
- `--skip-verify`: bo qua buoc so output bang `onnxruntime`
- `--opset`: chon ONNX opset
