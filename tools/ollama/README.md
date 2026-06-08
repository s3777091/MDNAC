# Ollama Export

This directory contains the checkpoint export helper that packages a repo Qwen3.5 checkpoint for the Ollama path:

```text
checkpoint_best.pt -> HF-style export -> GGUF -> Modelfile -> ollama create
```

## Export a checkpoint

```powershell
python tools\ollama\export_checkpoint.py `
  --checkpoint data\checkpoints\ava_tiny_0_8b\checkpoint_best.pt `
  --output-dir tools\ollama\exports `
  --ollama-model-name ava_tiny_instruction-ollama `
  --verify
```

The command writes a unique export folder containing:

- `hf/` with `config.json`, `generation_config.json`, tokenizer files, weights, and export metadata
- `Modelfile`
- `convert_to_gguf.ps1`
- `create_ollama_model.ps1`
- `ollama_export_summary.json`

## Verification behavior

`--verify` compares repo logits against a Hugging Face-compatible Qwen3.5 model class.
If a compatible `transformers` stack is not available, the export still succeeds and the summary records:

```json
{
  "verified": false,
  "verification_skipped": true,
  "reason": "transformers_unavailable"
}
```

When verification dependencies are present, the summary includes `max_abs_diff` and `seq_len`.

Common notebook issue:
If verification is skipped with a `huggingface-hub` version error, install a compatible stack in the active kernel, for example:

```powershell
pip install "huggingface_hub<1.0" "transformers>=4.45"
```

If you use the repo-managed environment, `uv sync` is the preferred fix.

## GGUF conversion

You can run GGUF conversion during export with `--llama-cpp-dir`, or later with the generated helper script:

```powershell
.\tools\ollama\exports\<your_export>\convert_to_gguf.ps1 -LlamaCppDir C:\path\to\llama.cpp
```

## Create the Ollama model

After `model.gguf` exists:

```powershell
.\tools\ollama\exports\<your_export>\create_ollama_model.ps1 -ModelName ava_tiny_instruction-ollama
```
