# Microbial DNA Compiler

Current training flow:

`{profile, sequence}` pairs -> `train.txt` + `tokenizer_map.json` -> MDC decoder-only pretraining/fine-tuning

## Installation

### Prerequisites

- **Python 3.11+** (managed automatically by `uv`)
- **uv** package manager (installed automatically if missing)
- **NVIDIA driver** with `nvidia-smi` working (for GPU variants only)

### Windows (PowerShell)

```powershell
# Auto-detect GPU and install best variant (recommended):
.\install.ps1 -Recreate

# Explicit CUDA 12.8 (RTX 40xx/50xx, CUDA driver 12.8+):
.\install.ps1 -Recreate -Torch cu128

# Explicit CUDA 12.6 (older GPUs, CUDA driver 12.6-12.7):
.\install.ps1 -Recreate -Torch cu126

# CPU only (no GPU):
.\install.ps1 -Recreate -Torch cpu

# Skip torch entirely:
.\install.ps1 -Recreate -Torch none
```

Optional flags: `-SkipVerify`, `-SkipKernel`, `-Python 3.12`.

### Linux (bash)

```bash
# Auto-detect GPU (recommended):
bash install.sh --recreate

# Explicit CUDA 12.8:
bash install.sh --recreate --torch cu128

# Explicit CUDA 12.6:
bash install.sh --recreate --torch cu126

# CPU only:
bash install.sh --recreate --torch cpu

# Skip verification and kernel install:
bash install.sh --recreate --skip-verify --skip-kernel
```

### Verify GPU

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

On a working GPU machine this prints `True` for `cuda.is_available()` and shows the CUDA version.

**Important**: `nvidia-smi` must work before CUDA torch can pass GPU verification. If it fails, install or update your NVIDIA driver first.

### Which CUDA variant to use

| nvidia-smi CUDA Version | Recommended variant |
|--------------------------|---------------------|
| 13.x or 12.8+          | `cu128`            |
| 12.6 – 12.7            | `cu126`            |
| < 12.6 or no GPU       | `cpu`              |

The default (`auto`) runs `nvidia-smi`, parses the CUDA version, and picks automatically.

### How torch installation works

1. `uv sync --frozen` installs torch from `uv.lock` (PyPI default, usually CPU on Windows).
2. The installer uses `uv pip install --reinstall --index-url <wheel-index>` to replace torch with the correct CUDA/CPU variant.
3. `pyproject.toml` declares `torch>=2.11` in base dependencies so `uv sync` always resolves a torch version compatible with the lock file.

## Local RefSeq Build

The local RefSeq compiler now writes three training artifacts:

- `train.txt` for sequence-only pretraining
- `tokenizer_map.json` for the protein tokenizer
- `instruction.jsonl` for metadata-profile-to-protein instruction tuning

`train.txt` now contains protein sequences only:

```text
<|protein|>MPEPTIDE<|endoftext|>
<|protein|>GLYSERQ<|endoftext|>
```

`instruction.jsonl` keeps the conditioning side separate, so the pretrain corpus stays sequence-only while instruction tuning can still learn from RefSeq-derived protein metadata. Each line is one protein example whose `instruction` is built from the protein profile metadata:

```json
{"instruction":"labels nitrogen fixation; keywords nitrogen fixation, nitrogenase; description nitrogen fixation protein; organism Bacillus subtilis; gene nifH; product nitrogenase iron protein; note nitrogen fixation regulator","input":"","output":"MPEPTIDE"}
```

This matches the intended two-stage flow:

1. a stronger upstream model converts a user request such as `how to increase drought tolerance of crops` into a protein-profile metadata string
2. the MDC sequence model consumes that metadata-style instruction and predicts a protein sequence

Build or rebuild the local RefSeq artifacts:

```bash
uv run python cmd/build_refseq_profile_text.py data/raw/refseq_bacteria_protein -o data/compiled/refseq_bacteria_protein --vocab-size 512 --instruction-min-proteins 10
```

For larger local RefSeq builds, add `--workers 0` to auto-use all detected CPU cores for the CPU-bound record compilation and `instruction.jsonl` rendering steps.

Use `--skip` when you only want to refresh part of the output set:

```bash
uv run python cmd/build_refseq_profile_text.py ... --skip tokenizer_map.json
uv run python cmd/build_refseq_profile_text.py ... --skip train,instruction.jsonl
uv run python cmd/build_refseq_profile_text.py --rebuild-tokenizer-map-from-train -o data/compiled/refseq_bacteria_protein
```

`--skip tokenizer_map.json` means do not write `tokenizer_map.json`.

`--skip train` requires an existing `train.txt` if `tokenizer_map.json` is still being written, because the tokenizer map is rebuilt from the on-disk training corpus when `train.txt` itself is skipped.

`--rebuild-tokenizer-map-from-train` is the explicit mode for rebuilding only `tokenizer_map.json` from the existing `train.txt` inside `--output-dir`, without rescanning the RefSeq archives.

Tokenizer builds are resumable by default. During `tokenizer_map.json` creation, the command writes a `sequence-tokenizer-resume-*.state.json` checkpoint and a matching cache file into `--output-dir` after the initial cache pass and after each completed BPE merge. If the process or VM stops before completion, rerun the same command with the same `train.txt`, `--vocab-size`, and `--tokenizer-train-line-limit`; it resumes from the last completed merge. Use `--no-tokenizer-resume` to force the old scratch-build behavior.

For large tokenizer rebuilds, add `--tokenizer-workers 0` to use all detected CPU cores for BPE cache counting and rewrite work, or pass a fixed worker count such as `--tokenizer-workers 8`.

The compiler is append-only. Re-running against the same output directory appends the current batch into `train.txt` and `instruction.jsonl`, then rebuilds `tokenizer_map.json` from the on-disk `train.txt`. It does not use `summary.json` or `history.json`.

## Protein Pretraining

The primary notebook is now:

- **`notebooks/stage_2_foundation_model/protein_pretrain.ipynb`** — unified training notebook

Legacy notebooks (kept for backward compatibility):

- `notebooks/stage_2_foundation_model/03_pretrain_protein_from_scratch.ipynb`
- `notebooks/stage_2_foundation_model/04_resume_protein_pretrain.ipynb`
- `notebooks/stage_2_foundation_model/06_model_evaluation/06_top1_benchmark.ipynb`
- `notebooks/stage_2_foundation_model/06_model_evaluation/07_plot_metrics.ipynb`

### Configuration

The unified notebook accepts `train.yaml` via three options:

1. **Default** — uses `train.yaml` at the repo root.
2. **Upload** — upload a custom YAML file (supports Colab, ipywidgets, or path input).
3. **Custom path** — specify an arbitrary filesystem path.

### Training Modes

| Mode | Description |
|------|-------------|
| `train_from_scratch` | Build tokenizer, create model, train from epoch 0 |
| `resume` | Restore from checkpoint and continue training |
| `auto` | Resume if checkpoint/resume_state.json exists, otherwise train from scratch |

### Key Features

- **Muon optimizer** is the default (`optimizer.type: muon`). Only use AdamW when explicitly set.
- **MinIO streaming** downloads parts on demand (one at a time), not the full dataset upfront.
- **`resume_state.json`** tracks progress (epoch, step, tokens, completed parts) for resumable runs.
- **`metrics_history.jsonl`** appends eval metrics at each checkpoint.
- **Supports**: CPU, single GPU, DataParallel, DDP (via torchrun).

All training logic lives in `libs.core.pretrain.protein_lm.trainer.ProteinPretrainTrainer`, which reuses existing helpers from `libs.core`.

The pretrain notebook loads shared paths, model settings, optimizer choice (`muon` default), multi-GPU options, and optional MinIO overrides from `train.yaml` at the repo root. Keep sensitive MinIO credentials in `.env` or environment variables and only put non-secret endpoint or bucket overrides in `train.yaml` when needed.

The notebooks call `libs.core` helpers to build or load the protein `SequenceTokenizer`, create causal-LM batches from `train.txt`, instantiate ProGen backbone configs for the MDC decoder, save/load resumable `progen_protein_lm` checkpoints, and benchmark protein next-token accuracy.

For large corpora, use the streaming dataloaders so training reads one text part at a time instead of loading one huge `train.txt` into memory. Local shards named `train_part_1.txt`, `train_part_2.txt`, ... are discovered with `discover_protein_train_text_paths`; MinIO/S3 shards can use the same names under one prefix.

```python
from libs.core import build_or_load_protein_tokenizer, create_streaming_protein_lm_dataloader

tokenizer_artifact = build_or_load_protein_tokenizer("data/compiled/refseq_bacteria_protein/train.txt")
train_loader = create_streaming_protein_lm_dataloader(
    tokenizer_artifact.tokenizer,
    prefix_uri="s3://microbial-dna-compiler/libs/data/models/datasets/refseq/protein/current/parts",
    context_length=512,
    batch_size=8,
    cache_dir="data/cache/minio-train-parts",
)
```

For local parts:

```python
from libs.core import discover_protein_train_text_paths, create_streaming_protein_lm_dataloader

part_paths = discover_protein_train_text_paths("data/compiled/refseq_bacteria_protein/train.txt")
train_loader = create_streaming_protein_lm_dataloader(
    tokenizer_artifact.tokenizer,
    part_paths=part_paths,
    context_length=512,
    batch_size=8,
)
```

For profile-aware pretraining, load the small `tokenizer_map.json` locally and stream only the train parts:

```python
from libs.core import (
    MDCProfileSequencePretrainArtifacts,
    create_streaming_mdc_profile_sequence_pretrain_dataloader,
)

artifacts = MDCProfileSequencePretrainArtifacts.from_tokenizer_map_file(
    "data/compiled/refseq_bacteria_profile_pretrain/tokenizer_map.json"
)
train_loader = create_streaming_mdc_profile_sequence_pretrain_dataloader(
    artifacts,
    prefix_uri="s3://microbial-dna-compiler/libs/data/models/datasets/refseq/profile/current/parts",
    batch_size=8,
    cache_dir="data/cache/minio-profile-parts",
)
```

By default each downloaded part is removed after it has been consumed. Set `keep_downloaded_parts=True` when you want the local cache to persist across epochs.

To pretrain on the same metadata-to-protein shape used by `instruction.jsonl`, build profile-aware pretrain artifacts from the JSONL file:

```powershell
cmd\build_profile_pretrain_from_instruction_jsonl.cmd data\compiled\refseq_bacteria_protein\instruction.jsonl -o data\compiled\refseq_bacteria_profile_pretrain
```

```bash
bash cmd/build_profile_pretrain_from_instruction_jsonl.sh data/compiled/refseq_bacteria_protein/instruction.jsonl -o data/compiled/refseq_bacteria_profile_pretrain
```

This writes a profile-aware `train.txt` where each line keeps `instruction` and optional `input` as the conditioning profile, followed by the protein `output` target. If `instruction.jsonl` sits next to the stage-1 protein `tokenizer_map.json`, the command auto-loads that map and preserves protein token IDs for instruction tuning. You can also pass it explicitly:

```powershell
cmd\build_profile_pretrain_from_instruction_jsonl.cmd data\compiled\refseq_bacteria_protein\instruction.jsonl -o data\compiled\refseq_bacteria_profile_pretrain --protein-tokenizer-map data\compiled\refseq_bacteria_protein\tokenizer_map.json
```

Use `--legacy-kmer-tokenizer` only when you intentionally want the older stage-2-only k-mer target tokenizer. That mode runs, but it does not cleanly reuse the stage-1 protein embedding rows.

If you need to collapse duplicates introduced by repeated append-only runs, use the dedupe command:

```powershell
cmd\dedupe_refseq_profile_text.cmd data\compiled\refseq_bacteria_protein
cmd\dedupe_refseq_profile_text.cmd data\compiled\refseq_bacteria_protein --dry-run
```

```bash
bash cmd/dedupe_refseq_profile_text.sh data/compiled/refseq_bacteria_protein
bash cmd/dedupe_refseq_profile_text.sh data/compiled/refseq_bacteria_protein --dry-run
```

The dedupe pass removes duplicate non-empty lines from both `train.txt` and `instruction.jsonl` while preserving the first occurrence. It only touches those two files so the pass stays I/O-bound and fast on large append-only corpora.

If you already have separate corpus shards and only want to join them, use the concat command:

```powershell
cmd\concat_text_files.cmd data\a\instruction.jsonl data\b\instruction.jsonl -o data\instruction.merged.jsonl
cmd\concat_text_files.cmd data\a\train.txt data\b\train.txt -o data\train.merged.txt --overwrite
```

```bash
bash cmd/concat_text_files.sh data/a/instruction.jsonl data/b/instruction.jsonl -o data/instruction.merged.jsonl
bash cmd/concat_text_files.sh data/a/train.txt data/b/train.txt -o data/train.merged.txt --overwrite
```

This is a streaming file concatenation pass. It keeps input order, does not parse JSONL, does not validate records, and does not remove duplicate lines. By default it inserts one separator newline only when a file boundary would otherwise glue two records together. Add `--raw` for exact byte concatenation.

If `instruction.jsonl` is too large to train on directly, downsample it with a streaming two-pass sampler that preserves coverage across `dataset_group x product` buckets while compressing extremely repeated proteins:

```powershell
cmd\downsample_instruction_jsonl.cmd data\instruction.jsonl -o data\instruction.50pct.jsonl --keep-ratio 0.5 --alpha 0.8
cmd\downsample_instruction_jsonl.cmd data\instruction.jsonl --dry-run --keep-ratio 0.5
```

```bash
bash cmd/downsample_instruction_jsonl.sh data/instruction.jsonl -o data/instruction.50pct.jsonl --keep-ratio 0.5 --alpha 0.8
bash cmd/downsample_instruction_jsonl.sh data/instruction.jsonl --dry-run --keep-ratio 0.5
```

Unlike chopping the file head/tail, this keeps at least one example per protein bucket, preserves dataset-group balance, and spreads selected records across each bucket with deterministic systematic sampling.

To add Foldseek-style `3Di` structure tokens to `instruction.jsonl`, use the reusable structure pipeline in `libs.core.structure` or run:

```text
notebooks/stage_2_foundation_model/05_update_instruction_3di_from_s3.ipynb
```

The notebook streams `.jsonl` parts from MinIO/S3, annotates missing top-level `3Di` fields from each protein `output`, uploads annotated parts to a separate prefix, and writes `manifest.3di.json`. It uses a SQLite cache so repeated protein sequences and resumed runs do not call the 3Di model again. The ProstT5 adapter is optional and loads `Rostlab/ProstT5` only when the notebook creates the provider; install `transformers` and `sentencepiece` in the active environment before running it.

If the output folder name matches a direct child folder under the input root, the build automatically scopes to that child folder. For example:

```bash
uv run python cmd/build_refseq_profile_text.py data/raw/refseq_bacteria_protein -o data/compiled/refseq_bacteria_protein/fungi
```

This will only compile files under `data/raw/refseq_bacteria_protein/fungi` and append that subset into `data/compiled/refseq_bacteria_protein/fungi`.

The maintained data workflow is command-line driven through `cmd/`. The older notebook-based data-fetch flow has been removed.

## What it does

- parses local annotation files (`.gff/.gff3`, `.gbk/.gbff`)
- extracts functional profiles from annotation keywords such as `drought tolerance` or `photosystem`
- builds supervised pairs like `{ "profile": "DNA gyrase subunit B", "sequence": "MPEPTIDE..." }`
- keeps one current on-disk training format: `train.txt` + `tokenizer_map.json`
- stores profile text and target sequence in the same training line
- tokenizes profile text with a from-scratch BPE tokenizer
- tokenizes protein targets with the stage-1 protein BPE tokenizer for instruction tuning
- loads the text corpus back into the MDC fused decoder input
- pulls sequences from ENA by query
- pulls sequences from DDBJ by accession list
- pulls protein sequences from NCBI by Entrez query or accession list
- normalizes protein sequences into a consistent amino-acid alphabet
- writes one training corpus file: `train.txt`
- writes one tokenizer map file: `tokenizer_map.json`
- can resume interrupted long-running data preparation jobs
- keeps dataset history, catalog listing, and delete/trash workflows

## MDC Profile-Aware Text Format

When you want to keep the `train.txt` + `tokenizer_map.json` workflow but still preserve the user conditioning profile, use the MDC profile-aware text format.

Generated outputs:

- `train.txt`
- `tokenizer_map.json`

Each line in `train.txt` keeps both the profile prompt and the target sequence:

```text
<|profile|>dna gyrase subunit b<|sep|><|protein|>MPEPTIDE<|endoftext|>
<|profile|>stress response protein<|sep|><|protein|>GLYSERQ<|endoftext|>
```

The combined `tokenizer_map.json` stores:

- one profile BPE tokenizer
- one sequence tokenizer, preferably the stage-1 protein `SequenceTokenizer`
- the fused vocabulary layout used by `libs/core`

When a stage-1 protein tokenizer is used, the fused layout keeps protein token IDs unchanged:

```text
0..protein_vocab-1        protein tokenizer rows from stage 1
protein_vocab..N          profile tokenizer rows
N                         <|sep|>
```

That lets a stage-1 checkpoint load into the expanded profile-tuning model by copying the original protein embedding and output-head rows directly.

This lets the data stay text-first on disk, while the loader still reconstructs the exact MDC input:

```text
[BOS] profile_ids [SEP] sequence_ids [EOS]
```

Example:

```python
from pathlib import Path

from libs.core import (
    MDCProfileSequencePretrainArtifacts,
    MDCProfileSequencePretrainDataset,
    MDCProfileSequenceRecord,
    create_mdc_profile_sequence_pretrain_dataloader,
    save_mdc_profile_sequence_pretrain_artifacts,
)

records = [
    MDCProfileSequenceRecord(
        profile="dna gyrase subunit b",
        sequence="MPEPTIDE",
        sequence_type="protein",
    ),
    MDCProfileSequenceRecord(
        profile="stress response protein",
        sequence="GLYSERQ",
        sequence_type="protein",
    ),
]

artifact = save_mdc_profile_sequence_pretrain_artifacts(
    records,
    output_dir=Path("artifacts/mdc-profile-text"),
    sequence_type="protein",
    profile_vocab_size=256,
    sequence_tokenizer_map_path=Path("data/compiled/refseq_bacteria_protein/tokenizer_map.json"),
)

artifacts = MDCProfileSequencePretrainArtifacts.from_directory(artifact.output_dir)
dataset = MDCProfileSequencePretrainDataset.from_artifacts(artifacts)
data_loader = create_mdc_profile_sequence_pretrain_dataloader(
    dataset,
    batch_size=4,
    shuffle=True,
)
```

The dataloader yields masked causal-LM batches where the loss is applied to the sequence side by default, not to the profile prompt.

## Tokenizer style

The tokenizer implementation in `libs/data/training/tokenizer.py` keeps a simple `encode` / `decode` interface, but now follows the BPE direction discussed later in chapter 2:

- `str_to_int`
- `int_to_str`
- `encode(...)`
- `decode(...)`

Important behavior:

- there is no `<|unk|>` fallback token
- the tokenizer starts from known sequence symbols and special tokens
- frequent pairs are merged into larger sequence tokens during training
- unseen invalid characters raise an error instead of being silently collapsed

For this project, the BPE vocabulary is specialized for sequence training instead of generic natural language.

## Project layout

```text
cmd/                  maintained command-line data entrypoints
libs/
  core/               MDC model, fusion, pretrain compiler
  data/
    backends/          dataset storage backends (local, MinIO)
    sources/           ENA, DDBJ, and NCBI fetchers
    training/
      normalization.py protein cleanup and filtering
      tokenizer/       BPE + k-mer tokenizers
      preparation/     resumable session-based fetch pipeline
      raw_pipeline/    FASTA/GFF/GenBank -> profile-sequence pairs
    utilities/         HTTP transport, parsers, storage helpers
models/
tests/
```

Local datasets are stored under `libs/data/models`:

```text
libs/data/models/
  catalog/
    datasets.csv
  datasets/
    ena/
      plant-root-bacteria/
        current/
          train.txt
          tokenizer_map.json
        history/
          20260412_101530_123456/
            ...
  trash/
    ena/
      plant-root-bacteria/
        20260412_111010_999999/
          ...
  sessions/
    ena/
      plant-root-bacteria/
        accessions.txt
        manifest.json
        train.txt
```

## Configuration

`config.yaml` is intentionally small now:

```yaml
storage_mode: local
data_root: libs/data
default_batch_size: 25
```

Environment variables that still override YAML:

- `MICROBIAL_DATA_STORAGE_MODE=local|minio`
- `MICROBIAL_DATA_ROOT=libs/data`
- `MICROBIAL_DATA_DEFAULT_BATCH_SIZE=25`
- `MICROBIAL_DATA_MINIO_ENDPOINT=https://s3.phuongdong.cloud`
- `MICROBIAL_DATA_MINIO_ACCESS_KEY=minioadmin`
- `MICROBIAL_DATA_MINIO_SECRET_KEY=minioadmin`
- `MICROBIAL_DATA_MINIO_BUCKET=microbial-dna-compiler`
- `MICROBIAL_DATA_MINIO_SECURE=true`
- `MICROBIAL_DATA_NCBI_TOOL=microbial-dna-compiler`
- `MICROBIAL_DATA_NCBI_EMAIL=you@example.com`
- `MICROBIAL_DATA_NCBI_API_KEY=...`

## Notes about sources

- **ENA** - query-driven. No credentials required.
- **DDBJ** - accession-driven because the stable `getentry` endpoint is accession-based. No credentials required.
- **NCBI** - E-utilities against the `protein` database. Supports Entrez queries and explicit accession lists. **Requires** `MICROBIAL_DATA_NCBI_EMAIL`; optionally set `MICROBIAL_DATA_NCBI_API_KEY` for higher rate limits (10 req/s vs 3 req/s).

All three sources retry on transient failures (empty responses, rate limiting, connection resets) with exponential backoff.

## Dependency management with uv

This repo now uses [`uv`](https://github.com/astral-sh/uv) as the default Python package manager.

Create or refresh the local environment:

```bash
uv sync
```

Include the optional MinIO backend dependencies:

```bash
uv sync --extra minio
```

Common dependency workflows:

```bash
uv add <package>
uv add --group dev <package>
uv add --optional minio <package>
uv remove <package>
```

Run commands inside the managed environment:

```bash
uv run python -m unittest discover -s tests -p "test_*.py"
```

### Ubuntu/Linux quick start

If you are setting up this repo on an Ubuntu or Linux server, use the bundled install script:

```bash
bash install.sh
```

The Linux equivalent of `cmd/build_refseq_profile_text.cmd` is:

```bash
bash cmd/build_refseq_profile_text.sh data/raw/refseq_bacteria_protein -o data/compiled/refseq_bacteria_protein/fungi/package_1 --vocab-size 256 --instruction-min-proteins 5 --workers 8
```

No notebook dependencies are installed by default.

## Run tests

```bash
uv run python -m unittest discover -s tests -p "test_*.py"
```
