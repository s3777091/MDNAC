# Structure Predictor API: OpenFold

This is GPU Worker 2. It accepts a completed protein sequence, runs OpenFold, and returns a
predicted structure as `.pdb` by default or `.cif` when requested.

OpenFold source and docs:

- https://github.com/aqlaboratory/openfold
- https://openfold.readthedocs.io/en/latest/Inference.html
- https://openfold.readthedocs.io/en/latest/Single_Sequence_Inference.html

## Worker boundary

Keep this worker separate from sequence completion:

```text
Worker 1: Sequence Completion -> POST /generate -> protein_sequence
Worker 2: Structure Predictor -> POST /predict-structure -> structure.pdb or structure.cif
```

The API wrapper lives in:

```text
api/structure_predictor/
api/config.structure.yaml
api/runpod_structure_app.py
```

## RunPod viability

This can run on RunPod if the worker image or mounted volume already contains OpenFold, CUDA,
model weights, and any required data. Do not rely on `runpod-flash` Python dependencies alone to
install OpenFold; OpenFold has a Linux/CUDA/PyTorch runtime, optional compiled kernels, and large
resource downloads.

Recommended first RunPod path:

```text
OpenFold SoloSeq
config_preset: seq_model_esm1b_ptm
checkpoint: /runpod-volume/openfold/resources/openfold_soloseq_params/seq_model_esm1b_ptm.pt
output: PDB, or mmCIF with --cif_output
```

SoloSeq avoids the full MSA database path for early testing, but OpenFold documents a 1022-residue
limit for this mode. Switch to MSA-based OpenFold later by filling `database_paths` in
`api/config.structure.yaml`.

## Expected production paths

Production defaults in `api/config.structure.yaml`:

```text
/opt/openfold
/runpod-volume/openfold/resources/openfold_soloseq_params/seq_model_esm1b_ptm.pt
/runpod-volume/openfold/data/pdb_mmcif/mmcif_files
/runpod-volume/mdnac/structure_predictions
```

The positional `template_mmcif_dir` is still required by OpenFold even when templates are not used,
so keep the directory present in the image or volume.

## Local API

```powershell
cd api
uv sync --extra structure
uv run mdnac-structure-api --env local
```

Request:

```powershell
curl -X POST http://127.0.0.1:8010/predict-structure `
  -H "Content-Type: application/json" `
  -d "{\"sequence\":\"MPEPTIDE\",\"name\":\"candidate_001\",\"output_format\":\"pdb\"}"
```

Response includes:

- `structure_format`: `pdb` or `cif`
- `structure_path`: stable path under the configured output root, ending in `structure.pdb` or `structure.cif`
- `structure_text`: inline file text when the structure is smaller than `max_response_structure_bytes`
- `stdout_tail` and `stderr_tail`: short OpenFold logs for debugging

## RunPod deploy

Use the structure config and entrypoint:

```powershell
cd api
uv sync --extra structure
$env:MDNAC_STRUCTURE_ENV="production"
flash run runpod_structure_app.py
flash deploy runpod_structure_app.py
```

If your `flash` command expects the default app file, deploy from this directory with
`runpod_structure_app.py` selected as the app module.

## MSA-based mode

For full MSA-based OpenFold, change:

```yaml
openfold:
  config_preset: model_3_ptm
  use_single_seq_mode: false
  openfold_checkpoint_path:
  jax_param_path: /runpod-volume/openfold/resources/params/params_model_3_ptm.npz
  database_paths:
    uniref90_database_path: /runpod-volume/openfold/data/uniref90/uniref90.fasta
    mgnify_database_path: /runpod-volume/openfold/data/mgnify/mgy_clusters_2018_12.fa
    pdb70_database_path: /runpod-volume/openfold/data/pdb70/pdb70
    uniclust30_database_path: /runpod-volume/openfold/data/uniclust30/uniclust30_2018_08/uniclust30_2018_08
    bfd_database_path: /runpod-volume/openfold/data/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt
```

OpenFold release lines may use `uniref30_database_path` instead of `uniclust30_database_path`, so
verify the installed `run_pretrained_openfold.py --help` inside the final worker image.
