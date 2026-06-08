# MDNAC Protein APIs

This is a standalone API project. Keep its dependencies and runtime config inside `api/`.

The workers are intentionally split so each GPU runtime stays easy to deploy and debug:

- [Sequence Completion API](docs/sequence-completion.md): ONNX model, `POST /generate`
- [Structure Predictor API: OpenFold](docs/structure-predictor-openfold.md): OpenFold GPU worker, `POST /predict-structure`

Recommended pipeline:

```text
profile/prompt -> sequence completion -> protein_sequence -> OpenFold structure prediction -> PDB/mmCIF
```

Worker entrypoints:

```text
api/runpod_app.py            # Worker 1: sequence completion
api/runpod_structure_app.py  # Worker 2: OpenFold structure predictor
```
