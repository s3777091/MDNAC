# Training Configs

- `train.yaml`: default protein pretraining config used when code does not pass `config_path`.
- `train.16gb.yaml`: fallback for smaller 16GB-style GPU training.
- `train.64gb.2gpu.yaml`: notebook default for 64GB RAM + 2 GPU protein pretraining.
- `train.resume.yaml`: resume protein pretraining from a checkpoint.
- `instruction.16gb.yaml`: stage-3 instruction tuning config.

Notebook platform differences are handled by `libs/notebook_runtime.py`; keep config files workload-specific, not OS-specific.

Notebook override environment variables:

- `MDNAC_TRAIN_CONFIG`: protein pretrain/eval YAML path.
- `MDNAC_INSTRUCTION_CONFIG`: instruction tuning YAML path.
