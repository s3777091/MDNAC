# Command Entrypoints

- `cmd/install/`: environment installers for Windows, Ubuntu/Linux, and Tesla/V100-style GPU machines.
- `cmd/data/`: RefSeq build, dedupe, concat, and download utilities.
- `cmd/instruction/`: instruction JSONL artifact and downsampling utilities.

Run commands from the repository root, for example:

```bash
bash cmd/install/install.sh --recreate
bash cmd/data/build_refseq_profile_text.sh data/raw/refseq_bacteria_protein -o data/compiled/refseq_bacteria_protein
bash cmd/instruction/downsample_instruction_jsonl.sh data/instruction.jsonl --dry-run
```

