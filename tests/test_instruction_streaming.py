from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from libs.instruction.trainer import InstructionTrainer, InstructionTrainingConfig


class InstructionStreamingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("tests/artifacts/instruction-streaming")
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_builds_artifacts_from_instruction_parts_when_merged_jsonl_is_missing(self) -> None:
        part_dir = self.root / "instruction_parts"
        part_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            {"instruction": "design drought protein", "input": "", "output": "MVLSPADKTN"},
            {"instruction": "design salt protein", "input": "", "output": "GKAHAGEYGM"},
        ]
        for index, row in enumerate(rows, start=1):
            (part_dir / f"instruction_part_{index}.jsonl").write_text(
                json.dumps(row) + "\n",
                encoding="utf-8",
            )

        config = InstructionTrainingConfig(
            instruction_jsonl=tuple(sorted(part_dir.glob("instruction_part_*.jsonl"))),
            artifact_source_jsonl=self.root / "missing_instruction.jsonl",
            base_checkpoint_path=self.root / "unused.pt",
            output_dir=self.root / "checkpoints",
            artifact_dir=self.root / "artifacts",
            artifact_profile_sample_size=10,
            profile_vocab_size=64,
        )
        trainer = InstructionTrainer.__new__(InstructionTrainer)
        trainer.config = config
        trainer.artifact_dir = Path(config.artifact_dir)
        trainer.instruction_paths = tuple(Path(path).resolve() for path in config.instruction_jsonl)

        artifacts = trainer._load_or_build_artifacts()

        self.assertTrue((self.root / "artifacts" / "tokenizer_map.json").exists())
        self.assertEqual("protein", artifacts.sequence_type)
        self.assertEqual(2, artifacts.record_count)


if __name__ == "__main__":
    unittest.main()
