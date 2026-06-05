from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from libs.instruction.data import (
    count_instruction_split_records,
    count_instruction_split_records_by_split,
)
from libs.instruction.schema import instruction_record_from_payload
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

    def test_profile_tokenizer_encodes_instruction_characters_outside_artifact_sample(self) -> None:
        part_dir = self.root / "instruction_parts"
        part_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            {"instruction": "design drought protein", "input": "", "output": "MVLSPADKTN"},
            {
                "instruction": "labels protein; product alpha-(1->3)-arabinofuranosyltransferase",
                "input": "",
                "output": "MVLSPADKTN",
            },
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
            artifact_profile_sample_size=1,
            profile_vocab_size=64,
        )
        trainer = InstructionTrainer.__new__(InstructionTrainer)
        trainer.config = config
        trainer.artifact_dir = Path(config.artifact_dir)
        trainer.instruction_paths = tuple(Path(path).resolve() for path in config.instruction_jsonl)

        artifacts = trainer._load_or_build_artifacts()
        record = instruction_record_from_payload(rows[1])
        encoded = artifacts.encode_record(record)

        self.assertGreater(encoded.profile_input_ids.numel(), 0)

    def test_counts_train_and_val_splits_in_one_pass(self) -> None:
        instruction_path = self.root / "instruction.jsonl"
        rows = [
            {"instruction": "design drought protein", "input": "", "output": "MVLSPADKTN"},
            {"instruction": "design salt protein", "input": "", "output": "GKAHAGEYGM"},
            {"instruction": "design heat protein", "input": "", "output": "MTEYKLVVVG"},
            {"instruction": "design cold protein", "input": "", "output": "GAGGVGKSAL"},
        ]
        instruction_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        progress: list[tuple[int, int, int, int]] = []

        counts = count_instruction_split_records_by_split(
            instruction_path,
            train_ratio=0.5,
            split_seed=7,
            progress_every=2,
            progress_callback=lambda *values: progress.append(values),
        )

        expected_train = count_instruction_split_records(
            instruction_path,
            split="train",
            train_ratio=0.5,
            split_seed=7,
        )
        expected_val = count_instruction_split_records(
            instruction_path,
            split="val",
            train_ratio=0.5,
            split_seed=7,
        )
        self.assertEqual(expected_train, counts["train"])
        self.assertEqual(expected_val, counts["val"])
        self.assertEqual(len(rows), counts["rows_seen"])
        self.assertEqual(0, counts["skipped_for_length"])
        self.assertEqual(len(rows), progress[-1][0])

    def test_compact_profile_prompt_format_keeps_instruction_and_input_compact(self) -> None:
        row = {
            "instruction": "task protein span completion; labels oxidoreductase",
            "input": "mask_length 8; left_flank AAAAAAAA; right_flank CCCCCCCC",
            "output": "MTEYKLVV",
        }

        record = instruction_record_from_payload(row, prompt_format="compact_profile")

        self.assertEqual(
            "task protein span completion; labels oxidoreductase; input "
            "mask_length 8; left_flank AAAAAAAA; right_flank CCCCCCCC",
            record.profile,
        )
        self.assertEqual("compact_profile", record.metadata["instruction_prompt_format"])
        self.assertNotIn("### Instruction", record.profile)


if __name__ == "__main__":
    unittest.main()
