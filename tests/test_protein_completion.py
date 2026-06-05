from __future__ import annotations

import json
import random
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from libs.protein_completion import (
    convert_instruction_jsonl_to_span_jsonl,
    convert_instruction_row_to_span_examples,
    validate_jsonl_file,
    validate_span_completion_row,
)
from libs.protein_completion.masking import STANDARD_AMINO_ACIDS


class ProteinCompletionConversionTests(unittest.TestCase):
    def test_converts_full_sequence_row_to_valid_span_examples(self) -> None:
        source_row = {
            "instruction": "labels nitrogen fixation; organism Bacillus subtilis",
            "input": "",
            "output": _random_protein_sequence(180, seed=7),
            "accession": "NP_000001.1",
            "metadata": {"dataset_group": "bacteria"},
            "output_format": "single protein sequence",
        }
        source_before = deepcopy(source_row)

        examples = convert_instruction_row_to_span_examples(
            source_row,
            source_index=0,
            examples_per_sequence=3,
            min_sequence_length=64,
            max_sequence_length=1024,
            min_mask_length=8,
            max_mask_length=16,
            left_flank_size=24,
            right_flank_size=24,
            rng=random.Random(42),
        )

        self.assertEqual(source_before, source_row)
        self.assertEqual(3, len(examples))
        for row in examples:
            validation = validate_span_completion_row(row, original_sequence=source_row["output"])
            mask_start = validation["mask_start"]
            mask_end = validation["mask_end"]
            self.assertEqual(source_row["output"][mask_start:mask_end], row["output"])
            self.assertIn(f"<MASK_{len(row['output'])}>", row["input"])
            self.assertEqual("protein missing span", row["output_format"])
            self.assertEqual("bacteria", row["metadata"]["dataset_group"])
            self.assertEqual("single protein sequence", row["metadata"]["source_output_format"])

    def test_converts_instruction_jsonl_and_writes_stats(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = root / "instruction.jsonl"
            output_path = root / "span" / "instruction.jsonl"
            stats_path = root / "span" / "stats.json"
            source_path.write_text(
                json.dumps(
                    {
                        "instruction": "labels oxidoreductase",
                        "input": "",
                        "output": _random_protein_sequence(128, seed=11),
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            stats = convert_instruction_jsonl_to_span_jsonl(
                source_path,
                output_path,
                stats_path=stats_path,
                examples_per_sequence=2,
                min_mask_length=8,
                max_mask_length=12,
                left_flank_size=16,
                right_flank_size=16,
                seed=42,
            )
            validation = validate_jsonl_file(output_path)
            stats_payload = json.loads(stats_path.read_text(encoding="utf-8"))

            self.assertEqual(1, stats["source_rows"])
            self.assertEqual(1, stats["accepted_source_rows"])
            self.assertEqual(2, stats["generated_span_rows"])
            self.assertEqual(2, validation["valid_rows"])
            self.assertEqual(2, stats_payload["generated_span_rows"])
            self.assertIsNotNone(stats_payload["example_before"])
            self.assertIsNotNone(stats_payload["example_after"])


def _random_protein_sequence(length: int, *, seed: int) -> str:
    rng = random.Random(seed)
    alphabet = sorted(STANDARD_AMINO_ACIDS)
    return "".join(rng.choice(alphabet) for _ in range(length))


if __name__ == "__main__":
    unittest.main()
