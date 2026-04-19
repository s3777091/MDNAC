from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from libs.core.pretrain.file_concat import concatenate_text_files


class ConcatenateTextFilesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("tests/artifacts/concat-text-files")
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)

    def test_concatenates_line_files_without_dedupe_and_preserves_record_boundary(self) -> None:
        first_path = self.root / "instruction.a.jsonl"
        second_path = self.root / "instruction.b.jsonl"
        output_path = self.root / "instruction.merged.jsonl"
        first_path.write_text('{"id": 1}\n{"id": 2}', encoding="utf-8")
        second_path.write_text('{"id": 2}\n{"id": 3}\n', encoding="utf-8")

        summary = concatenate_text_files(
            [first_path, second_path],
            output_path=output_path,
            buffer_size=5,
        )

        self.assertEqual('{"id": 1}\n{"id": 2}\n{"id": 2}\n{"id": 3}\n', output_path.read_text(encoding="utf-8"))
        self.assertEqual(2, summary.source_count)
        self.assertEqual(1, summary.inserted_separator_newlines)
        self.assertEqual(summary.source_bytes + 1, summary.output_bytes)

    def test_raw_mode_uses_exact_byte_concatenation(self) -> None:
        first_path = self.root / "train.a.txt"
        second_path = self.root / "train.b.txt"
        output_path = self.root / "train.merged.txt"
        first_path.write_bytes(b"AAA")
        second_path.write_bytes(b"BBB\n")

        summary = concatenate_text_files(
            [first_path, second_path],
            output_path=output_path,
            ensure_line_boundary=False,
            buffer_size=2,
        )

        self.assertEqual(b"AAABBB\n", output_path.read_bytes())
        self.assertEqual(0, summary.inserted_separator_newlines)
        self.assertEqual(summary.source_bytes, summary.output_bytes)

    def test_refuses_to_write_output_over_an_input_file(self) -> None:
        first_path = self.root / "train.a.txt"
        second_path = self.root / "train.b.txt"
        first_path.write_text("AAA\n", encoding="utf-8")
        second_path.write_text("BBB\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "output_path must be different"):
            concatenate_text_files([first_path, second_path], output_path=first_path, overwrite=True)


if __name__ == "__main__":
    unittest.main()
