from __future__ import annotations

import gzip
import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from libs.core.pretrain import refseq_local
from libs.core.pretrain.refseq_local import build_local_refseq_profile_text_artifacts
from libs.data.training import SequenceTokenizer
from libs.data.utilities.refseq_history import load_refseq_history


AMINO_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def build_sequence(index: int, length: int = 14) -> str:
    return "M" + "".join(AMINO_ALPHABET[(index + offset) % len(AMINO_ALPHABET)] for offset in range(length - 1))


def build_gpff_record(
    accession_index: int,
    *,
    product: str,
    description: str,
    sequence: str,
    note: str = "",
    organism: str = "Testus organismus",
) -> str:
    accession = f"NP_{accession_index:06d}"
    note_line = f'                     /note="{note}"\n' if note else ""
    return (
        f"LOCUS       {accession:<24}{len(sequence):>3} aa            linear   BCT 01-JAN-2026\n"
        f"DEFINITION  {description} [{organism}].\n"
        f"ACCESSION   {accession}\n"
        f"VERSION     {accession}.1\n"
        f"DBSOURCE    REFSEQ: accession NM_{accession_index:06d}.1\n"
        "KEYWORDS    RefSeq; Nitrogen fixation.\n"
        f"SOURCE      {organism}\n"
        f"  ORGANISM  {organism}\n"
        "            Bacteria; Testota.\n"
        "FEATURES             Location/Qualifiers\n"
        f"     source          1..{len(sequence)}\n"
        f'                     /organism="{organism}"\n'
        f"     Protein         1..{len(sequence)}\n"
        f'                     /product="{product}"\n'
        f"{note_line}"
        f"     CDS             1..{len(sequence)}\n"
        f'                     /gene="nif{accession_index}"\n'
        f'                     /coded_by="NM_{accession_index:06d}.1:1..{len(sequence) * 3}"\n'
        "ORIGIN\n"
        f"        1 {sequence.lower()}\n"
        "//\n"
    )


def build_faa_record(
    accession_index: int,
    *,
    description: str,
    sequence: str,
    organism: str = "Testus organismus",
) -> str:
    accession = f"NP_{accession_index:06d}.1"
    return f">{accession} {description} [{organism}]\n{sequence}\n"


class RefseqProfileTextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("tests/artifacts/refseq-profile-text")
        shutil.rmtree(self.root, ignore_errors=True)
        self.input_root = self.root / "input"
        self.output_dir = self.root / "output"
        self.group_dir = self.input_root / "test-group"
        self.group_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_builds_sequence_only_train_text_tokenizer_map_and_instruction_jsonl(self) -> None:
        self._write_refseq_bundle(record_count=11, updated_accessions={})

        summary = build_local_refseq_profile_text_artifacts(
            self.input_root,
            self.output_dir,
            vocab_size=64,
            instruction_min_proteins=10,
        )

        self.assertEqual(11, summary.source_record_count)
        self.assertEqual(11, summary.record_count)
        self.assertEqual(11, summary.instruction_record_count)
        self.assertEqual(1, summary.instruction_condition_count)
        self.assertEqual(0, summary.skipped_instruction_condition_count)
        self.assertEqual(11, summary.new_source_record_count)

        train_lines = Path(summary.train_text_path).read_text(encoding="utf-8").splitlines()
        self.assertEqual(11, len(train_lines))
        self.assertTrue(all(line.startswith("<|protein|>") for line in train_lines))
        self.assertTrue(all(line.endswith("<|endoftext|>") for line in train_lines))
        self.assertTrue(all("<|profile|>" not in line for line in train_lines))

        tokenizer = SequenceTokenizer.load_map(summary.tokenizer_map_path)
        train_text = Path(summary.train_text_path).read_text(encoding="utf-8")
        self.assertEqual(train_text, tokenizer.decode(tokenizer.encode(train_text)))

        instruction_lines = Path(summary.instruction_path).read_text(encoding="utf-8").splitlines()
        self.assertEqual(11, len(instruction_lines))
        instruction_payload = json.loads(instruction_lines[0])
        self.assertEqual("NP_000001.1", instruction_payload["accession"])
        self.assertEqual("", instruction_payload["input"])
        self.assertEqual(build_sequence(1), instruction_payload["output"])
        self.assertEqual("paired", instruction_payload["origin"])
        self.assertEqual("single protein sequence", instruction_payload["output_format"])
        self.assertIn("labels nitrogen fixation", instruction_payload["instruction"])
        self.assertIn("description nitrogen fixation protein 1", instruction_payload["instruction"])
        self.assertIn("organism Testus organismus", instruction_payload["instruction"])
        self.assertIn("gene nif1", instruction_payload["instruction"])
        self.assertIn("product nitrogenase helper 1", instruction_payload["instruction"])
        self.assertIn("note nitrogen fixation regulator", instruction_payload["instruction"])
        self.assertNotIn("task conditional sequence generation", instruction_payload["instruction"])
        self.assertEqual(["nitrogen fixation"], instruction_payload["derived_labels"])
        self.assertIn("nitrogen fixation", instruction_payload["derived_keywords"])
        self.assertEqual("test-group/bundle.1.protein", instruction_payload["metadata"]["dataset_bundle"])
        self.assertTrue(Path(summary.history_path).exists())
        self.assertEqual(0, summary.deleted_archive_count)
        self.assertTrue((self.group_dir / "bundle.1.protein.gpff.gz").exists())
        self.assertTrue((self.group_dir / "bundle.1.protein.faa.gz").exists())
        self.assertFalse((self.output_dir / "source_index.json").exists())

        summary_payload = json.loads(Path(summary.summary_path).read_text(encoding="utf-8"))
        self.assertNotIn("source_index_path", summary_payload)
        self.assertEqual(str(self.input_root), summary_payload["requested_input_root"])
        self.assertEqual(1, len(summary_payload["processed_bundles"]))
        self.assertEqual("test-group/bundle.1.protein", summary_payload["processed_bundles"][0]["bundle_key"])
        self.assertEqual(2, len(summary_payload["input_files"]))

        history = load_refseq_history(summary.history_path, input_root=self.input_root)
        gpff_entry = history["archives"]["test-group/bundle.1.protein.gpff.gz"]
        faa_entry = history["archives"]["test-group/bundle.1.protein.faa.gz"]
        self.assertEqual("compiled", gpff_entry["build_status"])
        self.assertEqual("compiled", faa_entry["build_status"])
        self.assertTrue(gpff_entry["present_on_disk"])
        self.assertTrue(faa_entry["present_on_disk"])

    def test_rebuild_reuses_existing_artifacts_when_archives_are_unchanged(self) -> None:
        self._write_refseq_bundle(record_count=3, updated_accessions={})

        first_summary = build_local_refseq_profile_text_artifacts(
            self.input_root,
            self.output_dir,
            vocab_size=64,
            instruction_min_proteins=1,
        )
        (self.output_dir / "source_index.json").write_text("{\"legacy\":true}\n", encoding="utf-8")
        second_summary = build_local_refseq_profile_text_artifacts(
            self.input_root,
            self.output_dir,
            vocab_size=64,
            instruction_min_proteins=1,
        )

        self.assertTrue((self.group_dir / "bundle.1.protein.gpff.gz").exists())
        self.assertTrue((self.group_dir / "bundle.1.protein.faa.gz").exists())
        self.assertFalse(first_summary.reused_existing_artifacts)
        self.assertTrue(second_summary.reused_existing_artifacts)
        self.assertEqual(0, second_summary.processed_archive_count)
        self.assertEqual(0, second_summary.deleted_archive_count)
        self.assertEqual(3, second_summary.record_count)
        self.assertEqual(0, second_summary.new_source_record_count)
        self.assertEqual(0, second_summary.updated_source_record_count)
        self.assertEqual(3, second_summary.unchanged_source_record_count)
        self.assertEqual(0, second_summary.removed_source_record_count)
        self.assertFalse((self.output_dir / "source_index.json").exists())

    def test_rebuild_detects_new_and_updated_records_without_duplicate_sequences(self) -> None:
        self._write_refseq_bundle(record_count=10, updated_accessions={})
        first_summary = build_local_refseq_profile_text_artifacts(
            self.input_root,
            self.output_dir,
            vocab_size=64,
            instruction_min_proteins=10,
        )

        original_sequence = build_sequence(5)
        updated_sequence = build_sequence(18)
        self._write_refseq_bundle(
            record_count=11,
            updated_accessions={
                5: {
                    "sequence": updated_sequence,
                    "description": "updated nitrogen fixation protein 5",
                    "product": "updated nitrogenase helper 5",
                    "note": "updated nitrogen fixation regulator",
                }
            },
        )
        second_summary = build_local_refseq_profile_text_artifacts(
            self.input_root,
            self.output_dir,
            vocab_size=64,
            instruction_min_proteins=10,
        )

        self.assertEqual(10, first_summary.record_count)
        self.assertEqual(11, second_summary.source_record_count)
        self.assertEqual(11, second_summary.record_count)
        self.assertEqual(1, second_summary.new_source_record_count)
        self.assertEqual(1, second_summary.updated_source_record_count)
        self.assertEqual(9, second_summary.unchanged_source_record_count)
        self.assertEqual(0, second_summary.removed_source_record_count)
        self.assertEqual(0, second_summary.duplicate_sequence_count)

        train_lines = Path(second_summary.train_text_path).read_text(encoding="utf-8").splitlines()
        self.assertEqual(11, len(train_lines))
        self.assertEqual(11, len(set(train_lines)))
        self.assertIn(f"<|protein|>{updated_sequence}<|endoftext|>", train_lines)
        self.assertNotIn(f"<|protein|>{original_sequence}<|endoftext|>", train_lines)
        self.assertFalse((self.output_dir / "source_index.json").exists())

    def test_skip_train_requires_existing_train_txt_for_tokenizer_map(self) -> None:
        self._write_refseq_bundle(record_count=3, updated_accessions={})

        with self.assertRaisesRegex(FileNotFoundError, "train.txt"):
            build_local_refseq_profile_text_artifacts(
                self.input_root,
                self.output_dir,
                vocab_size=64,
                instruction_min_proteins=1,
                skip_artifacts={"train"},
            )

        self.assertTrue((self.group_dir / "bundle.1.protein.gpff.gz").exists())
        self.assertTrue((self.group_dir / "bundle.1.protein.faa.gz").exists())

    def test_skip_train_and_instruction_rebuilds_only_tokenizer_map(self) -> None:
        self._write_refseq_bundle(record_count=3, updated_accessions={})
        initial_summary = build_local_refseq_profile_text_artifacts(
            self.input_root,
            self.output_dir,
            vocab_size=64,
            instruction_min_proteins=1,
        )
        original_train_text = Path(initial_summary.train_text_path).read_text(encoding="utf-8")
        original_instruction_text = Path(initial_summary.instruction_path).read_text(encoding="utf-8")
        Path(initial_summary.tokenizer_map_path).unlink()

        second_summary = build_local_refseq_profile_text_artifacts(
            self.input_root,
            self.output_dir,
            vocab_size=64,
            instruction_min_proteins=1,
            skip_artifacts={"train", "instruction.jsonl"},
        )

        self.assertFalse(second_summary.reused_existing_artifacts)
        self.assertEqual(0, second_summary.processed_archive_count)
        self.assertEqual(original_train_text, Path(second_summary.train_text_path).read_text(encoding="utf-8"))
        self.assertEqual(
            original_instruction_text,
            Path(second_summary.instruction_path).read_text(encoding="utf-8"),
        )
        tokenizer = SequenceTokenizer.load_map(second_summary.tokenizer_map_path)
        self.assertEqual(original_train_text, tokenizer.decode(tokenizer.encode(original_train_text)))

    def test_skip_instruction_keeps_incremental_state_for_later_rebuild(self) -> None:
        self._write_refseq_bundle(record_count=10, updated_accessions={})
        initial_summary = build_local_refseq_profile_text_artifacts(
            self.input_root,
            self.output_dir,
            vocab_size=64,
            instruction_min_proteins=10,
        )
        original_instruction_text = Path(initial_summary.instruction_path).read_text(encoding="utf-8")

        self._write_refseq_bundle(record_count=11, updated_accessions={})
        skipped_instruction_summary = build_local_refseq_profile_text_artifacts(
            self.input_root,
            self.output_dir,
            vocab_size=64,
            instruction_min_proteins=10,
            skip_artifacts={"instruction"},
        )

        self.assertEqual(11, skipped_instruction_summary.record_count)
        self.assertEqual(
            11,
            len(Path(skipped_instruction_summary.train_text_path).read_text(encoding="utf-8").splitlines()),
        )
        self.assertEqual(
            original_instruction_text,
            Path(skipped_instruction_summary.instruction_path).read_text(encoding="utf-8"),
        )

        rebuilt_summary = build_local_refseq_profile_text_artifacts(
            self.input_root,
            self.output_dir,
            vocab_size=64,
            instruction_min_proteins=10,
        )

        self.assertFalse(rebuilt_summary.reused_existing_artifacts)
        self.assertEqual(2, rebuilt_summary.processed_archive_count)
        self.assertEqual(
            11,
            len(Path(rebuilt_summary.instruction_path).read_text(encoding="utf-8").splitlines()),
        )
        rebuilt_summary_payload = json.loads(Path(rebuilt_summary.summary_path).read_text(encoding="utf-8"))
        self.assertEqual(
            11,
            sum(item["record_count"] for item in rebuilt_summary_payload["input_files"] if item["kind"] == "gpff"),
        )

    def test_max_records_only_marks_fully_processed_bundle(self) -> None:
        self._write_refseq_bundle(record_count=3, updated_accessions={}, bundle_name="bundle.1")
        self._write_refseq_bundle(record_count=3, updated_accessions={}, bundle_name="bundle.2", start_index=100)

        summary = build_local_refseq_profile_text_artifacts(
            self.input_root,
            self.output_dir,
            vocab_size=64,
            instruction_min_proteins=1,
            max_records=3,
        )

        self.assertEqual(2, summary.processed_archive_count)
        self.assertEqual(0, summary.deleted_archive_count)
        self.assertTrue((self.group_dir / "bundle.1.protein.gpff.gz").exists())
        self.assertTrue((self.group_dir / "bundle.1.protein.faa.gz").exists())
        self.assertTrue((self.group_dir / "bundle.2.protein.gpff.gz").exists())
        self.assertTrue((self.group_dir / "bundle.2.protein.faa.gz").exists())

        history = load_refseq_history(summary.history_path, input_root=self.input_root)
        first_gpff = history["archives"]["test-group/bundle.1.protein.gpff.gz"]
        second_gpff = history["archives"]["test-group/bundle.2.protein.gpff.gz"]
        self.assertEqual("compiled", first_gpff["build_status"])
        self.assertEqual("pending", second_gpff["build_status"])
        self.assertTrue(first_gpff["present_on_disk"])
        self.assertTrue(second_gpff["present_on_disk"])

    def test_output_subfolder_scopes_build_to_matching_input_folder(self) -> None:
        bacteria_dir = self.input_root / "bacteria"
        fungi_dir = self.input_root / "fungi"
        bacteria_dir.mkdir(parents=True, exist_ok=True)
        fungi_dir.mkdir(parents=True, exist_ok=True)
        bacteria_sequence = "MCCCCCCCCCCCCC"
        fungi_sequences = {
            100: {"sequence": "MWWWWWWWWWWWWW"},
            101: {"sequence": "MYYYYYYYYYYYYY"},
            102: {"sequence": "MVVVVVVVVVVVVV"},
        }
        self._write_refseq_bundle(
            record_count=2,
            updated_accessions={
                1: {"sequence": bacteria_sequence},
                2: {"sequence": "MDDDDDDDDDDDDD"},
            },
            bundle_name="bacteria.1",
            group_dir=bacteria_dir,
        )
        self._write_refseq_bundle(
            record_count=3,
            updated_accessions=fungi_sequences,
            bundle_name="fungi.1",
            start_index=100,
            group_dir=fungi_dir,
        )

        fungi_output_dir = self.output_dir / "fungi"
        summary = build_local_refseq_profile_text_artifacts(
            self.input_root,
            fungi_output_dir,
            vocab_size=64,
            instruction_min_proteins=1,
        )

        self.assertEqual(str(fungi_dir), summary.input_root)
        self.assertEqual(3, summary.record_count)
        train_lines = Path(summary.train_text_path).read_text(encoding="utf-8").splitlines()
        self.assertEqual(3, len(train_lines))
        self.assertIn("<|protein|>MWWWWWWWWWWWWW<|endoftext|>", train_lines)
        self.assertNotIn(f"<|protein|>{bacteria_sequence}<|endoftext|>", train_lines)

        summary_payload = json.loads(Path(summary.summary_path).read_text(encoding="utf-8"))
        self.assertEqual(str(self.input_root), summary_payload["requested_input_root"])
        self.assertTrue(all("fungi.1.protein" in item["bundle_key"] for item in summary_payload["input_files"]))

    def test_partial_input_appends_into_existing_compiled_root_without_dropping_other_packages(self) -> None:
        package_one_dir = self.input_root / "package_1"
        package_two_dir = self.input_root / "package_2"
        package_one_dir.mkdir(parents=True, exist_ok=True)
        package_two_dir.mkdir(parents=True, exist_ok=True)

        package_two_sequence = "MWWWWWWWWWWWWW"
        package_two_second_sequence = "MYYYYYYYYYYYYY"
        self._write_refseq_bundle(
            record_count=2,
            updated_accessions={},
            bundle_name="bundle.1",
            group_dir=package_one_dir,
        )
        self._write_refseq_bundle(
            record_count=2,
            updated_accessions={
                100: {"sequence": package_two_sequence},
                101: {"sequence": package_two_second_sequence},
            },
            bundle_name="bundle.1",
            start_index=100,
            group_dir=package_two_dir,
        )

        full_summary = build_local_refseq_profile_text_artifacts(
            self.input_root,
            self.output_dir,
            vocab_size=64,
            instruction_min_proteins=1,
        )
        self.assertEqual(4, full_summary.record_count)

        self._write_refseq_bundle(
            record_count=3,
            updated_accessions={},
            bundle_name="bundle.1",
            group_dir=package_one_dir,
        )

        partial_summary = build_local_refseq_profile_text_artifacts(
            package_one_dir,
            self.output_dir,
            vocab_size=64,
            instruction_min_proteins=1,
        )

        self.assertEqual(5, partial_summary.source_record_count)
        self.assertEqual(5, partial_summary.record_count)
        self.assertEqual(1, partial_summary.new_source_record_count)
        self.assertEqual(0, partial_summary.updated_source_record_count)
        self.assertEqual(4, partial_summary.unchanged_source_record_count)

        train_lines = Path(partial_summary.train_text_path).read_text(encoding="utf-8").splitlines()
        self.assertEqual(5, len(train_lines))
        self.assertIn(f"<|protein|>{package_two_sequence}<|endoftext|>", train_lines)
        self.assertIn(f"<|protein|>{build_sequence(3)}<|endoftext|>", train_lines)

        instruction_lines = Path(partial_summary.instruction_path).read_text(encoding="utf-8").splitlines()
        self.assertEqual(5, len(instruction_lines))
        instruction_accessions = {json.loads(line)["accession"] for line in instruction_lines}
        self.assertIn("NP_000003.1", instruction_accessions)
        self.assertIn("NP_000100.1", instruction_accessions)

        summary_payload = json.loads(Path(partial_summary.summary_path).read_text(encoding="utf-8"))
        processed_bundle_keys = {item["bundle_key"] for item in summary_payload["processed_bundles"]}
        self.assertIn("package_1/bundle.1.protein", processed_bundle_keys)
        self.assertIn("package_2/bundle.1.protein", processed_bundle_keys)

    def test_parallel_workers_match_serial_outputs(self) -> None:
        self._write_refseq_bundle(record_count=11, updated_accessions={})
        serial_output_dir = self.root / "serial-output"
        parallel_output_dir = self.root / "parallel-output"

        serial_summary = build_local_refseq_profile_text_artifacts(
            self.input_root,
            serial_output_dir,
            vocab_size=64,
            instruction_min_proteins=10,
            workers=1,
        )

        self._write_refseq_bundle(record_count=11, updated_accessions={})
        with patch.object(refseq_local, "PARALLEL_MIN_RECORDS", 1):
            parallel_summary = build_local_refseq_profile_text_artifacts(
                self.input_root,
                parallel_output_dir,
                vocab_size=64,
                instruction_min_proteins=10,
                workers=2,
            )

        self.assertEqual(
            Path(serial_summary.train_text_path).read_text(encoding="utf-8"),
            Path(parallel_summary.train_text_path).read_text(encoding="utf-8"),
        )
        self.assertEqual(
            Path(serial_summary.instruction_path).read_text(encoding="utf-8"),
            Path(parallel_summary.instruction_path).read_text(encoding="utf-8"),
        )
        self.assertEqual(
            json.loads(Path(serial_summary.tokenizer_map_path).read_text(encoding="utf-8")),
            json.loads(Path(parallel_summary.tokenizer_map_path).read_text(encoding="utf-8")),
        )

    def test_windows_parallel_path_uses_thread_pool(self) -> None:
        with patch.object(refseq_local.os, "name", "nt"):
            self.assertIs(refseq_local.ThreadPoolExecutor, refseq_local._executor_class_for_parallelism())

    def _write_refseq_bundle(
        self,
        *,
        record_count: int,
        updated_accessions: dict[int, dict[str, str]],
        bundle_name: str = "bundle.1",
        start_index: int = 1,
        group_dir: Path | None = None,
    ) -> None:
        gpff_parts: list[str] = []
        faa_parts: list[str] = []
        for accession_index in range(start_index, start_index + record_count):
            overrides = updated_accessions.get(accession_index, {})
            sequence = overrides.get("sequence", build_sequence(accession_index))
            description = overrides.get("description", f"nitrogen fixation protein {accession_index}")
            product = overrides.get("product", f"nitrogenase helper {accession_index}")
            note = overrides.get("note", "nitrogen fixation regulator")
            gpff_parts.append(
                build_gpff_record(
                    accession_index,
                    product=product,
                    description=description,
                    note=note,
                    sequence=sequence,
                )
            )
            faa_parts.append(
                build_faa_record(
                    accession_index,
                    description=description,
                    sequence=sequence,
                )
            )

        target_group_dir = group_dir or self.group_dir
        target_group_dir.mkdir(parents=True, exist_ok=True)
        self._write_gzip_text(target_group_dir / f"{bundle_name}.protein.gpff.gz", "".join(gpff_parts))
        self._write_gzip_text(target_group_dir / f"{bundle_name}.protein.faa.gz", "".join(faa_parts))

    def _write_gzip_text(self, path: Path, text: str) -> None:
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(text)


if __name__ == "__main__":
    unittest.main()
