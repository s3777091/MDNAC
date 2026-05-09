from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

import torch

from libs.core import (
    IGNORE_INDEX,
    MDCProfileCompilerConfig,
    MDCProfileSequencePretrainArtifacts,
    MDCProfileSequencePretrainDataset,
    MDCProfileSequenceRecord,
    MicrobialDecoderCoreApp,
    build_mdc_tiny_config,
    load_mdc_profile_sequence_records_from_session_artifact,
    create_mdc_profile_sequence_pretrain_dataloader,
    run_mdc_causal_lm_batch_epoch,
    save_mdc_profile_sequence_pretrain_from_instruction_jsonl,
    save_mdc_profile_sequence_pretrain_from_preparation_sessions,
    save_mdc_profile_sequence_pretrain_artifacts,
)
from libs.data.entities import PreparationSessionArtifact


class MDCProfileSequencePretrainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("tests/artifacts/mdc-profile-pretrain")
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)

        self.records = [
            MDCProfileSequenceRecord(
                profile="drought tolerance",
                sequence="MVLSPADKTN",
                sequence_type="protein",
                metadata={"accession": "ACC001"},
            ),
            MDCProfileSequenceRecord(
                profile="salt stress response",
                sequence="GKAHAGEYGM",
                sequence_type="protein",
                metadata={"accession": "ACC002"},
            ),
        ]
        self.artifact = save_mdc_profile_sequence_pretrain_artifacts(
            self.records,
            self.root,
            sequence_type="protein",
            kmer_size=3,
            profile_vocab_size=64,
        )
        self.session_root = self.root / "session"
        self.session_root.mkdir(parents=True, exist_ok=True)
        (self.session_root / "accessions.txt").write_text("ACC001.1\nACC002.1\n", encoding="utf-8")
        (self.session_root / "raw_index.json").write_text(
            json.dumps(
                {
                    "ACC001": {
                        "source_name": "ena",
                        "description": "Nitrogen fixation candidate",
                        "organism": "Bacillus subtilis",
                        "normalized_sequence": "MVLSPADKTN",
                        "sequence": "MVLSPADKTN",
                        "included_in_current_dataset": True,
                        "metadata": {
                            "moltype": "protein",
                            "genome": "chromosome",
                            "sequence_type": "protein",
                        },
                    },
                    "ACC002": {
                        "source_name": "ncbi",
                        "description": "Plastic degrader candidate",
                        "organism": "Pseudomonas putida",
                        "normalized_sequence": "GKAHAGEYGM",
                        "sequence": "GKAHAGEYGM",
                        "included_in_current_dataset": True,
                        "metadata": {
                            "biomol": "protein",
                            "completeness": "complete",
                            "sequence_type": "protein",
                        },
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self.session_artifact = PreparationSessionArtifact(
            source_name="ena",
            dataset_name="test-session",
            storage_mode="local",
            session_location=str(self.session_root),
            manifest_path=str(self.session_root / "manifest.json"),
            train_txt_path=str(self.session_root / "train.txt"),
            tokenizer_map_path=None,
            processed_count=2,
            total_count=2,
            record_count=2,
            dropped_record_count=0,
            is_complete=True,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_writes_profile_aware_train_txt_and_tokenizer_map(self) -> None:
        train_text = Path(self.artifact.train_text_path).read_text(encoding="utf-8")

        self.assertEqual(
            (
                "<|profile|>drought tolerance<|sep|><|protein|>MVLSPADKTN<|endoftext|>\n"
                "<|profile|>salt stress response<|sep|><|protein|>GKAHAGEYGM<|endoftext|>\n"
            ),
            train_text,
        )
        self.assertTrue(Path(self.artifact.tokenizer_map_path).exists())
        self.assertGreaterEqual(self.artifact.profile_vocab_size, 3)
        self.assertGreaterEqual(self.artifact.sequence_vocab_size, 3)

    def test_loads_artifacts_and_round_trips_profile_and_sequence(self) -> None:
        artifacts = MDCProfileSequencePretrainArtifacts.from_directory(self.root)

        self.assertEqual(2, artifacts.record_count)
        self.assertEqual("protein", artifacts.sequence_type)

        encoded_example = artifacts.encode_record(artifacts.examples[0])
        self.assertEqual(
            "drought tolerance",
            artifacts.decode_profile(encoded_example.profile_input_ids, skip_special=True),
        )
        self.assertEqual(
            "MVLSPADKTN",
            artifacts.decode_sequence(encoded_example.sequence_input_ids),
        )

    def test_builds_masked_causal_lm_batches_for_mdc_core(self) -> None:
        artifacts = MDCProfileSequencePretrainArtifacts.from_directory(self.root)
        encoded_example = artifacts.encode_record(artifacts.examples[0])
        fused_batch = artifacts.build_fused_batch([encoded_example])
        causal_batch = artifacts.build_causal_lm_batch([encoded_example])
        sequence_start = int(fused_batch.sequence_spans[0, 0])

        self.assertEqual((1, fused_batch.token_ids.size(1) - 1), tuple(causal_batch.input_ids.shape))
        self.assertTrue(torch.all(causal_batch.labels[0, : sequence_start - 1] == IGNORE_INDEX))
        self.assertEqual(
            int(fused_batch.token_ids[0, sequence_start]),
            int(causal_batch.labels[0, sequence_start - 1]),
        )

        app = MicrobialDecoderCoreApp(
            model_config=build_mdc_tiny_config(
                vocab_size=artifacts.layout.vocab_size,
                context_length=32,
            ),
            layout=artifacts.layout,
        )
        logits = app.forward_causal_lm_batch(causal_batch)
        self.assertEqual(
            (1, causal_batch.input_ids.size(1), artifacts.layout.vocab_size),
            tuple(logits.shape),
        )

    def test_runs_one_tiny_profile_aware_pretrain_epoch(self) -> None:
        artifacts = MDCProfileSequencePretrainArtifacts.from_directory(self.root)
        dataset = MDCProfileSequencePretrainDataset.from_artifacts(artifacts)
        data_loader = create_mdc_profile_sequence_pretrain_dataloader(
            dataset,
            batch_size=2,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
        )

        app = MicrobialDecoderCoreApp(
            model_config=build_mdc_tiny_config(
                vocab_size=artifacts.layout.vocab_size,
                context_length=32,
            ),
            layout=artifacts.layout,
        )
        optimizer = torch.optim.AdamW(app.parameters(), lr=1e-3)

        loss = run_mdc_causal_lm_batch_epoch(
            app,
            data_loader,
            optimizer,
            device="cpu",
        )

        self.assertTrue(torch.isfinite(torch.tensor(loss)))

    def test_loads_profile_records_from_preparation_session(self) -> None:
        records = load_mdc_profile_sequence_records_from_session_artifact(
            self.session_artifact,
            profile_config=MDCProfileCompilerConfig(metadata_fields=("genome", "moltype", "completeness")),
        )

        self.assertEqual(2, len(records))
        self.assertEqual("MVLSPADKTN", records[0].sequence)
        self.assertIn("task conditional sequence generation", records[0].profile)
        self.assertIn("labels nitrogen fixation", records[0].profile)
        self.assertIn("label source keyword rules", records[0].profile)
        self.assertIn("keywords nitrogen fixation", records[0].profile)
        self.assertIn("description Nitrogen fixation candidate", records[0].profile)
        self.assertIn("organism Bacillus subtilis", records[0].profile)
        self.assertIn("source ena", records[0].profile)
        self.assertIn("genome chromosome", records[0].profile)
        self.assertEqual("ACC001", records[0].metadata["accession"])
        self.assertEqual(["nitrogen fixation"], records[0].metadata["derived_labels"])
        self.assertEqual(["nitrogen fixation"], records[0].metadata["derived_keywords"])
        self.assertEqual("keyword rules", records[0].metadata["derived_label_source"])

        self.assertIn("labels complete sequence, protein sequence", records[1].profile)
        self.assertIn("label source structural metadata", records[1].profile)
        self.assertEqual(
            ["complete sequence", "protein sequence"],
            records[1].metadata["derived_labels"],
        )
        self.assertEqual([], records[1].metadata["derived_keywords"])
        self.assertEqual("structural metadata", records[1].metadata["derived_label_source"])

    def test_saves_profile_aware_artifacts_from_preparation_sessions(self) -> None:
        output_dir = self.root / "compiled-from-session"
        artifact = save_mdc_profile_sequence_pretrain_from_preparation_sessions(
            [self.session_artifact],
            output_dir,
            kmer_size=3,
            profile_vocab_size=64,
            profile_config=MDCProfileCompilerConfig(metadata_fields=("genome", "moltype", "completeness")),
        )

        compiled = MDCProfileSequencePretrainArtifacts.from_directory(output_dir)
        self.assertEqual(2, compiled.record_count)
        self.assertEqual("protein", compiled.sequence_type)
        self.assertTrue(Path(artifact.train_text_path).exists())
        self.assertIn("<|profile|>", Path(artifact.train_text_path).read_text(encoding="utf-8"))

    def test_saves_profile_aware_pretrain_artifacts_from_instruction_jsonl(self) -> None:
        instruction_path = self.root / "instruction.jsonl"
        instruction_path.write_text(
            "\n".join(
                json.dumps(payload, ensure_ascii=False)
                for payload in (
                    {
                        "instruction": "labels nitrogen fixation; product nitrogenase helper",
                        "input": "",
                        "output": "MVLSPADKTN",
                        "accession": "NP_000001.1",
                        "metadata": {"dataset_group": "bacteria"},
                    },
                    {
                        "instruction": "labels plastic degradation",
                        "input": "organism Pseudomonas putida",
                        "output": "gkahageygm",
                        "accession": "NP_000002.1",
                    },
                )
            )
            + "\n",
            encoding="utf-8",
        )

        output_dir = self.root / "compiled-from-instruction"
        artifact = save_mdc_profile_sequence_pretrain_from_instruction_jsonl(
            instruction_path,
            output_dir,
            kmer_size=3,
            profile_vocab_size=64,
        )

        compiled = MDCProfileSequencePretrainArtifacts.from_directory(output_dir)
        train_text = Path(artifact.train_text_path).read_text(encoding="utf-8")

        self.assertEqual(2, artifact.record_count)
        self.assertEqual(2, compiled.record_count)
        self.assertIn("<|profile|>labels nitrogen fixation, product nitrogenase helper<|sep|>", train_text)
        self.assertIn("labels plastic degradation; input organism Pseudomonas putida", train_text)
        self.assertIn("<|protein|>GKAHAGEYGM<|endoftext|>", train_text)
        self.assertEqual("MVLSPADKTN", compiled.examples[0].sequence)

if __name__ == "__main__":
    unittest.main()
