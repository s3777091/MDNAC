from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from libs.core import (
    IGNORE_INDEX,
    MDCProfileSequencePretrainArtifacts,
    MDCProfileSequenceRecord,
    build_or_load_protein_tokenizer_from_text_paths,
    build_or_load_protein_tokenizer,
    create_streaming_mdc_profile_sequence_pretrain_dataloader,
    create_streaming_protein_lm_dataloader,
    discover_protein_train_text_paths,
    load_protein_corpus_text_parts,
    save_mdc_profile_sequence_pretrain_artifacts,
)
from libs.data.training.streaming import list_minio_text_parts


class FakeS3Paginator:
    def __init__(self, objects: dict[tuple[str, str], bytes]) -> None:
        self.objects = objects

    def paginate(self, *, Bucket: str, Prefix: str):
        contents = []
        for bucket, key in sorted(self.objects):
            if bucket == Bucket and key.startswith(Prefix):
                contents.append({"Key": key, "Size": len(self.objects[(bucket, key)])})
        return [{"Contents": contents}]


class FakeS3Client:
    def __init__(self, objects: dict[tuple[str, str], str]) -> None:
        self.objects = {
            location: content.encode("utf-8")
            for location, content in objects.items()
        }
        self.downloads: list[str] = []

    def get_paginator(self, name: str) -> FakeS3Paginator:
        if name != "list_objects_v2":
            raise ValueError(name)
        return FakeS3Paginator(self.objects)

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        self.downloads.append(key)
        Path(filename).write_bytes(self.objects[(bucket, key)])


class StreamingMinioPretrainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("tests/artifacts/streaming-minio-pretrain")
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_lists_training_parts_under_minio_prefix(self) -> None:
        client = FakeS3Client(
            {
                ("bucket", "datasets/protein/current/parts/train-0002.txt"): "b\n",
                ("bucket", "datasets/protein/current/parts/train_part_10.txt"): "c\n",
                ("bucket", "datasets/protein/current/parts/train_part_2.txt"): "b\n",
                ("bucket", "datasets/protein/current/tokenizer_map.json"): "{}\n",
                ("bucket", "datasets/protein/current/parts/train-0001.txt"): "a\n",
            }
        )

        parts = list_minio_text_parts(
            prefix_uri="s3://bucket/datasets/protein/current",
            s3_client=client,
            suffixes=(".txt",),
        )

        self.assertEqual(
            [
                "datasets/protein/current/parts/train-0001.txt",
                "datasets/protein/current/parts/train-0002.txt",
                "datasets/protein/current/parts/train_part_2.txt",
                "datasets/protein/current/parts/train_part_10.txt",
            ],
            [part.key for part in parts],
        )

    def test_streams_protein_lm_batches_by_downloading_each_minio_part(self) -> None:
        train_path = self.root / "train.txt"
        train_path.write_text(
            (
                "<|protein|>MPEPTIDE<|endoftext|>\n"
                "<|protein|>GLYSERQ<|endoftext|>\n"
                "<|protein|>MVLSPADKTN<|endoftext|>\n"
            ),
            encoding="utf-8",
        )
        tokenizer = build_or_load_protein_tokenizer(train_path, vocab_size=64).tokenizer
        client = FakeS3Client(
            {
                ("bucket", "datasets/protein/current/parts/train-0001.txt"): (
                    "<|protein|>MPEPTIDE<|endoftext|>\n"
                    "<|protein|>GLYSERQ<|endoftext|>\n"
                ),
                ("bucket", "datasets/protein/current/parts/train-0002.txt"): (
                    "<|protein|>MVLSPADKTN<|endoftext|>\n"
                ),
                ("bucket", "datasets/protein/current/tokenizer_map.json"): "{}\n",
            }
        )

        data_loader = create_streaming_protein_lm_dataloader(
            tokenizer,
            prefix_uri="s3://bucket/datasets/protein/current",
            s3_client=client,
            cache_dir=self.root / "protein-cache",
            context_length=12,
            stride=6,
            batch_size=2,
            pin_memory=False,
        )

        batches = list(data_loader)

        self.assertGreaterEqual(len(batches), 1)
        self.assertTrue(all(batch.input_ids.size(0) <= 2 for batch in batches))
        self.assertTrue(torch.any(batches[0].labels != IGNORE_INDEX))
        self.assertEqual(
            [
                "datasets/protein/current/parts/train-0001.txt",
                "datasets/protein/current/parts/train-0002.txt",
            ],
            client.downloads,
        )
        self.assertFalse(any((self.root / "protein-cache").iterdir()))

    def test_discovers_and_streams_local_train_part_files(self) -> None:
        corpus_dir = self.root / "local-parts"
        corpus_dir.mkdir(parents=True, exist_ok=True)
        (corpus_dir / "train_part_10.txt").write_text(
            "<|protein|>MVLSPADKTN<|endoftext|>\n",
            encoding="utf-8",
        )
        (corpus_dir / "train_part_1.txt").write_text(
            "<|protein|>MPEPTIDE<|endoftext|>\n",
            encoding="utf-8",
        )
        (corpus_dir / "train_part_2.txt").write_text(
            "<|protein|>GLYSERQ<|endoftext|>\n",
            encoding="utf-8",
        )

        part_paths = discover_protein_train_text_paths(corpus_dir / "train.txt")
        tokenizer_artifact = build_or_load_protein_tokenizer_from_text_paths(
            part_paths,
            tokenizer_map_path=corpus_dir / "tokenizer_map.json",
            vocab_size=64,
        )
        data_loader = create_streaming_protein_lm_dataloader(
            tokenizer_artifact.tokenizer,
            part_paths=part_paths,
            context_length=12,
            stride=6,
            batch_size=2,
            pin_memory=False,
        )

        batch = next(iter(data_loader))

        self.assertEqual(
            ["train_part_1.txt", "train_part_2.txt", "train_part_10.txt"],
            [path.name for path in part_paths],
        )
        self.assertIn("<|protein|>GLYSERQ<|endoftext|>", load_protein_corpus_text_parts(part_paths))
        self.assertEqual(2, batch.input_ids.size(0))

    def test_streaming_local_parts_do_not_read_entire_part_into_memory(self) -> None:
        corpus_dir = self.root / "line-streamed-parts"
        corpus_dir.mkdir(parents=True, exist_ok=True)
        part_path = corpus_dir / "train_part_1.txt"
        part_path.write_text(
            (
                "<|protein|>MPEPTIDE<|endoftext|>\n"
                "<|protein|>GLYSERQ<|endoftext|>\n"
            ),
            encoding="utf-8",
        )
        tokenizer = build_or_load_protein_tokenizer(part_path, vocab_size=64).tokenizer
        data_loader = create_streaming_protein_lm_dataloader(
            tokenizer,
            part_paths=(part_path,),
            context_length=12,
            stride=6,
            batch_size=1,
            pin_memory=False,
        )

        with patch.object(Path, "read_text", side_effect=AssertionError("read_text should not be used")):
            batch = next(iter(data_loader))

        self.assertEqual(1, batch.input_ids.size(0))
        self.assertTrue(torch.any(batch.labels != IGNORE_INDEX))

    def test_streams_profile_aware_batches_from_minio_parts(self) -> None:
        records = [
            MDCProfileSequenceRecord(profile="drought tolerance", sequence="MVLSPADKTN"),
            MDCProfileSequenceRecord(profile="salt stress response", sequence="GKAHAGEYGM"),
        ]
        artifact = save_mdc_profile_sequence_pretrain_artifacts(
            records,
            self.root / "profile",
            kmer_size=3,
            profile_vocab_size=64,
        )
        train_lines = Path(artifact.train_text_path).read_text(encoding="utf-8").splitlines()
        tokenizer_only_artifacts = MDCProfileSequencePretrainArtifacts.from_tokenizer_map_file(
            Path(artifact.tokenizer_map_path)
        )
        client = FakeS3Client(
            {
                ("bucket", "datasets/profile/current/parts/train-0001.txt"): train_lines[0] + "\n",
                ("bucket", "datasets/profile/current/parts/train-0002.txt"): train_lines[1] + "\n",
            }
        )

        data_loader = create_streaming_mdc_profile_sequence_pretrain_dataloader(
            tokenizer_only_artifacts,
            prefix_uri="s3://bucket/datasets/profile/current/parts",
            s3_client=client,
            cache_dir=self.root / "profile-cache",
            batch_size=2,
            pin_memory=False,
        )
        batch = next(iter(data_loader))

        self.assertEqual(2, tokenizer_only_artifacts.record_count)
        self.assertEqual(2, batch.input_ids.size(0))
        self.assertTrue(torch.any(batch.labels != IGNORE_INDEX))
        self.assertEqual(
            [
                "datasets/profile/current/parts/train-0001.txt",
                "datasets/profile/current/parts/train-0002.txt",
            ],
            client.downloads,
        )


if __name__ == "__main__":
    unittest.main()
