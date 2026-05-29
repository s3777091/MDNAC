from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from libs.core.structure import (
    annotate_instruction_jsonl_3di,
    annotate_s3_instruction_jsonl_3di,
    normalize_prostt5_aa_sequence,
)


class Fake3DiProvider:
    model_name = "fake-3di"

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def predict_3di_batch(self, sequences):
        normalized = tuple(sequences)
        self.calls.append(normalized)
        return tuple("a" * len(sequence) for sequence in normalized)


class FakeS3Paginator:
    def __init__(self, objects: dict[tuple[str, str], bytes]) -> None:
        self.objects = objects

    def paginate(self, *, Bucket: str, Prefix: str):
        contents = []
        for bucket, key in sorted(self.objects):
            if bucket == Bucket and key.startswith(Prefix):
                contents.append(
                    {
                        "Key": key,
                        "Size": len(self.objects[(bucket, key)]),
                        "ETag": "fake",
                    }
                )
        return [{"Contents": contents}]


class FakeS3Client:
    def __init__(self, objects: dict[tuple[str, str], str]) -> None:
        self.objects = {
            location: content.encode("utf-8")
            for location, content in objects.items()
        }
        self.downloads: list[str] = []
        self.uploads: list[str] = []

    def get_paginator(self, name: str) -> FakeS3Paginator:
        if name != "list_objects_v2":
            raise ValueError(name)
        return FakeS3Paginator(self.objects)

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        self.downloads.append(key)
        Path(filename).write_bytes(self.objects[(bucket, key)])

    def upload_file(self, filename: str, bucket: str, key: str, **kwargs) -> None:
        del kwargs
        self.uploads.append(key)
        self.objects[(bucket, key)] = Path(filename).read_bytes()

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, ContentType: str | None = None) -> None:
        del ContentType
        self.objects[(Bucket, Key)] = Body


def instruction_record(sequence: str, **extra) -> str:
    payload = {
        "instruction": "labels protein sequence",
        "input": "",
        "output": sequence,
        "accession": f"ACC_{sequence}",
        **extra,
    }
    return json.dumps(payload) + "\n"


class Instruction3DiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("tests/artifacts/instruction-3di")
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_normalizes_rare_amino_acids_for_prostt5(self) -> None:
        self.assertEqual("MAXXXX", normalize_prostt5_aa_sequence("MAUOZJ"))

    def test_annotates_jsonl_reuses_existing_3di_and_deduplicates_batch(self) -> None:
        input_path = self.root / "instruction.jsonl"
        output_path = self.root / "instruction.3di.jsonl"
        input_path.write_text(
            "".join(
                (
                    instruction_record("MPEPTIDE"),
                    instruction_record("GLYSERQ", **{"3Di": "bcdefgh"}),
                    instruction_record("MPEPTIDE"),
                )
            ),
            encoding="utf-8",
        )
        provider = Fake3DiProvider()

        summary = annotate_instruction_jsonl_3di(
            input_path,
            output_path,
            provider,
            batch_size=2,
            cache_path=self.root / "3di.sqlite",
        )

        payloads = [
            json.loads(line)
            for line in output_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(["aaaaaaaa", "bcdefgh", "aaaaaaaa"], [payload["3Di"] for payload in payloads])
        self.assertEqual(3, summary.total_line_count)
        self.assertEqual(2, summary.new_3di_count)
        self.assertEqual(1, summary.reused_existing_count)
        self.assertEqual(1, summary.model_prediction_count)
        self.assertEqual([("MPEPTIDE",)], provider.calls)

    def test_annotates_jsonl_from_cache_without_calling_provider_again(self) -> None:
        cache_path = self.root / "3di.sqlite"
        first_input = self.root / "first.jsonl"
        first_output = self.root / "first.3di.jsonl"
        first_input.write_text(instruction_record("MPEPTIDE"), encoding="utf-8")
        provider = Fake3DiProvider()

        annotate_instruction_jsonl_3di(
            first_input,
            first_output,
            provider,
            batch_size=1,
            cache_path=cache_path,
        )

        second_input = self.root / "second.jsonl"
        second_output = self.root / "second.3di.jsonl"
        second_input.write_text(instruction_record("MPEPTIDE"), encoding="utf-8")
        summary = annotate_instruction_jsonl_3di(
            second_input,
            second_output,
            provider,
            batch_size=1,
            cache_path=cache_path,
        )

        self.assertEqual(1, summary.cache_hit_count)
        self.assertEqual(0, summary.model_prediction_count)
        self.assertEqual([("MPEPTIDE",)], provider.calls)

    def test_streams_s3_parts_and_uploads_annotated_parts_with_manifest(self) -> None:
        client = FakeS3Client(
            {
                ("bucket", "data/instruction/parts/instruction_part_1.jsonl"): (
                    instruction_record("MPEPTIDE")
                ),
                ("bucket", "data/instruction/parts/instruction_part_2.jsonl"): (
                    instruction_record("GLYSERQ", **{"3Di": "bcdefgh"})
                ),
                ("bucket", "data/instruction/parts/manifest.json"): "{}\n",
            }
        )
        provider = Fake3DiProvider()

        summary = annotate_s3_instruction_jsonl_3di(
            provider=provider,
            prefix_uri="s3://bucket/data/instruction/parts",
            output_prefix_uri="s3://bucket/data/instruction/parts_3di",
            s3_client=client,
            cache_path=self.root / "3di.sqlite",
            cache_dir=self.root / "s3-cache",
            batch_size=1,
            overwrite=True,
        )

        uploaded_first = client.objects[("bucket", "data/instruction/parts_3di/instruction_part_1.jsonl")]
        uploaded_second = client.objects[("bucket", "data/instruction/parts_3di/instruction_part_2.jsonl")]
        manifest = json.loads(client.objects[("bucket", "data/instruction/parts_3di/manifest.3di.json")])

        self.assertIn('"3Di":"aaaaaaaa"', uploaded_first.decode("utf-8"))
        self.assertIn('"3Di":"bcdefgh"', uploaded_second.decode("utf-8"))
        self.assertEqual(2, summary.part_count)
        self.assertEqual(2, summary.total_line_count)
        self.assertEqual(1, summary.new_3di_count)
        self.assertEqual(1, summary.reused_existing_count)
        self.assertEqual(1, summary.model_prediction_count)
        self.assertEqual(
            [
                "data/instruction/parts/instruction_part_1.jsonl",
                "data/instruction/parts/instruction_part_2.jsonl",
            ],
            client.downloads,
        )
        self.assertEqual(2, manifest["totals"]["total_line_count"])


if __name__ == "__main__":
    unittest.main()
