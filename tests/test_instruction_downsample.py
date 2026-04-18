from __future__ import annotations

import json
import shutil
import unittest
from collections import Counter
from pathlib import Path

from libs.core.pretrain.instruction_downsample import (
    SYSTEMATIC_PHASE_DENOMINATOR,
    _allocate_stratum_quotas,
    _instruction_stratum_key,
    _should_keep_occurrence,
    downsample_instruction_jsonl,
)


def build_instruction_line(accession: str, *, dataset_group: str, product: str, organism: str) -> str:
    payload = {
        "instruction": f"organism {organism}; product {product}",
        "input": "",
        "output": f"MSEQ{accession}",
        "accession": accession,
        "description": product,
        "organism": organism,
        "metadata": {
            "dataset_group": dataset_group,
            "product": product,
            "scientific_name": organism,
        },
        "derived_labels": ["protein sequence"],
        "derived_keywords": [],
        "label_source": "structural metadata",
        "origin": "paired",
        "output_format": "single protein sequence",
    }
    return json.dumps(payload) + "\n"


class InstructionDownsampleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("tests/artifacts/instruction-downsample")
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)

    def test_allocate_stratum_quotas_hits_exact_target(self) -> None:
        quotas = _allocate_stratum_quotas(
            {
                "hypothetical protein": 100,
                "dna polymerase": 20,
                "capsid protein": 4,
                "unique regulator": 1,
            },
            target_count=63,
            alpha=0.8,
        )

        self.assertEqual(63, sum(quotas.values()))
        self.assertEqual(1, quotas["unique regulator"])
        self.assertGreaterEqual(quotas["capsid protein"], 1)
        self.assertLess(quotas["hypothetical protein"], 50)

    def test_systematic_sampler_spreads_records_instead_of_taking_prefix(self) -> None:
        selected_indexes = [
            index
            for index in range(10)
            if _should_keep_occurrence(
                seen_count=index,
                total_count=10,
                keep_count=3,
                phase_numerator=SYSTEMATIC_PHASE_DENOMINATOR // 2,
            )
        ]

        self.assertEqual([1, 4, 8], selected_indexes)

    def test_downsample_instruction_jsonl_preserves_bucket_coverage(self) -> None:
        input_path = self.root / "instruction.jsonl"
        output_path = self.root / "instruction.downsampled.jsonl"
        input_path.write_text(
            "".join(
                [
                    build_instruction_line(f"BAC_A_{index}", dataset_group="bacteria", product="enzyme A", organism=f"Bacillus {index}")
                    for index in range(8)
                ]
                + [
                    build_instruction_line(f"BAC_B_{index}", dataset_group="bacteria", product="enzyme B", organism=f"Bacillus B {index}")
                    for index in range(4)
                ]
                + [
                    build_instruction_line(f"VIR_C_{index}", dataset_group="viral", product="capsid protein", organism=f"Virus C {index}")
                    for index in range(3)
                ]
                + [
                    build_instruction_line("VIR_H_0", dataset_group="viral", product="hypothetical protein", organism="Virus H 0"),
                ]
            ),
            encoding="utf-8",
        )

        summary = downsample_instruction_jsonl(
            input_path,
            output_path=output_path,
            keep_ratio=0.5,
            alpha=0.8,
        )

        kept_payloads = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        kept_strata = Counter(_instruction_stratum_key(payload) for payload in kept_payloads)
        group_counts = Counter(payload["metadata"]["dataset_group"] for payload in kept_payloads)

        self.assertEqual(16, summary.total_line_count)
        self.assertEqual(8, summary.target_line_count)
        self.assertEqual(8, summary.written_line_count)
        self.assertEqual(
            {
                ("bacteria", "enzyme a"),
                ("bacteria", "enzyme b"),
                ("viral", "capsid protein"),
                ("viral", "hypothetical protein"),
            },
            set(kept_strata),
        )
        self.assertEqual(6, group_counts["bacteria"])
        self.assertEqual(2, group_counts["viral"])


if __name__ == "__main__":
    unittest.main()
