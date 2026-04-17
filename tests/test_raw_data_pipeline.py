from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

import torch

from libs.data.training import KmerTokenizer, ProfileBPETokenizer, RawDataPipeline, audit_fasta_and_gff, audit_genbank
from libs.data.training.raw_pipeline.defaults import DEFAULT_KEYWORD_RULES


FASTA_TEXT = (
    ">chr1 Oryza sativa chromosome 1\n"
    "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG\n"
)

GFF_TEXT = (
    "##gff-version 3\n"
    "chr1\tPhytozome\tgene\t1\t18\t.\t+\t.\tID=gene1;Name=DRO1;Note=drought tolerance regulator\n"
    "chr1\tPhytozome\tCDS\t19\t33\t.\t-\t0\tID=cds1;Parent=gene2;product=photosystem II protein\n"
    "chr1\tPhytozome\tgene\t34\t39\t.\t+\t.\tID=gene3;Name=ACT1;Note=housekeeping gene\n"
)

GENBANK_TEXT = (
    "LOCUS       TEST0001                39 bp    DNA     linear   PLN 01-JAN-2026\n"
    "DEFINITION  Test plant locus.\n"
    "ACCESSION   TEST0001\n"
    "VERSION     TEST0001.1\n"
    "SOURCE      Oryza sativa\n"
    "  ORGANISM  Oryza sativa\n"
    "FEATURES             Location/Qualifiers\n"
    "     gene            1..18\n"
    "                     /gene=\"DRO1\"\n"
    "                     /note=\"drought tolerance regulator\"\n"
    "     CDS             complement(19..33)\n"
    "                     /product=\"photosystem II protein\"\n"
    "                     /note=\"photosynthesis helper\"\n"
    "ORIGIN\n"
    "        1 atggccattg taatgggccg ctgaaagggt gcccgatag\n"
    "//\n"
)

GENBANK_GO_AND_BOUNDARY_TEXT = (
    "LOCUS       TEST0002                39 bp    DNA     linear   PLN 01-JAN-2026\n"
    "DEFINITION  Test plant locus with GO qualifiers.\n"
    "ACCESSION   TEST0002\n"
    "VERSION     TEST0002.1\n"
    "SOURCE      Oryza sativa\n"
    "  ORGANISM  Oryza sativa\n"
    "FEATURES             Location/Qualifiers\n"
    "     gene            1..18\n"
    "                     /go_process=\"DNA repair|0006281||IEA\"\n"
    "                     /product=\"uncharacterized protein\"\n"
    "     CDS             complement(19..33)\n"
    "                     /product=\"chlorophyllase isoform\"\n"
    "                     /note=\"pigment turnover helper\"\n"
    "ORIGIN\n"
    "        1 atggccattg taatgggccg ctgaaagggt gcccgatag\n"
    "//\n"
)


class RawDataPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("tests/artifacts/raw-pipeline")
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)
        self.fasta_path = self.root / "oryza.fasta"
        self.gff_path = self.root / "oryza.gff3"
        self.genbank_path = self.root / "oryza.gbk"
        self.go_genbank_path = self.root / "oryza-go.gbk"
        self.fasta_path.write_text(FASTA_TEXT, encoding="utf-8")
        self.gff_path.write_text(GFF_TEXT, encoding="utf-8")
        self.genbank_path.write_text(GENBANK_TEXT, encoding="utf-8")
        self.go_genbank_path.write_text(GENBANK_GO_AND_BOUNDARY_TEXT, encoding="utf-8")
        self.pipeline = RawDataPipeline()

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_builds_profile_sequence_pairs_from_fasta_and_gff(self) -> None:
        pairs = self.pipeline.build_pairs_from_fasta_and_gff(
            fasta_path=self.fasta_path,
            annotation_path=self.gff_path,
            organism="Oryza sativa",
        )

        self.assertEqual(2, len(pairs))
        self.assertEqual("Drought tolerant gene in Oryza sativa", pairs[0].profile)
        self.assertEqual("ATGGCCATTGTAATGGGC", pairs[0].sequence)
        self.assertEqual("Photosynthesis gene in Oryza sativa", pairs[1].profile)
        self.assertEqual("GGCACCCTTTCAGCG", pairs[1].sequence)
        self.assertEqual("-", pairs[1].strand)

    def test_builds_profile_sequence_pairs_from_genbank(self) -> None:
        pairs = self.pipeline.build_pairs_from_genbank(self.genbank_path)

        self.assertEqual(2, len(pairs))
        self.assertEqual("Oryza sativa", pairs[0].organism)
        self.assertEqual("TEST0001", pairs[0].accession)
        self.assertEqual("Drought tolerant gene in Oryza sativa", pairs[0].profile)
        self.assertEqual("Photosynthesis gene in Oryza sativa", pairs[1].profile)
        self.assertEqual("GGCACCCTTTCAGCG", pairs[1].sequence)

    def test_prepare_from_fasta_and_gff_exports_pt_dataset_and_maps(self) -> None:
        output_dir = self.root / "tensor-dataset"

        artifact = self.pipeline.prepare_from_fasta_and_gff(
            dataset_name="oryza-stress",
            fasta_path=self.fasta_path,
            annotation_path=self.gff_path,
            organism="Oryza sativa",
            output_dir=output_dir,
            kmer_size=3,
            profile_vocab_size=96,
        )

        self.assertEqual(2, artifact.pair_count)
        self.assertTrue(Path(artifact.tensor_dataset_path).exists())
        self.assertTrue(Path(artifact.profile_tokenizer_path).exists())
        self.assertTrue(Path(artifact.sequence_tokenizer_path).exists())
        self.assertTrue(Path(artifact.pairs_path).exists())
        self.assertTrue(Path(artifact.manifest_path).exists())

        dataset = torch.load(artifact.tensor_dataset_path)
        self.assertEqual((2,), tuple(dataset["profile_lengths"].shape))
        self.assertEqual((2,), tuple(dataset["sequence_lengths"].shape))
        self.assertEqual(dataset["profile_input_ids"].shape, dataset["profile_attention_mask"].shape)
        self.assertEqual(dataset["sequence_input_ids"].shape, dataset["sequence_attention_mask"].shape)
        self.assertEqual(2, len(dataset["metadata"]))
        self.assertEqual(3, dataset["config"]["kmer_size"])

        pairs_payload = [
            json.loads(line)
            for line in Path(artifact.pairs_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual("Drought tolerant gene in Oryza sativa", pairs_payload[0]["profile"])

    def test_reads_go_qualifiers_and_avoids_partial_keyword_false_positives(self) -> None:
        pairs = self.pipeline.build_pairs_from_genbank(self.go_genbank_path)

        self.assertEqual(1, len(pairs))
        self.assertEqual("DNA repair gene in Oryza sativa", pairs[0].profile)
        self.assertEqual("dna repair", pairs[0].matched_keywords[0])

    def test_audit_genbank_reports_match_distribution_and_unmatched_examples(self) -> None:
        report = audit_genbank(self.go_genbank_path, top_k_examples=5)

        self.assertEqual(2, report.summary.total_features)
        self.assertEqual(2, report.summary.eligible_feature_count)
        self.assertEqual(1, report.summary.matched_feature_count)
        self.assertEqual(1, report.summary.unmatched_feature_count)
        self.assertEqual({"gene": 1, "CDS": 1}, report.summary.feature_type_counts)
        self.assertEqual("DNA repair", next(iter(report.summary.label_counts)))
        self.assertEqual("CDS", next(iter(report.summary.unmatched_feature_type_counts)))
        self.assertIn("chlorophyllase isoform", report.summary.top_unmatched_annotations[0][0])

    def test_audit_fasta_and_gff_reports_label_counts(self) -> None:
        report = audit_fasta_and_gff(
            fasta_path=self.fasta_path,
            annotation_path=self.gff_path,
            organism="Oryza sativa",
            top_k_examples=5,
        )

        self.assertEqual(3, report.summary.total_features)
        self.assertEqual(3, report.summary.eligible_feature_count)
        self.assertEqual(2, report.summary.matched_feature_count)
        self.assertEqual(1, report.summary.unmatched_feature_count)
        self.assertEqual(0, report.summary.multi_label_feature_count)
        self.assertEqual(1, report.summary.label_counts["drought tolerant"])
        self.assertEqual(1, report.summary.label_counts["photosynthesis"])

    def test_default_rules_cover_broader_dna_and_rna_problem_space(self) -> None:
        broad_gff_path = self.root / "broad.gff3"
        broad_gff_path.write_text(
            (
                "##gff-version 3\n"
                "chr1\tPhytozome\tgene\t1\t18\t.\t+\t.\tID=gene1;Note=DNA repair helicase\n"
                "chr1\tPhytozome\tgene\t19\t33\t.\t+\t.\tID=gene2;Note=miRNA biogenesis dicer factor\n"
            ),
            encoding="utf-8",
        )

        pairs = self.pipeline.build_pairs_from_fasta_and_gff(
            fasta_path=self.fasta_path,
            annotation_path=broad_gff_path,
            organism="Oryza sativa",
        )

        profiles = {pair.profile for pair in pairs}
        self.assertIn("DNA repair gene in Oryza sativa", profiles)
        self.assertIn("RNA silencing gene in Oryza sativa", profiles)


class DefaultKeywordRuleTests(unittest.TestCase):
    def test_default_rules_are_not_limited_to_four_narrow_tasks(self) -> None:
        labels = {rule.label for rule in DEFAULT_KEYWORD_RULES}

        self.assertGreaterEqual(len(DEFAULT_KEYWORD_RULES), 20)
        self.assertIn("DNA repair", labels)
        self.assertIn("RNA processing", labels)
        self.assertIn("CRISPR defense", labels)
        self.assertIn("antibiotic resistance", labels)


class ProfileBPETokenizerTests(unittest.TestCase):
    def test_profile_bpe_round_trip_and_unknown_character_failure(self) -> None:
        text = "Drought tolerant gene in Oryza sativa\nPhotosynthesis gene in Oryza sativa\n"
        tokenizer = ProfileBPETokenizer.from_text(text, vocab_size=96)

        token_ids = tokenizer.encode("Drought tolerant gene in Oryza sativa", add_bos=True, add_eos=True)

        self.assertEqual(
            "Drought tolerant gene in Oryza sativa",
            tokenizer.decode(token_ids, skip_special=True),
        )

        with self.assertRaises(ValueError):
            tokenizer.encode("Drought tolerant gene #1")


class KmerTokenizerTests(unittest.TestCase):
    def test_triplet_encode_decode_round_trip(self) -> None:
        tokenizer = KmerTokenizer.from_sequences(["ATGCAT"], kmer_size=3)

        token_ids = tokenizer.encode("ATGCAT", add_bos=True, add_eos=True)

        self.assertEqual("ATGCAT", tokenizer.decode(token_ids, skip_special=True))


if __name__ == "__main__":
    unittest.main()
