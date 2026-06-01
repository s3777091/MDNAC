"""Tests for the candidate validation pipeline."""

from __future__ import annotations

import unittest

from libs.core.structure.candidates import (
    CandidateValidationConfig,
    GeneratedProteinCandidate,
    rank_candidates,
    validate_generated_candidate,
    validate_sequence_basic,
    validate_structure_prediction,
)
from libs.core.structure.types import StructurePrediction


class TestValidateSequenceBasic(unittest.TestCase):
    def setUp(self) -> None:
        self.config = CandidateValidationConfig()

    def test_valid_sequence_passes(self) -> None:
        sequence = "ACDEFGHIKLMNPQRSTVWY" * 3  # 60 residues, all valid
        scores, reasons = validate_sequence_basic(sequence, self.config)
        self.assertEqual(reasons, [])
        self.assertEqual(scores["validity"], 1.0)
        self.assertEqual(scores["length"], 1.0)

    def test_empty_sequence_fails(self) -> None:
        scores, reasons = validate_sequence_basic("", self.config)
        self.assertIn("empty_sequence", reasons)
        self.assertEqual(scores["validity"], 0.0)

    def test_invalid_amino_acid_fails(self) -> None:
        scores, reasons = validate_sequence_basic("MPEPTIDE123", self.config)
        self.assertIn("invalid_amino_acid", reasons)
        self.assertLess(scores["validity"], 1.0)

    def test_too_many_x_fails(self) -> None:
        # 6 X out of 10 = 60% > 5%
        scores, reasons = validate_sequence_basic("MXXXXXXAAA", self.config)
        self.assertIn("too_many_x", reasons)

    def test_short_sequence_fails(self) -> None:
        config = CandidateValidationConfig(min_length=50)
        scores, reasons = validate_sequence_basic("MPEPTIDE", config)
        self.assertIn("length_outside_window", reasons)
        self.assertLess(scores["length"], 1.0)

    def test_long_sequence_fails(self) -> None:
        config = CandidateValidationConfig(min_length=5, max_length=10)
        scores, reasons = validate_sequence_basic("ACDEFGHIKLMNPQRSTVWY", config)
        self.assertIn("length_outside_window", reasons)

    def test_x_fraction_below_threshold_passes(self) -> None:
        # 1 X out of 100 = 1% < 5%
        sequence = "ACDEFGHIKLMNPQRSTVWY" * 5  # 100 residues, all valid
        scores, reasons = validate_sequence_basic(sequence, self.config)
        self.assertEqual(reasons, [])


class TestValidateStructurePrediction(unittest.TestCase):
    def setUp(self) -> None:
        self.config = CandidateValidationConfig(min_plddt=0.7)

    def test_missing_prediction_reports_missing_provider(self) -> None:
        scores, reasons = validate_structure_prediction(None, self.config)
        self.assertIn("missing_structure_provider", reasons)
        self.assertEqual(scores["model_confidence"], 0.0)

    def test_low_plddt_fails(self) -> None:
        prediction = StructurePrediction(
            sequence="MPEPTIDE",
            model_name="test-model",
            plddt=0.5,
        )
        scores, reasons = validate_structure_prediction(prediction, self.config)
        self.assertIn("low_plddt", reasons)

    def test_sufficient_plddt_passes(self) -> None:
        prediction = StructurePrediction(
            sequence="MPEPTIDE",
            model_name="test-model",
            plddt=0.85,
        )
        scores, reasons = validate_structure_prediction(prediction, self.config)
        self.assertNotIn("low_plddt", reasons)
        self.assertGreater(scores["model_confidence"], 0.0)

    def test_missing_provider_does_not_silently_pass(self) -> None:
        config = CandidateValidationConfig(min_confidence=0.5)
        scores, reasons = validate_structure_prediction(None, config)
        self.assertIn("missing_structure_provider", reasons)
        self.assertFalse(len(reasons) == 0)

    def test_low_ptm_fails(self) -> None:
        config = CandidateValidationConfig(min_ptm=0.6)
        prediction = StructurePrediction(
            sequence="MPEPTIDE",
            model_name="test-model",
            ptm=0.4,
        )
        _, reasons = validate_structure_prediction(prediction, config)
        self.assertIn("low_ptm", reasons)

    def test_low_iptm_fails(self) -> None:
        config = CandidateValidationConfig(min_iptm=0.5)
        prediction = StructurePrediction(
            sequence="MPEPTIDE",
            model_name="test-model",
            iptm=0.3,
        )
        _, reasons = validate_structure_prediction(prediction, config)
        self.assertIn("low_iptm", reasons)


class TestValidateGeneratedCandidate(unittest.TestCase):
    def test_valid_candidate_passes(self) -> None:
        candidate = GeneratedProteinCandidate(
            profile="dna gyrase",
            sequence="ACDEFGHIKLMNPQRSTVWY" * 3,
        )
        config = CandidateValidationConfig()
        result = validate_generated_candidate(candidate, config)
        # Without a structure prediction, "missing_structure_provider" is a reason
        self.assertIn("missing_structure_provider", result.reasons)

    def test_valid_candidate_with_prediction_passes(self) -> None:
        prediction = StructurePrediction(
            sequence="ACDEFGHIKLMNPQRSTVWY" * 3,
            model_name="test",
            plddt=0.9,
        )
        candidate = GeneratedProteinCandidate(
            profile="dna gyrase",
            sequence="ACDEFGHIKLMNPQRSTVWY" * 3,
            prediction=prediction,
        )
        config = CandidateValidationConfig()
        result = validate_generated_candidate(candidate, config)
        self.assertTrue(result.passed)
        self.assertEqual(result.reasons, ())

    def test_invalid_sequence_fails_even_with_good_prediction(self) -> None:
        prediction = StructurePrediction(
            sequence="123",
            model_name="test",
            plddt=0.9,
        )
        candidate = GeneratedProteinCandidate(
            profile="test",
            sequence="123",
            prediction=prediction,
        )
        config = CandidateValidationConfig()
        result = validate_generated_candidate(candidate, config)
        self.assertFalse(result.passed)
        self.assertIn("invalid_amino_acid", result.reasons)


class TestRankCandidates(unittest.TestCase):
    def test_ranks_by_validation_score_descending(self) -> None:
        c1 = GeneratedProteinCandidate(profile="p", sequence="A", validation_score=0.3)
        c2 = GeneratedProteinCandidate(profile="p", sequence="B", validation_score=0.9)
        c3 = GeneratedProteinCandidate(profile="p", sequence="C", validation_score=0.6)

        ranked = rank_candidates([c1, c2, c3])
        self.assertEqual([c.sequence for c in ranked], ["B", "C", "A"])

    def test_empty_list(self) -> None:
        self.assertEqual(rank_candidates([]), [])


if __name__ == "__main__":
    unittest.main()
