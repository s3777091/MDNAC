"""Tests for contact constraint evaluation."""

from __future__ import annotations

import unittest

import numpy as np

from libs.core.structure.contact_constraints import (
    ContactConstraint,
    build_contact_constraints_from_msa,
    evaluate_contact_constraints,
    evaluate_triangle_geometry,
)


class TestBuildContactConstraintsFromMSA(unittest.TestCase):
    def test_produces_constraints_from_aligned_msa(self) -> None:
        # Simple MSA with covarying positions
        msa = [
            "ACDEFGHIKLM",
            "ACDEYGHIKLM",
            "ACDEFGHIRLM",
            "ACDEYGHIRLM",
            "ACDEFGHIKLM",
            "ACDEYGHIRLM",
        ]
        constraints = build_contact_constraints_from_msa(msa, top_k=5, min_separation=3)
        self.assertIsInstance(constraints, tuple)
        self.assertLessEqual(len(constraints), 5)
        for c in constraints:
            self.assertIsInstance(c, ContactConstraint)
            self.assertGreaterEqual(abs(c.j - c.i), 3)
            self.assertEqual(c.max_distance, 8.0)

    def test_empty_msa_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_contact_constraints_from_msa([], top_k=5)

    def test_single_sequence_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_contact_constraints_from_msa(["ACDEFG"], top_k=5)


class TestEvaluateContactConstraints(unittest.TestCase):
    def test_all_constraints_satisfied(self) -> None:
        # 4x4 distance matrix where close pairs are within 8A
        distances = np.array([
            [0.0, 5.0, 6.0, 12.0],
            [5.0, 0.0, 4.0, 10.0],
            [6.0, 4.0, 0.0, 7.0],
            [12.0, 10.0, 7.0, 0.0],
        ])
        constraints = [
            ContactConstraint(i=0, j=1, max_distance=8.0),
            ContactConstraint(i=1, j=2, max_distance=8.0),
            ContactConstraint(i=2, j=3, max_distance=8.0),
        ]
        score, reasons = evaluate_contact_constraints(distances, constraints)
        self.assertEqual(score, 1.0)
        self.assertEqual(reasons, [])

    def test_violated_constraint(self) -> None:
        distances = np.array([
            [0.0, 15.0, 6.0],
            [15.0, 0.0, 4.0],
            [6.0, 4.0, 0.0],
        ])
        constraints = [
            ContactConstraint(i=0, j=1, max_distance=8.0),  # 15 > 8 -> violated
            ContactConstraint(i=1, j=2, max_distance=8.0),  # 4 <= 8 -> satisfied
        ]
        score, reasons = evaluate_contact_constraints(distances, constraints)
        self.assertEqual(score, 0.5)
        self.assertEqual(len(reasons), 1)
        self.assertIn("contact_violated(0,1", reasons[0])

    def test_empty_constraints_returns_perfect_score(self) -> None:
        distances = np.eye(3)
        score, reasons = evaluate_contact_constraints(distances, [])
        self.assertEqual(score, 1.0)
        self.assertEqual(reasons, [])

    def test_non_square_matrix_raises(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_contact_constraints(
                np.zeros((3, 4)),
                [ContactConstraint(i=0, j=1)],
            )

    def test_min_distance_constraint(self) -> None:
        distances = np.array([
            [0.0, 2.0],
            [2.0, 0.0],
        ])
        constraints = [
            ContactConstraint(i=0, j=1, min_distance=5.0, max_distance=None),
        ]
        score, reasons = evaluate_contact_constraints(distances, constraints)
        self.assertEqual(score, 0.0)
        self.assertEqual(len(reasons), 1)


class TestEvaluateTriangleGeometry(unittest.TestCase):
    def test_valid_coordinates_give_high_score(self) -> None:
        # Simple triangle in 3D space - must satisfy triangle inequality
        coordinates = np.array([
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [1.5, 2.6, 0.0],
        ])
        score = evaluate_triangle_geometry(coordinates)
        self.assertEqual(score, 1.0)

    def test_collinear_points_still_consistent(self) -> None:
        # Collinear points satisfy triangle inequality (d_ac = d_ab + d_bc)
        coordinates = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ])
        score = evaluate_triangle_geometry(coordinates)
        self.assertEqual(score, 1.0)

    def test_invalid_coordinates_shape_raises(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_triangle_geometry(np.zeros((3, 2)))


if __name__ == "__main__":
    unittest.main()
