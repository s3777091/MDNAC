from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys


API_ROOT = Path(__file__).resolve().parents[1] / "api"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))


class TestOpenFoldStructureAPI(unittest.TestCase):
    def test_builds_soloseq_cif_command(self) -> None:
        from structure_predictor.config import OpenFoldSettings
        from structure_predictor.openfold import build_openfold_command

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = OpenFoldSettings(
                repo_path=root / "openfold",
                python_executable="python",
                output_root=root / "out",
                template_mmcif_dir=root / "mmcif",
                config_preset="seq_model_esm1b_ptm",
                model_device="cuda:0",
                output_format="pdb",
                timeout_seconds=60,
                cpus=2,
                min_sequence_length=1,
                max_sequence_length=1022,
                include_structure_text=True,
                max_response_structure_bytes=1000,
                use_precomputed_alignments=False,
                precomputed_alignments_dir=None,
                openfold_checkpoint_path=root / "seq_model_esm1b_ptm.pt",
                jax_param_path=None,
                data_random_seed=7,
                skip_relaxation=True,
                long_sequence_inference=False,
                use_single_seq_mode=True,
                database_paths={},
                binary_paths={},
                extra_args=("--precision", "bf16"),
            )

            command = build_openfold_command(
                settings,
                fasta_dir=root / "fasta",
                output_dir=root / "prediction",
                output_format="cif",
                config_preset="seq_model_esm1b_ptm",
            )

        self.assertIn("--cif_output", command)
        self.assertIn("--openfold_checkpoint_path", command)
        self.assertIn("--skip_relaxation", command)
        self.assertIn("--use_single_seq_mode", command)
        self.assertIn("--precision", command)
        self.assertNotIn("--use_precomputed_alignments", command)

    def test_sequence_validation_rejects_invalid_residue(self) -> None:
        from structure_predictor.openfold import normalize_protein_sequence

        with self.assertRaisesRegex(ValueError, "invalid residues"):
            normalize_protein_sequence("MPEPTIDE1", min_length=1, max_length=1022)

    def test_finds_relaxed_structure_first(self) -> None:
        from structure_predictor.openfold import find_structure_output

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            predictions = output_dir / "predictions"
            predictions.mkdir()
            unrelaxed = predictions / "candidate_model_unrelaxed.pdb"
            relaxed = predictions / "candidate_model_relaxed.pdb"
            unrelaxed.write_text("unrelaxed", encoding="utf-8")
            relaxed.write_text("relaxed", encoding="utf-8")

            result = find_structure_output(output_dir, output_format="pdb")

        self.assertEqual("candidate_model_relaxed.pdb", result.name)


if __name__ == "__main__":
    unittest.main()
