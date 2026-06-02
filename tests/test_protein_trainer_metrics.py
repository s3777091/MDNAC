from __future__ import annotations

import json
import math
import shutil
import unittest
from pathlib import Path

from libs.core.pretrain.protein_lm.services import DataLoaderFactory, MetricsWriter
from libs.core.pretrain.protein_lm.trainer import (
    ProteinPretrainTrainer,
    _format_eval_loss,
    _restore_best_val_loss,
)


class ProteinTrainerMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("tests/artifacts/protein-trainer-metrics")
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_validation_unavailable_is_logged_as_not_available(self) -> None:
        self.assertEqual("n/a", _format_eval_loss(float("nan"), available=False))

    def test_validation_unavailable_never_saves_best_checkpoint(self) -> None:
        trainer = ProteinPretrainTrainer.__new__(ProteinPretrainTrainer)
        trainer._best_val_loss = math.inf
        trainer._best_metric_name = "val_loss"
        trainer._save_best_checkpoint = lambda: self.fail("best checkpoint should not be saved")

        improved = trainer._maybe_save_best(4.2, has_validation_loader=False, save_best=True)

        self.assertFalse(improved)
        self.assertTrue(math.isinf(trainer._best_val_loss))
        self.assertEqual("val_loss", trainer._best_metric_name)

    def test_restore_keeps_validation_best_loss(self) -> None:
        best_loss = _restore_best_val_loss(
            {"best_metric_name": "val_loss", "val_losses": [4.2]},
            best_loss=4.2,
        )

        self.assertEqual(4.2, best_loss)

    def test_restore_ignores_legacy_train_loss_best_checkpoint(self) -> None:
        best_loss = _restore_best_val_loss(
            {"best_metric_name": "train_loss", "val_losses": [float("nan")]},
            best_loss=4.2,
        )

        self.assertTrue(math.isinf(best_loss))

    def test_metrics_writer_serializes_nonfinite_losses_as_null(self) -> None:
        metrics_path = self.root / "metrics_history.jsonl"
        MetricsWriter(metrics_path).append(
            epoch=1,
            global_step=200,
            tokens_seen=10,
            train_loss=4.2,
            val_loss=float("nan"),
        )

        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        self.assertEqual(4.2, payload["train_loss"])
        self.assertIsNone(payload["val_loss"])

    def test_discovers_local_parts_from_configured_cache_dir(self) -> None:
        cache_dir = self.root / "data" / "cache" / "protein_train_parts"
        compiled_dir = self.root / "data" / "compiled" / "refseq_bacteria_protein"
        cache_dir.mkdir(parents=True, exist_ok=True)
        compiled_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "train_part_10.txt").write_text("<|protein|>CCC<|endoftext|>\n", encoding="utf-8")
        (cache_dir / "train_part_2.txt").write_text("<|protein|>BBB<|endoftext|>\n", encoding="utf-8")
        (cache_dir / "train_part_1.txt").write_text("<|protein|>AAA<|endoftext|>\n", encoding="utf-8")

        trainer = ProteinPretrainTrainer.__new__(ProteinPretrainTrainer)
        trainer._paths = {
            "train_text_path": compiled_dir / "train.txt",
            "train_part_cache_dir": cache_dir,
        }
        trainer._data_cfg = {
            "train_part_glob": "train_part_*.txt",
            "prefer_local_train_parts": True,
        }

        local_paths = trainer._discover_local_paths()

        self.assertEqual(
            [
                cache_dir / "train_part_1.txt",
                cache_dir / "train_part_2.txt",
                cache_dir / "train_part_10.txt",
            ],
            list(local_paths),
        )

    def test_streams_single_local_part_but_not_plain_train_text(self) -> None:
        factory = DataLoaderFactory.__new__(DataLoaderFactory)
        factory._data_cfg = {
            "stream_local_train_parts": True,
            "train_part_glob": "train_part_*.txt",
        }

        self.assertTrue(factory._use_local_streaming((Path("train_part_1.txt"),)))
        self.assertFalse(factory._use_local_streaming((Path("train.txt"),)))


if __name__ == "__main__":
    unittest.main()
