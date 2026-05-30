from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from libs.core.pretrain.protein_lm.resume_state import (
    create_resume_state,
    load_resume_state,
    save_resume_state,
    update_resume_state_metrics,
    update_resume_state_progress,
)


class ResumeStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("tests/artifacts/resume-state-test")
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_create_and_save_resume_state(self) -> None:
        state = create_resume_state(
            mode="train_from_scratch",
            config_path="train.yaml",
            training_config_snapshot_path="data/ckpt/snapshot.json",
            checkpoint_path="data/ckpt/last.pt",
            best_checkpoint_path="data/ckpt/best.pt",
            final_checkpoint_path="data/ckpt/final.pt",
            model_info={"progen_model_size": "0.8B", "context_length": 512, "stride": 256},
            optimizer_info={"type": "muon", "learning_rate": 3e-4},
            runtime_info={"device": "cpu", "distributed": False, "rank": 0, "world_size": 1},
        )
        self.assertEqual("train_from_scratch", state["mode"])
        self.assertEqual(0, state["progress"]["global_step"])
        self.assertIsNotNone(state["run_id"])

        path = self.root / "resume_state.json"
        save_resume_state(state, path)
        self.assertTrue(path.exists())

        loaded = load_resume_state(path)
        self.assertIsNotNone(loaded)
        self.assertEqual(state["run_id"], loaded["run_id"])

    def test_atomic_write_creates_backup(self) -> None:
        path = self.root / "resume_state.json"
        state = create_resume_state(
            mode="resume",
            config_path="x",
            training_config_snapshot_path="x",
            checkpoint_path="x",
            best_checkpoint_path="x",
            final_checkpoint_path="x",
            model_info={},
            optimizer_info={},
            runtime_info={},
        )
        save_resume_state(state, path)

        update_resume_state_progress(state, global_step=100)
        save_resume_state(state, path)

        backup = path.with_suffix(".json.bak")
        self.assertTrue(backup.exists())
        backup_data = json.loads(backup.read_text(encoding="utf-8"))
        self.assertEqual(0, backup_data["progress"]["global_step"])

        current = load_resume_state(path)
        self.assertEqual(100, current["progress"]["global_step"])

    def test_update_progress_and_metrics(self) -> None:
        state = create_resume_state(
            mode="train_from_scratch",
            config_path="x",
            training_config_snapshot_path="x",
            checkpoint_path="x",
            best_checkpoint_path="x",
            final_checkpoint_path="x",
            model_info={},
            optimizer_info={},
            runtime_info={},
        )
        update_resume_state_progress(state, epoch=2, global_step=50, tokens_seen=1000)
        self.assertEqual(2, state["progress"]["epoch"])
        self.assertEqual(50, state["progress"]["global_step"])
        self.assertEqual(1000, state["progress"]["tokens_seen"])

        update_resume_state_metrics(state, train_loss=0.5, val_loss=0.6, best_loss=0.5)
        self.assertEqual(0.5, state["metrics"]["latest_train_loss"])
        self.assertEqual(0.6, state["metrics"]["latest_val_loss"])
        self.assertEqual(0.5, state["metrics"]["best_loss"])
        self.assertEqual([0.5], state["metrics"]["train_losses"])
        self.assertEqual([0.6], state["metrics"]["val_losses"])

    def test_load_nonexistent_returns_none(self) -> None:
        result = load_resume_state(self.root / "nonexistent.json")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
