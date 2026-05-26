from __future__ import annotations

import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from libs.core.pretrain.training_config import (
    apply_protein_training_optimizer_settings,
    build_protein_training_data_config,
    create_protein_training_optimizer,
    describe_protein_training_optimizers,
    load_protein_training_config,
)


class FakeMuon(torch.optim.SGD):
    def __init__(self, params, *, lr, weight_decay=0.0, adjust_lr_fn=None):
        self.adjust_lr_fn = adjust_lr_fn
        super().__init__(params, lr=lr, weight_decay=weight_decay)


class ProteinTrainingConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("tests/artifacts/protein-training-config")
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)
        self.env_keys = (
            "MICROBIAL_DATA_MINIO_ACCESS_KEY",
            "MICROBIAL_DATA_MINIO_SECRET_KEY",
            "MICROBIAL_DATA_MINIO_PREFIX",
        )
        self.original_env = {key: os.environ.get(key) for key in self.env_keys}
        for key in self.env_keys:
            os.environ.pop(key, None)

        (self.root / "config.yaml").write_text(
            (
                "storage_mode: local\n"
                "data_root: libs/data\n"
                "default_batch_size: 25\n"
                "minio:\n"
                "  endpoint_url: http://base.minio:9000\n"
                "  bucket_name: base-bucket\n"
                "  secure: true\n"
            ),
            encoding="utf-8",
        )
        (self.root / ".env").write_text(
            (
                "MICROBIAL_DATA_MINIO_ACCESS_KEY=env-access\n"
                "MICROBIAL_DATA_MINIO_SECRET_KEY=env-secret\n"
                "MICROBIAL_DATA_MINIO_PREFIX=protein/root\n"
            ),
            encoding="utf-8",
        )
        (self.root / "train.yaml").write_text(
            (
                "paths:\n"
                "  train_text_path: data/compiled/custom/train.txt\n"
                "  checkpoint_dir: data/checkpoints/custom\n"
                "data:\n"
                "  batch_size: 3\n"
                "  pin_memory: false\n"
                "model:\n"
                "  context_length: 64\n"
                "  stride: 32\n"
                "optimizer:\n"
                "  type: muon\n"
                "  learning_rate: 0.001\n"
                "  muon_learning_rate: 0.002\n"
                "  weight_decay: 0.2\n"
                "  fused: false\n"
                "runtime:\n"
                "  data_parallel_device_ids: [0, 1]\n"
                "resume:\n"
                "  restore_optimizer_state: false\n"
                "minio:\n"
                "  train_parts_prefix_uri: s3://bucket/prefix\n"
                "  endpoint_url: http://yaml.minio:9000\n"
                "  bucket_name: yaml-bucket\n"
                "  secure: false\n"
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(self.root, ignore_errors=True)

    def test_loads_train_yaml_with_resolved_paths(self) -> None:
        config = load_protein_training_config(self.root)

        self.assertEqual(
            (self.root / "data/compiled/custom/train.txt").resolve(),
            config["paths"]["train_text_path"],
        )
        self.assertEqual(
            (self.root / "data/compiled/custom/tokenizer_map.json").resolve(),
            config["paths"]["tokenizer_map_path"],
        )
        self.assertEqual("muon", config["optimizer"]["type"])
        self.assertEqual((0, 1), config["runtime"]["data_parallel_device_ids"])
        self.assertFalse(config["data"]["pin_memory"])
        self.assertEqual(64, config["model"]["context_length"])
        self.assertFalse(config["resume"]["restore_optimizer_state"])

    def test_resolves_auto_booleans_to_runtime_defaults(self) -> None:
        train_config_path = self.root / "train.yaml"
        train_config_path.write_text(
            train_config_path.read_text(encoding="utf-8")
            .replace("  pin_memory: false\n", "  pin_memory: auto\n")
            .replace("  fused: false\n", "  fused: auto\n"),
            encoding="utf-8",
        )

        with patch.object(torch.cuda, "is_available", return_value=False):
            cpu_config = load_protein_training_config(self.root)
        self.assertFalse(cpu_config["data"]["pin_memory"])
        self.assertTrue(cpu_config["optimizer"]["fused"])

        with patch.object(torch.cuda, "is_available", return_value=True):
            cuda_config = load_protein_training_config(self.root)
        self.assertTrue(cuda_config["data"]["pin_memory"])

    def test_builds_minio_data_config_from_train_yaml_overrides(self) -> None:
        config = load_protein_training_config(self.root)

        data_config = build_protein_training_data_config(self.root, config)

        self.assertIsNotNone(data_config)
        assert data_config is not None
        self.assertEqual("http://yaml.minio:9000", data_config.minio.endpoint_url)
        self.assertEqual("env-access", data_config.minio.access_key)
        self.assertEqual("env-secret", data_config.minio.secret_key)
        self.assertEqual("yaml-bucket", data_config.minio.bucket_name)
        self.assertEqual("protein/root", data_config.minio.root_prefix)
        self.assertFalse(data_config.minio.secure)

    def test_creates_muon_optimizer_and_reapplies_yaml_hyperparameters(self) -> None:
        config = load_protein_training_config(self.root)
        model = torch.nn.Sequential(
            torch.nn.Embedding(8, 4),
            torch.nn.Linear(4, 4),
        )

        original_muon = getattr(torch.optim, "Muon", None)
        torch.optim.Muon = FakeMuon
        try:
            optimizer = create_protein_training_optimizer(
                model,
                config["optimizer"],
                device="cpu",
            )
        finally:
            if original_muon is None:
                delattr(torch.optim, "Muon")
            else:
                torch.optim.Muon = original_muon

        self.assertEqual(["FakeMuon", "AdamW"], describe_protein_training_optimizers(optimizer))
        self.assertEqual("match_rms_adamw", optimizer[0].adjust_lr_fn)

        for opt in optimizer:
            for group in opt.param_groups:
                group["lr"] = 99.0
                group["weight_decay"] = 0.0

        apply_protein_training_optimizer_settings(optimizer, config["optimizer"])

        self.assertEqual(0.002, optimizer[0].param_groups[0]["lr"])
        self.assertEqual(0.001, optimizer[1].param_groups[0]["lr"])
        self.assertEqual(0.2, optimizer[0].param_groups[0]["weight_decay"])
        self.assertEqual(0.2, optimizer[1].param_groups[0]["weight_decay"])


if __name__ == "__main__":
    unittest.main()
