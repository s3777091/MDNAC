from __future__ import annotations

import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data.distributed import DistributedSampler

from libs.core import (
    MDCDecoderModel,
    build_mdc_config_from_progen_config,
    build_or_load_protein_tokenizer,
    build_progen_config,
    create_protein_lm_dataloader,
    load_protein_pretrain_checkpoint,
    prepare_mdc_training_runtime,
    save_protein_pretrain_checkpoint,
)
from libs.core.pretrain.distributed import partition_items_for_worker


class FakeParallelWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class FakeDistributedDataParallel(FakeParallelWrapper):
    def __init__(self, module: torch.nn.Module, *, device_ids=None, find_unused_parameters=False) -> None:
        super().__init__(module)
        self.device_ids = device_ids
        self.find_unused_parameters = find_unused_parameters


class FakeDataParallel(FakeParallelWrapper):
    def __init__(self, module: torch.nn.Module, *, device_ids=None, output_device=None) -> None:
        super().__init__(module)
        self.device_ids = device_ids
        self.output_device = output_device


class TrainingRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("tests/artifacts/training-runtime")
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)
        self.train_path = self.root / "train.txt"
        self.train_path.write_text(
            (
                "<|protein|>MPEPTIDE<|endoftext|>\n"
                "<|protein|>GLYSERQ<|endoftext|>\n"
                "<|protein|>MVLSPADKTN<|endoftext|>\n"
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_prepare_training_runtime_initializes_ddp_from_explicit_context(self) -> None:
        model = torch.nn.Linear(4, 4)

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.set_device") as mock_set_device,
            patch.object(torch.nn.Module, "to", lambda self, *args, **kwargs: self),
            patch("torch.distributed.is_available", return_value=True),
            patch("torch.distributed.is_initialized", return_value=False),
            patch("torch.distributed.init_process_group") as mock_init_process_group,
            patch("torch.nn.parallel.DistributedDataParallel", FakeDistributedDataParallel),
        ):
            runtime = prepare_mdc_training_runtime(
                model,
                device="cuda",
                multi_gpu="ddp",
                rank=1,
                local_rank=1,
                world_size=2,
            )

        self.assertTrue(runtime.distributed)
        self.assertFalse(runtime.data_parallel)
        self.assertEqual(torch.device("cuda", 1), runtime.device)
        self.assertIsInstance(runtime.model, FakeDistributedDataParallel)
        mock_set_device.assert_called_once_with(1)
        mock_init_process_group.assert_called_once()

    def test_prepare_training_runtime_uses_config_mode_names(self) -> None:
        model = torch.nn.Linear(4, 4)

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=2),
            patch.object(torch.nn.Module, "to", lambda self, *args, **kwargs: self),
            patch("torch.nn.DataParallel", FakeDataParallel),
        ):
            runtime = prepare_mdc_training_runtime(
                model,
                device="cuda:0",
                multi_gpu="data_parallel",
                data_parallel_device_ids=[0, 1],
            )

        self.assertFalse(runtime.distributed)
        self.assertTrue(runtime.data_parallel)
        self.assertIsInstance(runtime.model, FakeDataParallel)
        self.assertEqual([0, 1], runtime.model.device_ids)

    def test_prepare_training_runtime_disables_parallelism_with_none_mode(self) -> None:
        model = torch.nn.Linear(4, 4)

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=2),
            patch.object(torch.nn.Module, "to", lambda self, *args, **kwargs: self),
        ):
            runtime = prepare_mdc_training_runtime(
                model,
                device="cuda:0",
                multi_gpu="none",
            )

        self.assertFalse(runtime.distributed)
        self.assertFalse(runtime.data_parallel)

    def test_prepare_training_runtime_sets_windows_ddp_libuv_env(self) -> None:
        model = torch.nn.Linear(4, 4)
        original_use_libuv = os.environ.pop("USE_LIBUV", None)
        try:
            with (
                patch("platform.system", return_value="Windows"),
                patch("torch.cuda.is_available", return_value=True),
                patch("torch.cuda.set_device"),
                patch.object(torch.nn.Module, "to", lambda self, *args, **kwargs: self),
                patch("torch.distributed.is_available", return_value=True),
                patch("torch.distributed.is_initialized", return_value=False),
                patch("torch.distributed.init_process_group") as mock_init_process_group,
                patch("torch.nn.parallel.DistributedDataParallel", FakeDistributedDataParallel),
            ):
                prepare_mdc_training_runtime(
                    model,
                    device="cuda",
                    multi_gpu="ddp",
                    rank=0,
                    local_rank=0,
                    world_size=2,
                )
            self.assertEqual("0", os.environ.get("USE_LIBUV"))
            mock_init_process_group.assert_called_once()
            self.assertEqual("gloo", mock_init_process_group.call_args.kwargs["backend"])
        finally:
            if original_use_libuv is None:
                os.environ.pop("USE_LIBUV", None)
            else:
                os.environ["USE_LIBUV"] = original_use_libuv

    def test_create_protein_loader_uses_distributed_sampler_when_requested(self) -> None:
        tokenizer_artifact = build_or_load_protein_tokenizer(self.train_path, vocab_size=64)

        data_loader = create_protein_lm_dataloader(
            self.train_path.read_text(encoding="utf-8"),
            tokenizer_artifact.tokenizer,
            context_length=12,
            stride=6,
            batch_size=2,
            shuffle=True,
            pin_memory=False,
            distributed=True,
            rank=1,
            world_size=2,
        )

        self.assertIsInstance(data_loader.sampler, DistributedSampler)
        self.assertEqual(2, data_loader.sampler.num_replicas)
        self.assertEqual(1, data_loader.sampler.rank)

    def test_partition_items_for_worker_combines_rank_and_worker_sharding(self) -> None:
        items = list(range(12))

        rank_zero_items, rank_zero_partition = partition_items_for_worker(
            items,
            rank=0,
            world_size=2,
            worker_id=1,
            num_workers=2,
        )
        rank_one_items, rank_one_partition = partition_items_for_worker(
            items,
            rank=1,
            world_size=2,
            worker_id=1,
            num_workers=2,
        )

        self.assertEqual(1, rank_zero_partition)
        self.assertEqual(3, rank_one_partition)
        self.assertEqual([1, 5, 9], rank_zero_items)
        self.assertEqual([3, 7, 11], rank_one_items)

    def test_checkpoint_save_and_load_accept_parallel_wrappers(self) -> None:
        tokenizer_artifact = build_or_load_protein_tokenizer(self.train_path, vocab_size=64)
        base_config = build_progen_config(
            "0.8B",
            vocab_size=tokenizer_artifact.vocab_size,
            context_length=12,
            dtype=torch.float32,
        )
        model_config = build_mdc_config_from_progen_config(
            {
                **base_config,
                "emb_dim": 32,
                "n_heads": 4,
                "n_layers": 2,
                "hidden_dim": 64,
                "head_dim": 8,
                "n_kv_groups": 2,
                "linear_key_head_dim": 8,
                "linear_value_head_dim": 8,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 2,
            },
            dtype=torch.float32,
        )
        model = MDCDecoderModel(model_config)
        wrapped_model = FakeParallelWrapper(model)
        optimizer = torch.optim.AdamW(wrapped_model.parameters(), lr=1e-3)

        checkpoint_path = save_protein_pretrain_checkpoint(
            self.root / "checkpoint_best.pt",
            model=wrapped_model,
            optimizer=optimizer,
            model_config=model_config,
            tokenizer=tokenizer_artifact.tokenizer,
            tokenizer_map_path=tokenizer_artifact.tokenizer_map_path,
            epoch=1,
            global_step=1,
            tokens_seen=12,
            train_losses=[1.0],
            val_losses=[1.5],
            training_args={"multi_gpu": "auto"},
        )

        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
        self.assertTrue(all(not key.startswith("module.") for key in checkpoint_payload["model_state_dict"]))

        resumed_model = FakeParallelWrapper(MDCDecoderModel(model_config))
        checkpoint = load_protein_pretrain_checkpoint(
            checkpoint_path,
            model=resumed_model,
            optimizer=torch.optim.AdamW(resumed_model.parameters(), lr=1e-3),
        )

        self.assertEqual(1, checkpoint["epoch"])
        self.assertEqual(1, checkpoint["global_step"])


if __name__ == "__main__":
    unittest.main()
