from __future__ import annotations

import math
import unittest

import torch

from libs.core.pretrain.lr_schedule import (
    LRScheduleConfig,
    WarmupCosineLRScheduler,
    build_warmup_cosine_scheduler,
    cosine_warmup_lr_scale,
)


def _make_optimizer(lr: float) -> torch.optim.Optimizer:
    param = torch.nn.Parameter(torch.zeros(2))
    return torch.optim.AdamW([param], lr=lr)


class CosineWarmupScaleTests(unittest.TestCase):
    def test_linear_warmup_ramps_from_small_to_one(self) -> None:
        scales = [
            cosine_warmup_lr_scale(step, warmup_steps=10, total_steps=100, min_lr_ratio=0.1)
            for step in range(10)
        ]
        self.assertAlmostEqual(scales[0], 0.1)  # (0 + 1) / 10
        self.assertAlmostEqual(scales[9], 1.0)  # (9 + 1) / 10
        self.assertEqual(scales, sorted(scales))  # monotonically increasing

    def test_cosine_decays_to_min_lr_ratio(self) -> None:
        peak = cosine_warmup_lr_scale(10, warmup_steps=10, total_steps=110, min_lr_ratio=0.1)
        mid = cosine_warmup_lr_scale(60, warmup_steps=10, total_steps=110, min_lr_ratio=0.1)
        end = cosine_warmup_lr_scale(110, warmup_steps=10, total_steps=110, min_lr_ratio=0.1)
        self.assertAlmostEqual(peak, 1.0)
        self.assertAlmostEqual(mid, 0.55, places=5)  # 0.1 + 0.9 * 0.5
        self.assertAlmostEqual(end, 0.1, places=6)

    def test_holds_at_peak_when_horizon_unknown(self) -> None:
        after_warmup = cosine_warmup_lr_scale(50, warmup_steps=10, total_steps=None, min_lr_ratio=0.1)
        self.assertEqual(after_warmup, 1.0)


class WarmupCosineLRSchedulerTests(unittest.TestCase):
    def test_applies_scaled_lr_to_param_groups(self) -> None:
        optimizer = _make_optimizer(1e-3)
        scheduler = WarmupCosineLRScheduler(
            [optimizer], warmup_steps=4, total_steps=None, min_lr_ratio=0.1, last_step=0
        )
        # step 0 => scale 1/4
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-3 * 0.25)
        scheduler.step()  # step 1 => 2/4
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-3 * 0.5)
        scheduler.step()  # step 2 => 3/4
        scheduler.step()  # step 3 => 4/4 = peak
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-3)

    def test_resume_sets_lr_for_restored_step(self) -> None:
        optimizer = _make_optimizer(1e-3)
        scheduler = WarmupCosineLRScheduler(
            [optimizer], warmup_steps=10, total_steps=110, min_lr_ratio=0.0, last_step=60
        )
        expected = 1e-3 * 0.5 * (1.0 + math.cos(math.pi * 0.5))
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], expected, places=8)

    def test_scales_multiple_optimizers_independently(self) -> None:
        opt_a = _make_optimizer(1e-3)
        opt_b = _make_optimizer(5e-4)
        scheduler = WarmupCosineLRScheduler(
            [opt_a, opt_b], warmup_steps=2, total_steps=None, min_lr_ratio=0.1, last_step=0
        )
        self.assertAlmostEqual(opt_a.param_groups[0]["lr"], 1e-3 * 0.5)
        self.assertAlmostEqual(opt_b.param_groups[0]["lr"], 5e-4 * 0.5)


class BuildSchedulerTests(unittest.TestCase):
    def test_returns_none_when_disabled(self) -> None:
        optimizer = _make_optimizer(1e-3)
        schedule = LRScheduleConfig.from_optimizer_config({"lr_scheduler": "none"})
        self.assertIsNone(
            build_warmup_cosine_scheduler([optimizer], schedule, max_steps=None, last_step=0)
        )

    def test_max_steps_resolves_cosine_horizon(self) -> None:
        optimizer = _make_optimizer(1e-3)
        schedule = LRScheduleConfig.from_optimizer_config(
            {"lr_scheduler": "cosine", "warmup_steps": 5, "min_lr_ratio": 0.0}
        )
        scheduler = build_warmup_cosine_scheduler(
            [optimizer], schedule, max_steps=105, last_step=0
        )
        assert scheduler is not None
        self.assertTrue(scheduler.decays)
        self.assertEqual(scheduler.total_steps, 105)

    def test_warmup_ratio_uses_resolved_total(self) -> None:
        schedule = LRScheduleConfig.from_optimizer_config(
            {"lr_scheduler": "cosine", "warmup_ratio": 0.1}
        )
        self.assertEqual(schedule.resolve_warmup_steps(1000), 100)


if __name__ == "__main__":
    unittest.main()
