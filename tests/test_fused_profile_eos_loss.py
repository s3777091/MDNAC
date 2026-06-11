from __future__ import annotations

import unittest

import torch

from libs.core.interfaces import IGNORE_INDEX, FusedProfileSequenceBatch


def _make_fused_batch() -> FusedProfileSequenceBatch:
    # Fused row layout: [BOS, profile, SEP, span0, span1, EOS]
    #                     0    1        2    3      4      5
    return FusedProfileSequenceBatch(
        token_ids=torch.tensor([[1, 4, 3, 5, 6, 2]], dtype=torch.long),
        attention_mask=torch.ones((1, 6), dtype=torch.long),
        profile_spans=torch.tensor([[1, 2]], dtype=torch.long),
        separator_positions=torch.tensor([2], dtype=torch.long),
        sequence_spans=torch.tensor([[3, 5]], dtype=torch.long),
    )


def _supervised_positions(batch) -> list[int]:
    # +1 maps a shifted label column back to its original fused position.
    return [col + 1 for col, label in enumerate(batch.labels[0].tolist()) if label != IGNORE_INDEX]


class FusedProfileEosLossTests(unittest.TestCase):
    def test_default_excludes_eos_from_loss(self) -> None:
        batch = _make_fused_batch().to_causal_lm_batch()
        # Only the two span tokens (original positions 3 and 4) are supervised.
        self.assertEqual(_supervised_positions(batch), [3, 4])

    def test_include_eos_supervises_trailing_eos(self) -> None:
        batch = _make_fused_batch().to_causal_lm_batch(include_eos_in_loss=True)
        # Span tokens plus the EOS at original position 5.
        self.assertEqual(_supervised_positions(batch), [3, 4, 5])
        eos_label_column = 5 - 1
        self.assertEqual(int(batch.labels[0, eos_label_column]), 2)

    def test_include_eos_keeps_prompt_masked(self) -> None:
        batch = _make_fused_batch().to_causal_lm_batch(include_eos_in_loss=True)
        # Positions before the span (BOS, profile, SEP) stay unsupervised.
        self.assertTrue(torch.all(batch.labels[0, :2] == IGNORE_INDEX))


if __name__ == "__main__":
    unittest.main()
