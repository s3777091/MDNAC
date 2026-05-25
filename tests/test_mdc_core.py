from __future__ import annotations

import unittest

import torch

from libs.core import (
    IGNORE_INDEX,
    MDCDecoderCache,
    MicrobialDecoderCoreApp,
    ProfileSequenceBatchBuilder,
    build_mdc_tiny_config,
)
from libs.core.mdc import MDCDecoderModel
from models.qwen3_5.modeling import KVCache, Qwen3_5Model as ReferenceQwen3_5Model


class ProfileSequenceBatchBuilderTests(unittest.TestCase):
    def test_fuses_profile_and_sequence_modalities_into_one_decoder_sequence(self) -> None:
        payload = {
            "profile_input_ids": torch.tensor(
                [
                    [1, 10, 11, 2, 0],
                    [1, 12, 2, 0, 0],
                ],
                dtype=torch.long,
            ),
            "profile_attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 0, 0],
                ],
                dtype=torch.long,
            ),
            "sequence_input_ids": torch.tensor(
                [
                    [1, 5, 6, 2, 0],
                    [1, 7, 2, 0, 0],
                ],
                dtype=torch.long,
            ),
            "sequence_attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 0, 0],
                ],
                dtype=torch.long,
            ),
            "config": {
                "profile_vocab_size": 32,
                "sequence_vocab_size": 16,
            },
        }

        builder = ProfileSequenceBatchBuilder.from_raw_tensor_payload(payload)
        batch = builder.build_from_raw_tensor_payload(payload)
        layout = builder.layout

        expected_first = torch.tensor(
            [
                layout.bos_token_id,
                layout.profile_offset + 10,
                layout.profile_offset + 11,
                layout.sep_token_id,
                layout.sequence_offset + 5,
                layout.sequence_offset + 6,
                layout.eos_token_id,
            ],
            dtype=torch.long,
        )
        expected_second = torch.tensor(
            [
                layout.bos_token_id,
                layout.profile_offset + 12,
                layout.sep_token_id,
                layout.sequence_offset + 7,
                layout.eos_token_id,
                layout.pad_token_id,
                layout.pad_token_id,
            ],
            dtype=torch.long,
        )

        self.assertTrue(torch.equal(batch.token_ids[0], expected_first))
        self.assertTrue(torch.equal(batch.token_ids[1], expected_second))
        self.assertTrue(torch.equal(batch.attention_mask[0], torch.tensor([1, 1, 1, 1, 1, 1, 1])))
        self.assertTrue(torch.equal(batch.attention_mask[1], torch.tensor([1, 1, 1, 1, 1, 0, 0])))
        self.assertTrue(torch.equal(batch.profile_spans[0], torch.tensor([1, 3])))
        self.assertEqual(3, int(batch.separator_positions[0]))
        self.assertTrue(torch.equal(batch.sequence_spans[0], torch.tensor([4, 6])))

    def test_builds_sequence_only_labels_for_causal_language_model_training(self) -> None:
        payload = {
            "profile_input_ids": torch.tensor([[1, 10, 11, 2]], dtype=torch.long),
            "profile_attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
            "sequence_input_ids": torch.tensor([[1, 5, 6, 2]], dtype=torch.long),
            "sequence_attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
            "config": {
                "profile_vocab_size": 32,
                "sequence_vocab_size": 16,
            },
        }

        batch = ProfileSequenceBatchBuilder.from_raw_tensor_payload(payload).build_from_raw_tensor_payload(payload)
        causal_batch = batch.to_causal_lm_batch(train_on_prompt=False, include_separator_in_loss=False)

        expected_labels = torch.tensor(
            [
                [
                    IGNORE_INDEX,
                    IGNORE_INDEX,
                    IGNORE_INDEX,
                    batch.token_ids[0, 4],
                    batch.token_ids[0, 5],
                    batch.token_ids[0, 6],
                ]
            ],
            dtype=torch.long,
        )

        self.assertTrue(torch.equal(causal_batch.input_ids, batch.token_ids[:, :-1]))
        self.assertTrue(torch.equal(causal_batch.attention_mask, batch.attention_mask[:, :-1]))
        self.assertTrue(torch.equal(causal_batch.labels, expected_labels))


class MDCDecoderModelTests(unittest.TestCase):
    def test_reference_qwen_kv_cache_matches_full_forward(self) -> None:
        cfg = {
            "vocab_size": 97,
            "context_length": 16,
            "emb_dim": 32,
            "n_heads": 4,
            "n_layers": 2,
            "hidden_dim": 64,
            "head_dim": 8,
            "qk_norm": False,
            "n_kv_groups": 2,
            "rope_base": 10_000.0,
            "partial_rotary_factor": 1.0,
            "rms_norm_eps": 1e-6,
            "linear_conv_kernel_dim": 2,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "dtype": torch.float32,
        }

        torch.manual_seed(321)
        model = ReferenceQwen3_5Model(cfg)
        model.eval()
        token_ids = torch.randint(0, cfg["vocab_size"], (1, 6), dtype=torch.long)

        full_logits = model(token_ids)
        cache = KVCache(n_layers=cfg["n_layers"])
        model.reset_kv_cache(cache)
        cached_logits = torch.cat(
            [model(token_ids[:, pos : pos + 1], cache=cache) for pos in range(token_ids.size(1))],
            dim=1,
        )

        torch.testing.assert_close(cached_logits, full_logits, atol=1e-5, rtol=1e-4)
        cached_k, _ = cache.get(1)
        self.assertEqual((1, cfg["n_kv_groups"], token_ids.size(1), cfg["head_dim"]), tuple(cached_k.shape))

    def test_mdc_kv_cache_matches_full_forward(self) -> None:
        cfg = {
            "vocab_size": 97,
            "context_length": 16,
            "emb_dim": 32,
            "n_heads": 4,
            "n_layers": 2,
            "hidden_dim": 64,
            "head_dim": 8,
            "qk_norm": False,
            "n_kv_groups": 2,
            "rope_base": 10_000.0,
            "partial_rotary_factor": 1.0,
            "rms_norm_eps": 1e-6,
            "linear_conv_kernel_dim": 2,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "dtype": torch.float32,
        }

        torch.manual_seed(654)
        model = MDCDecoderModel(cfg)
        model.eval()
        token_ids = torch.randint(0, cfg["vocab_size"], (1, 6), dtype=torch.long)

        full_logits = model(token_ids)
        cache = MDCDecoderCache(n_layers=cfg["n_layers"])
        model.reset_kv_cache(cache)
        cached_logits = torch.cat(
            [model(token_ids[:, pos : pos + 1], cache=cache) for pos in range(token_ids.size(1))],
            dim=1,
        )

        torch.testing.assert_close(cached_logits, full_logits, atol=1e-5, rtol=1e-4)
        cached_k, _ = cache.get(1)
        self.assertEqual((1, cfg["n_kv_groups"], token_ids.size(1), cfg["head_dim"]), tuple(cached_k.shape))

    def test_matches_reference_qwen_core_forward(self) -> None:
        cfg = {
            "vocab_size": 97,
            "context_length": 16,
            "emb_dim": 32,
            "n_heads": 4,
            "n_layers": 2,
            "hidden_dim": 64,
            "head_dim": 8,
            "qk_norm": False,
            "n_kv_groups": 2,
            "rope_base": 10_000.0,
            "partial_rotary_factor": 1.0,
            "rms_norm_eps": 1e-6,
            "linear_conv_kernel_dim": 2,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "dtype": torch.float32,
        }

        torch.manual_seed(123)
        reference_model = ReferenceQwen3_5Model(cfg)
        mdc_model = MDCDecoderModel(cfg)
        mdc_model.load_state_dict(reference_model.state_dict())

        token_ids = torch.randint(0, cfg["vocab_size"], (2, 8), dtype=torch.long)
        attention_mask = torch.ones_like(token_ids)

        reference_logits = reference_model(token_ids, attn_mask=attention_mask)
        mdc_logits = mdc_model(token_ids, attn_mask=attention_mask)
        torch.testing.assert_close(mdc_logits, reference_logits)

    def test_core_app_runs_end_to_end_from_raw_tensor_payload(self) -> None:
        payload = {
            "profile_input_ids": torch.tensor([[1, 10, 11, 2]], dtype=torch.long),
            "profile_attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
            "sequence_input_ids": torch.tensor([[1, 5, 6, 2]], dtype=torch.long),
            "sequence_attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
            "config": {
                "profile_vocab_size": 32,
                "sequence_vocab_size": 16,
            },
        }

        app = MicrobialDecoderCoreApp.from_raw_tensor_payload(
            payload,
            model_config=build_mdc_tiny_config(vocab_size=1),
        )

        logits, batch = app.forward_from_raw_tensor_payload(payload)

        self.assertEqual((1, batch.token_ids.size(1), app.layout.vocab_size), tuple(logits.shape))
        self.assertEqual(app.layout.vocab_size, app.model.cfg["vocab_size"])


if __name__ == "__main__":
    unittest.main()
