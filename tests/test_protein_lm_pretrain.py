from __future__ import annotations

import shutil
import unittest
from pathlib import Path

import torch

from libs.core import (
    MDCDecoderModel,
    PROGEN_BACKBONE_FAMILY,
    PROGEN_PROTEIN_MODEL_FAMILY,
    build_mdc_config_from_progen_config,
    build_or_load_protein_tokenizer,
    build_progen_config,
    compute_mdc_causal_lm_loss,
    create_muon_optimizers,
    create_protein_lm_dataloader,
    generate_protein_text,
    load_protein_corpus_text,
    load_protein_pretrain_checkpoint,
    load_protein_pretrain_checkpoint_for_profile_tuning,
    save_protein_pretrain_checkpoint,
    split_protein_corpus_text,
)


class ProteinLMPretrainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("tests/artifacts/protein-lm-pretrain")
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

    def test_builds_and_reuses_self_built_protein_tokenizer(self) -> None:
        artifact = build_or_load_protein_tokenizer(self.train_path, vocab_size=64)

        self.assertTrue(artifact.rebuilt)
        self.assertTrue(artifact.tokenizer_map_path.exists())
        self.assertEqual("protein", artifact.tokenizer.sequence_type)
        self.assertEqual(
            "<|protein|>MPEPTIDE<|endoftext|>",
            artifact.tokenizer.decode(artifact.tokenizer.encode("<|protein|>MPEPTIDE<|endoftext|>")),
        )

        loaded = build_or_load_protein_tokenizer(self.train_path, vocab_size=64)
        self.assertFalse(loaded.rebuilt)
        self.assertEqual(artifact.vocab_size, loaded.vocab_size)

    def test_builds_progen_config_for_protein_mdc_model(self) -> None:
        tokenizer_artifact = build_or_load_protein_tokenizer(self.train_path, vocab_size=64)
        progen_config = build_progen_config(
            "0.8B",
            vocab_size=tokenizer_artifact.vocab_size,
            context_length=16,
            dtype=torch.float32,
        )

        self.assertEqual("ProGen/ProGen-0.8B", progen_config["model_name"])
        self.assertEqual(PROGEN_BACKBONE_FAMILY, progen_config["backbone_family"])
        self.assertEqual(1024, progen_config["emb_dim"])
        self.assertEqual(24, progen_config["n_layers"])
        self.assertTrue(bool(progen_config["qk_norm"]))

        tiny_progen_config = {
            **progen_config,
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
        }
        model_config = build_mdc_config_from_progen_config(
            tiny_progen_config,
            dtype=torch.float32,
        )

        self.assertEqual(tokenizer_artifact.vocab_size, model_config.vocab_size)
        self.assertEqual(("linear_attention", "linear_attention"), model_config.layer_types)
        self.assertEqual(8, model_config.head_dim)
        self.assertTrue(model_config.qk_norm)

    def test_rejects_unsupported_progen_2b_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown ProGen config"):
            build_progen_config("2B", vocab_size=64, context_length=16, dtype=torch.float32)

    def test_trains_one_batch_and_resumes_checkpoint(self) -> None:
        corpus = load_protein_corpus_text(self.train_path)
        train_text, _ = split_protein_corpus_text(corpus, train_ratio=0.67)
        tokenizer_artifact = build_or_load_protein_tokenizer(self.train_path, vocab_size=64)
        data_loader = create_protein_lm_dataloader(
            train_text,
            tokenizer_artifact.tokenizer,
            context_length=12,
            stride=6,
            batch_size=2,
            shuffle=False,
        )

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
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        batch = next(iter(data_loader))
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch.input_ids, attn_mask=batch.attention_mask)
        loss = compute_mdc_causal_lm_loss(logits, batch.labels)
        loss.backward()
        optimizer.step()

        self.assertTrue(torch.isfinite(loss.detach()))

        checkpoint_path = save_protein_pretrain_checkpoint(
            self.root / "checkpoint_last.pt",
            model=model,
            optimizer=optimizer,
            model_config=model_config,
            tokenizer=tokenizer_artifact.tokenizer,
            tokenizer_map_path=tokenizer_artifact.tokenizer_map_path,
            epoch=1,
            global_step=1,
            tokens_seen=int(batch.attention_mask.sum().item()),
            train_losses=[float(loss.item())],
            val_losses=[],
            training_args={"context_length": 12},
        )

        resumed_model = MDCDecoderModel(model_config)
        resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
        checkpoint = load_protein_pretrain_checkpoint(
            checkpoint_path,
            model=resumed_model,
            optimizer=resumed_optimizer,
        )

        self.assertEqual(PROGEN_PROTEIN_MODEL_FAMILY, checkpoint["model_family"])
        self.assertEqual(PROGEN_BACKBONE_FAMILY, checkpoint["backbone_family"])
        self.assertEqual(1, checkpoint["global_step"])
        self.assertEqual(str(tokenizer_artifact.tokenizer_map_path.resolve()), checkpoint["tokenizer_map_path"])

    def test_loads_protein_checkpoint_into_expanded_profile_vocab_model(self) -> None:
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
        torch.manual_seed(123)
        protein_model = MDCDecoderModel(model_config)
        protein_embedding = protein_model.tok_emb.weight.detach().clone()
        protein_head = protein_model.out_head.weight.detach().clone()
        checkpoint_path = save_protein_pretrain_checkpoint(
            self.root / "checkpoint_profile_seed.pt",
            model=protein_model,
            optimizer=None,
            model_config=model_config,
            tokenizer=tokenizer_artifact.tokenizer,
            tokenizer_map_path=tokenizer_artifact.tokenizer_map_path,
            epoch=1,
            global_step=1,
            tokens_seen=0,
            train_losses=[],
            val_losses=[],
        )

        expanded_config = model_config.with_vocab_size(tokenizer_artifact.vocab_size + 8)
        torch.manual_seed(456)
        profile_model = MDCDecoderModel(expanded_config)
        extra_embedding_before = profile_model.tok_emb.weight[tokenizer_artifact.vocab_size :].detach().clone()
        extra_head_before = profile_model.out_head.weight[tokenizer_artifact.vocab_size :].detach().clone()

        result = load_protein_pretrain_checkpoint_for_profile_tuning(
            checkpoint_path,
            model=profile_model,
        )

        self.assertEqual(tokenizer_artifact.vocab_size, result["copied_vocab_rows"])
        self.assertTrue(torch.equal(protein_embedding, profile_model.tok_emb.weight[: tokenizer_artifact.vocab_size]))
        self.assertTrue(torch.equal(protein_head, profile_model.out_head.weight[: tokenizer_artifact.vocab_size]))
        self.assertTrue(torch.equal(extra_embedding_before, profile_model.tok_emb.weight[tokenizer_artifact.vocab_size :]))
        self.assertTrue(torch.equal(extra_head_before, profile_model.out_head.weight[tokenizer_artifact.vocab_size :]))

    def test_generate_protein_text_cache_matches_uncached(self) -> None:
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
        torch.manual_seed(123)
        model = MDCDecoderModel(model_config)

        cached = generate_protein_text(
            model,
            tokenizer_artifact.tokenizer,
            device="cpu",
            max_new_tokens=3,
            context_length=12,
            use_cache=True,
            stop_at_endoftext=False,
        )
        uncached = generate_protein_text(
            model,
            tokenizer_artifact.tokenizer,
            device="cpu",
            max_new_tokens=3,
            context_length=12,
            use_cache=False,
            stop_at_endoftext=False,
        )

        self.assertEqual(uncached, cached)

    def test_create_muon_optimizers_groups_matrix_params(self) -> None:
        class FakeMuon(torch.optim.SGD):
            def __init__(self, params, *, lr, weight_decay=0.0, adjust_lr_fn=None):
                self.adjust_lr_fn = adjust_lr_fn
                super().__init__(params, lr=lr, weight_decay=weight_decay)

        original_muon = getattr(torch.optim, "Muon", None)
        torch.optim.Muon = FakeMuon
        try:
            model = torch.nn.Sequential(
                torch.nn.Embedding(8, 4),
                torch.nn.Linear(4, 4),
                torch.nn.LayerNorm(4),
            )
            optimizers = create_muon_optimizers(
                model,
                adamw_learning_rate=1e-3,
                muon_learning_rate=2e-3,
                weight_decay=0.01,
            )
        finally:
            if original_muon is None:
                delattr(torch.optim, "Muon")
            else:
                torch.optim.Muon = original_muon

        self.assertEqual(2, len(optimizers))
        self.assertIsInstance(optimizers[0], FakeMuon)
        self.assertIsInstance(optimizers[1], torch.optim.AdamW)
        self.assertEqual("match_rms_adamw", optimizers[0].adjust_lr_fn)

        muon_param_ids = {
            id(parameter)
            for group in optimizers[0].param_groups
            for parameter in group["params"]
        }
        adamw_param_ids = {
            id(parameter)
            for group in optimizers[1].param_groups
            for parameter in group["params"]
        }

        self.assertIn(id(model[1].weight), muon_param_ids)
        self.assertNotIn(id(model[0].weight), muon_param_ids)
        self.assertIn(id(model[0].weight), adamw_param_ids)

    def test_create_muon_optimizers_requires_native_torch_muon(self) -> None:
        original_muon = getattr(torch.optim, "Muon", None)
        if original_muon is not None:
            delattr(torch.optim, "Muon")
        try:
            model = torch.nn.Sequential(
                torch.nn.Embedding(8, 4),
                torch.nn.Linear(4, 4),
            )
            with self.assertRaisesRegex(RuntimeError, "torch.optim.Muon is required"):
                create_muon_optimizers(
                    model,
                    adamw_learning_rate=1e-3,
                    weight_decay=0.01,
                )
        finally:
            if original_muon is not None:
                torch.optim.Muon = original_muon

if __name__ == "__main__":
    unittest.main()
