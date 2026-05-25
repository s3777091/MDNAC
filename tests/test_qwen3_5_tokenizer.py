from __future__ import annotations

import unittest

from models.qwen3_5.instruction import build_instruction_generation_tokenizer_settings
from models.qwen3_5.modeling import Qwen3_5Tokenizer


class Qwen3_5TokenizerTests(unittest.TestCase):
    def test_instruction_generation_settings_do_not_request_thinking_tags(self) -> None:
        settings = build_instruction_generation_tokenizer_settings()

        self.assertEqual(True, settings["apply_chat_template"])
        self.assertEqual(True, settings["add_generation_prompt"])
        self.assertNotIn("add_thinking", settings)
        self.assertNotIn("thinking_template", settings)

    def test_chat_template_never_inserts_think_tags(self) -> None:
        tokenizer = object.__new__(Qwen3_5Tokenizer)
        tokenizer.add_generation_prompt = True
        tokenizer.add_thinking = True
        tokenizer.thinking_template = "tagged"

        wrapped = tokenizer._wrap_chat("Create a protein sequence for kinase activity.")

        self.assertIn("<|im_start|>assistant\n", wrapped)
        self.assertNotIn("<think>", wrapped)
        self.assertNotIn("</think>", wrapped)

    def test_think_tags_are_not_registered_as_model_specials(self) -> None:
        self.assertNotIn("<think>", Qwen3_5Tokenizer._SPECIALS)
        self.assertNotIn("</think>", Qwen3_5Tokenizer._SPECIALS)


if __name__ == "__main__":
    unittest.main()
