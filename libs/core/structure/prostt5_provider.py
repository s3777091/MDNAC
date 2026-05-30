"""ProstT5 model wrapper for AA-to-3Di prediction."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence

from .instruction_3di import (
    AA_TO_3DI_PREFIX,
    DEFAULT_PROSTT5_MODEL_NAME,
    normalize_3di_structure,
    normalize_prostt5_aa_sequence,
)


def _default_prostt5_generation_kwargs() -> dict[str, object]:
    return {
        "do_sample": False,
        "num_beams": 3,
        "repetition_penalty": 1.2,
    }


class ProstT5Structure3DiProvider:
    """AA-to-3Di provider backed by the optional Rostlab/ProstT5 model."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_PROSTT5_MODEL_NAME,
        device: str | None = None,
        use_half: bool | None = None,
        generation_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.use_half = use_half
        self.generation_kwargs = dict(generation_kwargs or _default_prostt5_generation_kwargs())
        self._torch = None
        self._tokenizer = None
        self._model = None
        self._resolved_device = None

    @property
    def resolved_device(self) -> str | None:
        return str(self._resolved_device) if self._resolved_device is not None else self.device

    def predict_3di_batch(self, sequences: Sequence[str]) -> Sequence[str]:
        if not sequences:
            return ()

        self._load()
        assert self._torch is not None
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._resolved_device is not None

        normalized_sequences = [normalize_prostt5_aa_sequence(sequence) for sequence in sequences]

        # Run the batch prediction.
        structures = self._generate_3di(normalized_sequences)

        # Validate each result; retry individually for large mismatches.
        final: list[str] = []
        for i, (seq, structure_3di) in enumerate(zip(normalized_sequences, structures)):
            delta = len(structure_3di) - len(seq) if structure_3di else -len(seq)

            if structure_3di and abs(delta) <= 1:
                # Good result (exact or ±1 off-by-one). Trim to match.
                if delta != 0:
                    warnings.warn(
                        f"ProstT5 length mismatch: predicted {len(structure_3di)} tokens "
                        f"for a {len(seq)}-residue sequence. Trimming to match.",
                        stacklevel=2,
                    )
                final.append(structure_3di[: len(seq)])
                continue

            # Large mismatch or empty — retry this sequence individually.
            warnings.warn(
                f"ProstT5 batch prediction mismatched for sequence index {i} "
                f"(predicted {len(structure_3di) if structure_3di else 0} vs "
                f"{len(seq)} residues). Retrying individually.",
                stacklevel=2,
            )
            retry = self._generate_3di([seq])
            retry_3di = retry[0] if retry else ""
            retry_delta = len(retry_3di) - len(seq) if retry_3di else -len(seq)

            if retry_3di and abs(retry_delta) <= 1:
                if retry_delta != 0:
                    warnings.warn(
                        f"ProstT5 individual retry trimmed: {len(retry_3di)} → {len(seq)}.",
                        stacklevel=2,
                    )
                final.append(retry_3di[: len(seq)])
            elif retry_3di:
                warnings.warn(
                    f"ProstT5 individual retry still mismatched "
                    f"(predicted {len(retry_3di)} vs {len(seq)} residues). "
                    f"Skipping 3Di for this sequence.",
                    stacklevel=2,
                )
                final.append("")
            else:
                warnings.warn(
                    f"ProstT5 returned an empty 3Di prediction for a "
                    f"{len(seq)}-residue sequence. Skipping.",
                    stacklevel=2,
                )
                final.append("")

        return tuple(final)

    def _generate_3di(self, normalized_sequences: list[str]) -> list[str]:
        """Low-level generation: encode → generate → decode → normalize."""
        assert self._torch is not None
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._resolved_device is not None

        prepared_inputs = [
            f"{AA_TO_3DI_PREFIX} {' '.join(sequence)}"
            for sequence in normalized_sequences
        ]
        lengths = [len(sequence) for sequence in normalized_sequences]

        encoded = self._tokenizer(
            prepared_inputs,
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt",
        )
        encoded = {key: value.to(self._resolved_device) for key, value in encoded.items()}

        with self._torch.inference_mode():
            generated = self._model.generate(
                encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_new_tokens=max(lengths),
                num_return_sequences=1,
                **self.generation_kwargs,
            )

        decoded = self._tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [normalize_3di_structure(value) for value in decoded]

    def _load(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "ProstT5Structure3DiProvider requires optional dependencies: "
                "transformers and sentencepiece. Install them in the active environment before running 3Di annotation."
            ) from exc

        resolved_device = torch.device(self.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, do_lower_case=False, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(resolved_device)

        use_half = self.use_half
        if use_half is None:
            use_half = resolved_device.type == "cuda"
        if use_half and resolved_device.type != "cpu":
            model.half()
        else:
            model.float()
        model.eval()

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._resolved_device = resolved_device
