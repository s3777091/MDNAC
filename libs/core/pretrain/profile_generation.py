"""Profile-conditioned protein candidate generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from libs.core.pretrain.profiled import MDCProfileSequencePretrainArtifacts
    from libs.core.structure.candidates import GeneratedProteinCandidate


def generate_profile_conditioned_protein(
    model: torch.nn.Module,
    artifacts: MDCProfileSequencePretrainArtifacts,
    profile: str,
    *,
    device: torch.device | str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int | None = 50,
    num_candidates: int = 32,
) -> tuple[GeneratedProteinCandidate, ...]:
    import torch as _torch

    from libs.core.pretrain.profiled import (
        PROFILE_START_TOKEN,
        PROTEIN_SEQUENCE_START_TOKEN,
        TRAIN_END_TOKEN,
        TRAIN_SEPARATOR_TOKEN,
    )
    from libs.core.structure.candidates import GeneratedProteinCandidate

    # Encode the profile prompt
    profile_tokenizer = artifacts.profile_tokenizer
    sequence_tokenizer = artifacts.sequence_tokenizer
    layout = artifacts.layout

    if profile_tokenizer is None or sequence_tokenizer is None or layout is None:
        raise NotImplementedError(
            "Profile-conditioned generation requires fully initialized "
            "MDCProfileSequencePretrainArtifacts with profile_tokenizer, "
            "sequence_tokenizer, and layout. Ensure artifacts are loaded "
            "from a valid tokenizer_map.json."
        )

    # Build the fused prompt: [BOS/profile_start] profile_ids [SEP] [protein_start]
    profile_text = f"{PROFILE_START_TOKEN}{profile}{TRAIN_SEPARATOR_TOKEN}{PROTEIN_SEQUENCE_START_TOKEN}"

    try:
        prompt_ids = layout.encode_profile_prompt(profile_text)
    except AttributeError:
        raise NotImplementedError(
            "The current FusedVocabularyLayout does not expose encode_profile_prompt(). "
            "Profile-conditioned generation requires a layout method that encodes "
            "the profile prefix into fused token IDs for autoregressive decoding. "
            "TODO: Implement FusedVocabularyLayout.encode_profile_prompt() or "
            "use the ProfileSequenceBatchBuilder to construct prompts."
        )

    # Determine EOS token
    try:
        eos_token_id = layout.token_to_id(TRAIN_END_TOKEN)
    except (AttributeError, KeyError):
        eos_token_id = sequence_tokenizer.str_to_int.get(TRAIN_END_TOKEN)
        if eos_token_id is None:
            raise NotImplementedError(
                "Cannot determine EOS token ID from layout or sequence_tokenizer."
            )

    # Generate candidates
    base_model = model
    unwrap = getattr(model, "module", None)
    if unwrap is not None:
        base_model = unwrap
    base_model.eval()

    prompt_tensor = _torch.tensor(prompt_ids, dtype=_torch.long, device=device).unsqueeze(0)
    candidates: list[GeneratedProteinCandidate] = []

    with _torch.no_grad():
        for _ in range(num_candidates):
            token_ids = prompt_tensor.clone()

            for _ in range(max_new_tokens):
                logits = base_model(token_ids)
                next_logits = logits[:, -1, :]

                if temperature > 0.0:
                    next_logits = next_logits / temperature
                    if top_k is not None:
                        top_values, _ = _torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                        threshold = top_values[:, -1].unsqueeze(-1)
                        next_logits = _torch.where(
                            next_logits < threshold,
                            _torch.full_like(next_logits, float("-inf")),
                            next_logits,
                        )
                    probs = _torch.softmax(next_logits, dim=-1)
                    next_token = _torch.multinomial(probs, num_samples=1)
                else:
                    next_token = _torch.argmax(next_logits, dim=-1, keepdim=True)

                token_ids = _torch.cat([token_ids, next_token], dim=1)

                if int(next_token.item()) == eos_token_id:
                    break

            # Decode only the generated protein tokens (after the prompt)
            generated_ids = token_ids[0, prompt_tensor.size(1):].tolist()
            # Remove EOS if present
            if generated_ids and generated_ids[-1] == eos_token_id:
                generated_ids = generated_ids[:-1]

            try:
                sequence = layout.decode_sequence_ids(generated_ids)
            except AttributeError:
                sequence = sequence_tokenizer.decode(generated_ids)

            candidates.append(
                GeneratedProteinCandidate(
                    profile=profile,
                    sequence=sequence,
                    generation_score=None,
                )
            )

    return tuple(candidates)
