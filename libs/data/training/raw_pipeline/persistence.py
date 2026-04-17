from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch

from libs.data.training.kmer import KmerTokenizer
from libs.data.training.profile_tokenizer import ProfileBPETokenizer
from libs.data.training.raw_pipeline.helpers import slugify
from libs.data.training.raw_pipeline.models import ProfileSequencePair, RawTensorDatasetArtifact


def persist_pairs(
    dataset_name: str,
    pairs: Sequence[ProfileSequencePair],
    output_dir: Path,
    sequence_type: str,
    kmer_size: int,
    profile_vocab_size: int,
    max_profile_length: int | None,
    max_sequence_length: int | None,
) -> RawTensorDatasetArtifact:
    if not pairs:
        raise ValueError("No profile/sequence pairs were produced from the provided raw data.")

    output_dir.mkdir(parents=True, exist_ok=True)

    pairs_path = output_dir / "pairs.jsonl"
    profile_tokenizer_path = output_dir / "profile_tokenizer.json"
    sequence_tokenizer_path = output_dir / "sequence_tokenizer.json"
    tensor_dataset_path = output_dir / "dataset.pt"
    manifest_path = output_dir / "manifest.json"

    profile_corpus = "\n".join(pair.profile for pair in pairs) + "\n"
    profile_tokenizer = ProfileBPETokenizer.from_text(profile_corpus, vocab_size=profile_vocab_size)
    sequence_tokenizer = KmerTokenizer.from_sequences(
        (pair.sequence for pair in pairs),
        kmer_size=kmer_size,
        sequence_type=sequence_type,
    )

    profile_encoded = [
        profile_tokenizer.encode(pair.profile, add_bos=True, add_eos=True)
        for pair in pairs
    ]
    sequence_encoded = [
        sequence_tokenizer.encode(pair.sequence, add_bos=True, add_eos=True)
        for pair in pairs
    ]

    profile_input_ids, profile_attention_mask, profile_lengths = pad_encoded(
        profile_encoded,
        pad_token_id=profile_tokenizer.pad_token_id,
        max_length=max_profile_length,
    )
    sequence_input_ids, sequence_attention_mask, sequence_lengths = pad_encoded(
        sequence_encoded,
        pad_token_id=sequence_tokenizer.pad_token_id,
        max_length=max_sequence_length,
    )

    pairs_path.write_text(
        "\n".join(json.dumps(pair.to_dict(), ensure_ascii=False) for pair in pairs) + "\n",
        encoding="utf-8",
    )
    profile_tokenizer.save_map(profile_tokenizer_path)
    sequence_tokenizer.save_map(sequence_tokenizer_path)

    dataset_slug = slugify(dataset_name)
    tensor_payload = {
        "profile_input_ids": profile_input_ids,
        "profile_attention_mask": profile_attention_mask,
        "profile_lengths": profile_lengths,
        "sequence_input_ids": sequence_input_ids,
        "sequence_attention_mask": sequence_attention_mask,
        "sequence_lengths": sequence_lengths,
        "metadata": [pair.to_dict() for pair in pairs],
        "config": {
            "dataset_name": dataset_slug,
            "sequence_type": sequence_type,
            "kmer_size": kmer_size,
            "profile_vocab_size": profile_tokenizer.vocab_size,
            "sequence_vocab_size": sequence_tokenizer.vocab_size,
        },
    }
    torch.save(tensor_payload, tensor_dataset_path)

    manifest = {
        "dataset_name": dataset_slug,
        "pair_count": len(pairs),
        "pairs_path": str(pairs_path),
        "tensor_dataset_path": str(tensor_dataset_path),
        "profile_tokenizer_path": str(profile_tokenizer_path),
        "sequence_tokenizer_path": str(sequence_tokenizer_path),
        "kmer_size": kmer_size,
        "profile_vocab_size": profile_tokenizer.vocab_size,
        "sequence_vocab_size": sequence_tokenizer.vocab_size,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return RawTensorDatasetArtifact(
        dataset_name=dataset_slug,
        output_dir=str(output_dir),
        pair_count=len(pairs),
        pairs_path=str(pairs_path),
        tensor_dataset_path=str(tensor_dataset_path),
        profile_tokenizer_path=str(profile_tokenizer_path),
        sequence_tokenizer_path=str(sequence_tokenizer_path),
        manifest_path=str(manifest_path),
        kmer_size=kmer_size,
        profile_vocab_size=profile_tokenizer.vocab_size,
        sequence_vocab_size=sequence_tokenizer.vocab_size,
    )


def pad_encoded(
    encoded_sequences: Sequence[Sequence[int]],
    pad_token_id: int,
    max_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not encoded_sequences:
        empty = torch.zeros((0, 0), dtype=torch.long)
        lengths = torch.zeros((0,), dtype=torch.long)
        return empty, empty, lengths

    target_length = max_length or max(len(sequence) for sequence in encoded_sequences)
    input_ids = torch.full((len(encoded_sequences), target_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(encoded_sequences), target_length), dtype=torch.long)
    lengths: list[int] = []

    for row_index, sequence in enumerate(encoded_sequences):
        truncated = list(sequence[:target_length])
        sequence_length = len(truncated)
        lengths.append(sequence_length)
        if sequence_length == 0:
            continue
        input_ids[row_index, :sequence_length] = torch.tensor(truncated, dtype=torch.long)
        attention_mask[row_index, :sequence_length] = 1

    return input_ids, attention_mask, torch.tensor(lengths, dtype=torch.long)
