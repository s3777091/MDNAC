from __future__ import annotations

DEFAULT_SPECIAL_TOKENS = (
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|endoftext|>",
    "<|protein|>",
)

DEFAULT_VOCAB_SIZES = {
    "protein": 256,
}


PROTEIN_AMINO_ACIDS = (
    "\n", "A", "C", "D", "E", "F", "G", "H", "I", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y", "X",
)


def base_tokens(sequence_type: str) -> tuple[str, ...]:
    if sequence_type != "protein":
        raise ValueError("SequenceTokenizer only supports protein vocabularies.")
    return PROTEIN_AMINO_ACIDS
