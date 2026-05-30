"""CLI interface - command-line entrypoints.

Each command parses args and delegates to an application use case or service.
"""

from __future__ import annotations


def build_refseq_profile_text() -> int:
    """Entrypoint: build profile-aware training text from RefSeq archives."""
    from cmd.build_refseq_profile_text import main
    return main()


def concat_text_files() -> int:
    """Entrypoint: concatenate multiple text files."""
    from cmd.concat_text_files import main
    return main()


def dedupe_refseq_profile_text() -> int:
    """Entrypoint: deduplicate RefSeq profile text artifacts."""
    from cmd.dedupe_refseq_profile_text import main
    return main()


def downsample_instruction_jsonl() -> int:
    """Entrypoint: downsample instruction JSONL with stratified sampling."""
    from cmd.downsample_instruction_jsonl import main
    return main()


def build_profile_pretrain_from_instruction_jsonl() -> int:
    """Entrypoint: build profile pretrain artifacts from instruction JSONL."""
    from cmd.build_profile_pretrain_from_instruction_jsonl import main
    return main()


def download_http_index() -> int:
    """Entrypoint: download files from HTTP directory index."""
    from cmd.download_http_index import main
    return main()
