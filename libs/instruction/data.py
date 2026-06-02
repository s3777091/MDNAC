from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from libs.core.pretrain.distributed import (
    partition_items_for_worker,
    resolve_mdc_distributed_context,
)
from libs.core.pretrain.profiled import (
    MDCEncodedProfileSequenceExample,
    MDCProfileSequenceBatchCollator,
    MDCProfileSequencePretrainArtifacts,
)

from .schema import (
    belongs_to_split,
    instruction_record_from_payload,
    resolve_instruction_paths,
)


class InstructionJsonlStreamingDataset(IterableDataset[MDCEncodedProfileSequenceExample]):
    def __init__(
        self,
        artifacts: MDCProfileSequencePretrainArtifacts,
        paths: str | Path | Sequence[str | Path],
        *,
        split: str,
        train_ratio: float = 0.95,
        split_seed: int = 42,
        default_sequence_type: str = "protein",
        instruction_field: str = "instruction",
        input_field: str = "input",
        output_field: str = "output",
        max_sequence_length: int | None = None,
        shuffle_files: bool = True,
        shuffle_records: bool = True,
        shuffle_buffer_size: int = 2048,
        seed: int = 123,
        distributed: bool | None = None,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("split must be one of: 'train', 'val'.")
        if not 0.0 < float(train_ratio) < 1.0:
            raise ValueError("train_ratio must be between 0 and 1.")
        if shuffle_buffer_size <= 0:
            raise ValueError("shuffle_buffer_size must be greater than 0.")

        self.artifacts = artifacts
        self.paths = resolve_instruction_paths(paths)
        self.split = split
        self.train_ratio = float(train_ratio)
        self.split_seed = int(split_seed)
        self.default_sequence_type = default_sequence_type
        self.instruction_field = instruction_field
        self.input_field = input_field
        self.output_field = output_field
        self.max_sequence_length = max_sequence_length
        self.shuffle_files = bool(shuffle_files)
        self.shuffle_records = bool(shuffle_records)
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.seed = int(seed)
        self.epoch = 0

        resolved_rank, _, resolved_world_size = resolve_mdc_distributed_context(
            rank=rank,
            world_size=world_size,
        )
        self.use_distributed = bool(distributed) if distributed is not None else resolved_world_size > 1
        self.rank = resolved_rank if self.use_distributed else 0
        self.world_size = resolved_world_size if self.use_distributed else 1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        worker_id, num_workers = _worker_context()
        _, partition_index = partition_items_for_worker(
            [None],
            rank=self.rank,
            world_size=self.world_size,
            worker_id=worker_id,
            num_workers=num_workers,
        )
        partition_count = max(1, self.world_size * num_workers)
        paths = list(self.paths)
        rng = random.Random(self.seed + self.epoch * 1000003 + partition_index)
        if self.shuffle_files:
            rng.shuffle(paths)

        examples = self._iter_partitioned_examples(
            paths,
            partition_index=partition_index,
            partition_count=partition_count,
        )
        if self.shuffle_records:
            examples = _iter_bounded_shuffle(
                examples,
                rng=rng,
                buffer_size=self.shuffle_buffer_size,
            )
        yield from examples

    def _iter_partitioned_examples(
        self,
        paths: Sequence[Path],
        *,
        partition_index: int,
        partition_count: int,
    ):
        row_index = -1
        for path in paths:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    if not raw_line.strip():
                        continue
                    row_index += 1
                    if row_index % partition_count != partition_index:
                        continue
                    payload = _load_jsonl_payload(path, line_number, raw_line)
                    if not belongs_to_split(
                        payload,
                        split=self.split,
                        train_ratio=self.train_ratio,
                        split_seed=self.split_seed,
                        fallback_key=f"{path}:{line_number}",
                    ):
                        continue
                    record = instruction_record_from_payload(
                        payload,
                        default_sequence_type=self.default_sequence_type,
                        instruction_field=self.instruction_field,
                        input_field=self.input_field,
                        output_field=self.output_field,
                    )
                    encoded = self.artifacts.encode_record(record)
                    if self.max_sequence_length is not None:
                        fused = self.artifacts.build_fused_batch([encoded])
                        if fused.token_ids.size(1) > int(self.max_sequence_length):
                            continue
                    yield encoded


def create_instruction_dataloader(
    artifacts: MDCProfileSequencePretrainArtifacts,
    paths: str | Path | Sequence[str | Path],
    *,
    split: str,
    train_ratio: float = 0.95,
    split_seed: int = 42,
    default_sequence_type: str = "protein",
    instruction_field: str = "instruction",
    input_field: str = "input",
    output_field: str = "output",
    max_sequence_length: int | None = None,
    batch_size: int = 2,
    drop_last: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    shuffle_files: bool = True,
    shuffle_records: bool = True,
    shuffle_buffer_size: int = 2048,
    seed: int = 123,
    distributed: bool | None = None,
    rank: int | None = None,
    world_size: int | None = None,
    train_on_prompt: bool = False,
    include_separator_in_loss: bool = False,
) -> DataLoader:
    dataset = InstructionJsonlStreamingDataset(
        artifacts,
        paths,
        split=split,
        train_ratio=train_ratio,
        split_seed=split_seed,
        default_sequence_type=default_sequence_type,
        instruction_field=instruction_field,
        input_field=input_field,
        output_field=output_field,
        max_sequence_length=max_sequence_length,
        shuffle_files=shuffle_files,
        shuffle_records=shuffle_records,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )
    collator = MDCProfileSequenceBatchCollator(
        artifacts,
        train_on_prompt=train_on_prompt,
        include_separator_in_loss=include_separator_in_loss,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
    )


def count_instruction_split_records(
    paths: str | Path | Sequence[str | Path],
    *,
    split: str,
    train_ratio: float = 0.95,
    split_seed: int = 42,
    artifacts: MDCProfileSequencePretrainArtifacts | None = None,
    default_sequence_type: str = "protein",
    instruction_field: str = "instruction",
    input_field: str = "input",
    output_field: str = "output",
    max_sequence_length: int | None = None,
) -> int:
    count = 0
    for path in resolve_instruction_paths(paths):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                payload = _load_jsonl_payload(path, line_number, raw_line)
                if belongs_to_split(
                    payload,
                    split=split,
                    train_ratio=train_ratio,
                    split_seed=split_seed,
                    fallback_key=f"{path}:{line_number}",
                ):
                    if artifacts is not None and max_sequence_length is not None:
                        record = instruction_record_from_payload(
                            payload,
                            default_sequence_type=default_sequence_type,
                            instruction_field=instruction_field,
                            input_field=input_field,
                            output_field=output_field,
                        )
                        encoded = artifacts.encode_record(record)
                        fused = artifacts.build_fused_batch([encoded])
                        if fused.token_ids.size(1) > int(max_sequence_length):
                            continue
                    count += 1
    return count


def _load_jsonl_payload(path: Path, line_number: int, raw_line: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}:{line_number}.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Instruction JSONL row must be an object at {path}:{line_number}.")
    return payload


def _worker_context() -> tuple[int, int]:
    worker_info = get_worker_info()
    if worker_info is None:
        return 0, 1
    return int(worker_info.id), int(worker_info.num_workers)


def _iter_bounded_shuffle(
    examples,
    *,
    rng: random.Random,
    buffer_size: int,
):
    buffer: list[MDCEncodedProfileSequenceExample] = []
    for example in examples:
        buffer.append(example)
        if len(buffer) >= buffer_size:
            index = rng.randrange(len(buffer))
            buffer[index], buffer[-1] = buffer[-1], buffer[index]
            yield buffer.pop()

    while buffer:
        index = rng.randrange(len(buffer))
        buffer[index], buffer[-1] = buffer[-1], buffer[index]
        yield buffer.pop()
