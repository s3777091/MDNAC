from __future__ import annotations

import os
import platform
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeVar

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


MDCMultiGPUMode = Literal["auto", "none", "data_parallel", "ddp"]

_T = TypeVar("_T")
_STATE_DICT_WRAPPER_PREFIXES = ("module.", "_orig_mod.")


@dataclass(slots=True)
class MDCTrainingRuntime:
    model: torch.nn.Module
    device: torch.device
    distributed: bool
    data_parallel: bool
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def prepare_mdc_training_runtime(
    model: torch.nn.Module,
    *,
    device: torch.device | str | None = None,
    multi_gpu: MDCMultiGPUMode | str = "auto",
    rank: int | None = None,
    local_rank: int | None = None,
    world_size: int | None = None,
    backend: str | None = None,
    find_unused_parameters: bool = False,
    data_parallel_device_ids: Sequence[int] | None = None,
) -> MDCTrainingRuntime:
    resolved_rank, resolved_local_rank, resolved_world_size = resolve_mdc_distributed_context(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )
    resolved_mode = _normalize_multi_gpu_mode(multi_gpu)
    resolved_device = resolve_mdc_training_device(
        device,
        local_rank=resolved_local_rank if resolved_world_size > 1 else None,
    )

    if resolved_mode in {"auto", "ddp"} and resolved_world_size > 1:
        if not torch.distributed.is_available():
            raise RuntimeError("torch.distributed is not available in this PyTorch build.")

        if resolved_device.type == "cuda":
            torch.cuda.set_device(resolved_device.index if resolved_device.index is not None else resolved_local_rank)

        if not torch.distributed.is_initialized():
            if "MASTER_ADDR" not in os.environ:
                os.environ["MASTER_ADDR"] = "localhost"
            if "MASTER_PORT" not in os.environ:
                os.environ["MASTER_PORT"] = "12355"

            resolved_backend = backend or _default_process_group_backend(resolved_device)
            torch.distributed.init_process_group(
                backend=resolved_backend,
                rank=resolved_rank,
                world_size=resolved_world_size,
            )

        model = model.to(resolved_device)
        device_ids = None
        if resolved_device.type == "cuda" and resolved_device.index is not None:
            device_ids = [resolved_device.index]

        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=device_ids,
            find_unused_parameters=find_unused_parameters,
        )
        return MDCTrainingRuntime(
            model=model,
            device=resolved_device,
            distributed=True,
            data_parallel=False,
            rank=resolved_rank,
            local_rank=resolved_local_rank,
            world_size=resolved_world_size,
        )

    model = model.to(resolved_device)

    if (
        resolved_mode in {"auto", "data_parallel"}
        and resolved_world_size == 1
        and resolved_device.type == "cuda"
        and torch.cuda.device_count() > 1
    ):
        resolved_device_ids = _resolve_data_parallel_device_ids(
            resolved_device,
            data_parallel_device_ids=data_parallel_device_ids,
        )
        primary_device = torch.device("cuda", resolved_device_ids[0])
        if primary_device != resolved_device:
            model = model.to(primary_device)
            resolved_device = primary_device

        model = torch.nn.DataParallel(
            model,
            device_ids=list(resolved_device_ids),
            output_device=resolved_device_ids[0],
        )
        return MDCTrainingRuntime(
            model=model,
            device=resolved_device,
            distributed=False,
            data_parallel=True,
            rank=resolved_rank,
            local_rank=resolved_local_rank,
            world_size=resolved_world_size,
        )

    return MDCTrainingRuntime(
        model=model,
        device=resolved_device,
        distributed=False,
        data_parallel=False,
        rank=resolved_rank,
        local_rank=resolved_local_rank,
        world_size=resolved_world_size,
    )


def cleanup_mdc_distributed_training() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def create_mdc_distributed_sampler(
    dataset: Dataset[object],
    *,
    shuffle: bool,
    distributed: bool | None = None,
    rank: int | None = None,
    world_size: int | None = None,
    seed: int = 0,
    drop_last: bool = False,
) -> DistributedSampler[object] | None:
    resolved_rank, _, resolved_world_size = resolve_mdc_distributed_context(
        rank=rank,
        world_size=world_size,
    )
    use_distributed = bool(distributed) if distributed is not None else resolved_world_size > 1
    if not use_distributed or resolved_world_size <= 1:
        return None

    return DistributedSampler(
        dataset,
        num_replicas=resolved_world_size,
        rank=resolved_rank,
        shuffle=shuffle,
        seed=int(seed),
        drop_last=drop_last,
    )


def normalize_parallel_state_dict(state_dict: dict[str, object]) -> dict[str, object]:
    normalized = dict(state_dict)
    while normalized:
        stripped_prefix = None
        for prefix in _STATE_DICT_WRAPPER_PREFIXES:
            if all(key.startswith(prefix) for key in normalized):
                stripped_prefix = prefix
                break

        if stripped_prefix is None:
            break

        normalized = {
            key[len(stripped_prefix):]: value
            for key, value in normalized.items()
        }
    return normalized


def partition_items_for_worker(
    items: Sequence[_T],
    *,
    rank: int | None = None,
    world_size: int | None = None,
    worker_id: int | None = None,
    num_workers: int | None = None,
) -> tuple[list[_T], int]:
    resolved_rank, _, resolved_world_size = resolve_mdc_distributed_context(
        rank=rank,
        world_size=world_size,
    )
    resolved_num_workers = int(num_workers or 1)
    resolved_worker_id = int(worker_id or 0)
    if resolved_num_workers < 1:
        raise ValueError("num_workers must be greater than 0.")
    if resolved_worker_id < 0 or resolved_worker_id >= resolved_num_workers:
        raise ValueError("worker_id must be within [0, num_workers).")

    partition_count = resolved_world_size * resolved_num_workers
    partition_index = resolved_rank * resolved_num_workers + resolved_worker_id
    return list(items[partition_index::partition_count]), partition_index


def resolve_mdc_distributed_context(
    *,
    rank: int | None = None,
    local_rank: int | None = None,
    world_size: int | None = None,
) -> tuple[int, int, int]:
    resolved_world_size = int(world_size if world_size is not None else os.environ.get("WORLD_SIZE", "1"))
    if resolved_world_size < 1:
        raise ValueError("world_size must be greater than 0.")

    if local_rank is None:
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
        elif rank is not None:
            local_rank = int(rank)
        else:
            local_rank = 0

    if rank is None:
        if "RANK" in os.environ:
            rank = int(os.environ["RANK"])
        else:
            rank = int(local_rank)

    return int(rank), int(local_rank), resolved_world_size


def resolve_mdc_training_device(
    device: torch.device | str | None = None,
    *,
    local_rank: int | None = None,
) -> torch.device:
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda", 0 if local_rank is None else int(local_rank))

        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    resolved_device = device if isinstance(device, torch.device) else torch.device(device)
    if resolved_device.type == "cuda" and resolved_device.index is None and local_rank is not None:
        return torch.device("cuda", int(local_rank))
    return resolved_device


def set_mdc_data_loader_epoch(data_loader: DataLoader[object], epoch: int) -> None:
    sampler = getattr(data_loader, "sampler", None)
    set_epoch = getattr(sampler, "set_epoch", None)
    if callable(set_epoch):
        set_epoch(int(epoch))

    dataset = getattr(data_loader, "dataset", None)
    dataset_set_epoch = getattr(dataset, "set_epoch", None)
    if callable(dataset_set_epoch):
        dataset_set_epoch(int(epoch))


def unwrap_mdc_training_model(model: torch.nn.Module) -> torch.nn.Module:
    resolved_model = model
    while True:
        next_model = None
        wrapped_module = getattr(resolved_model, "module", None)
        if isinstance(wrapped_module, torch.nn.Module):
            next_model = wrapped_module
        else:
            compiled_module = getattr(resolved_model, "_orig_mod", None)
            if isinstance(compiled_module, torch.nn.Module):
                next_model = compiled_module

        if next_model is None or next_model is resolved_model:
            return resolved_model
        resolved_model = next_model


def _default_process_group_backend(device: torch.device) -> str:
    if platform.system() == "Windows":
        os.environ.setdefault("USE_LIBUV", "0")
        return "gloo"
    if device.type != "cuda":
        return "gloo"
    return "nccl"


def _normalize_multi_gpu_mode(multi_gpu: MDCMultiGPUMode | str) -> MDCMultiGPUMode:
    resolved_mode = str(multi_gpu).strip().lower()
    aliases = {
        "off": "none",
        "dp": "data_parallel",
    }
    resolved_mode = aliases.get(resolved_mode, resolved_mode)
    if resolved_mode not in {"auto", "none", "data_parallel", "ddp"}:
        raise ValueError("multi_gpu must be one of: 'auto', 'none', 'data_parallel', 'ddp'.")
    return resolved_mode  # type: ignore[return-value]


def _resolve_data_parallel_device_ids(
    device: torch.device,
    *,
    data_parallel_device_ids: Sequence[int] | None = None,
) -> tuple[int, ...]:
    if data_parallel_device_ids is not None:
        resolved_device_ids = tuple(int(device_id) for device_id in data_parallel_device_ids)
    else:
        primary_device_id = int(device.index if device.index is not None else 0)
        visible_device_count = torch.cuda.device_count()
        resolved_device_ids = (primary_device_id,) + tuple(
            device_id
            for device_id in range(visible_device_count)
            if device_id != primary_device_id
        )

    if not resolved_device_ids:
        raise ValueError("data_parallel_device_ids must not be empty.")
    return resolved_device_ids
