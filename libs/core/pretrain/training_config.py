from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml

from libs.data.config import DataConfig, MinioConfig

from .training import create_muon_optimizers

DEFAULT_PROTEIN_TRAIN_CONFIG_PATH = Path("config/train.yaml")
LEGACY_PROTEIN_TRAIN_CONFIG_PATH = Path("train.yaml")


def load_protein_training_config(
    project_root: Path | str,
    config_path: Path | str | None = None,
) -> dict[str, Any]:
    resolved_project_root = Path(project_root).resolve()
    resolved_config_path = _resolve_config_path(resolved_project_root, config_path)
    config_mapping = _load_yaml_mapping(resolved_config_path)

    train_text_path = _as_project_path(
        _nested_get(config_mapping, "paths", "train_text_path")
        or Path("data/compiled/refseq_bacteria_protein/train.txt"),
        project_root=resolved_project_root,
    )
    tokenizer_map_path = _as_project_path(
        _nested_get(config_mapping, "paths", "tokenizer_map_path")
        or train_text_path.with_name("tokenizer_map.json"),
        project_root=resolved_project_root,
    )
    checkpoint_dir = _as_project_path(
        _nested_get(config_mapping, "paths", "checkpoint_dir")
        or Path("data/checkpoints/protein_from_scratch"),
        project_root=resolved_project_root,
    )
    train_part_cache_dir = _as_project_path(
        _nested_get(config_mapping, "paths", "train_part_cache_dir")
        or Path("data/cache/protein_train_parts"),
        project_root=resolved_project_root,
    )

    pin_memory = _resolve_auto_bool(
        _nested_get(config_mapping, "data", "pin_memory"),
        default=torch.cuda.is_available(),
    )
    fused_adamw = _resolve_auto_bool(
        _nested_get(config_mapping, "optimizer", "fused"),
        default=True,
    )

    resume_state_path = _as_project_path(
        _nested_get(config_mapping, "paths", "resume_state_path")
        or checkpoint_dir / "resume_state.json",
        project_root=resolved_project_root,
    )
    metrics_history_path = _as_project_path(
        _nested_get(config_mapping, "paths", "metrics_history_path")
        or checkpoint_dir / "metrics_history.jsonl",
        project_root=resolved_project_root,
    )
    training_config_snapshot_path = _as_project_path(
        _nested_get(config_mapping, "paths", "training_config_snapshot_path")
        or checkpoint_dir / "training_config.snapshot.json",
        project_root=resolved_project_root,
    )

    mode_name = str(_nested_get(config_mapping, "mode", "name") or "train_from_scratch")
    if mode_name not in {"train_from_scratch", "resume", "auto"}:
        raise ValueError("mode.name must be one of: train_from_scratch, resume, auto")
    resume_if_available = _bool_value(
        _nested_get(config_mapping, "mode", "resume_if_available"),
        True,
    )

    resolved_config: dict[str, Any] = {
        "config_path": resolved_config_path,
        "mode": {
            "name": mode_name,
            "resume_if_available": resume_if_available,
        },
        "paths": {
            "train_text_path": train_text_path,
            "tokenizer_map_path": tokenizer_map_path,
            "checkpoint_dir": checkpoint_dir,
            "train_part_cache_dir": train_part_cache_dir,
            "resume_state_path": resume_state_path,
            "metrics_history_path": metrics_history_path,
            "training_config_snapshot_path": training_config_snapshot_path,
        },
        "data": {
            "train_part_glob": str(_nested_get(config_mapping, "data", "train_part_glob") or "train_part_*.txt"),
            "prefer_local_train_parts": _bool_value(
                _nested_get(config_mapping, "data", "prefer_local_train_parts"),
                True,
            ),
            "stream_local_train_parts": _bool_value(
                _nested_get(config_mapping, "data", "stream_local_train_parts"),
                True,
            ),
            "keep_downloaded_train_parts": _bool_value(
                _nested_get(config_mapping, "data", "keep_downloaded_train_parts"),
                False,
            ),
            "cleanup_completed_parts": _bool_value(
                _nested_get(config_mapping, "data", "cleanup_completed_parts"),
                False,
            ),
            "validate_cached_parts": _bool_value(
                _nested_get(config_mapping, "data", "validate_cached_parts"),
                True,
            ),
            "train_ratio": float(_nested_get(config_mapping, "data", "train_ratio") or 0.9),
            "split_seed": int(_nested_get(config_mapping, "data", "split_seed") or 42),
            "batch_size": int(_nested_get(config_mapping, "data", "batch_size") or 2),
            "num_workers": int(_nested_get(config_mapping, "data", "num_workers") or 0),
            "pin_memory": pin_memory,
            "shuffle_parts": _bool_value(
                _nested_get(config_mapping, "data", "shuffle_parts"),
                False,
            ),
            "shuffle_examples": _bool_value(
                _nested_get(config_mapping, "data", "shuffle_examples"),
                True,
            ),
            "shuffle_buffer_size": int(_nested_get(config_mapping, "data", "shuffle_buffer_size") or 8192),
        },
        "model": {
            "progen_model_size": str(_nested_get(config_mapping, "model", "progen_model_size") or "0.8B"),
            "progen_config_overrides": dict(
                _mapping_value(
                    _nested_get(config_mapping, "model", "progen_config_overrides"),
                    default={
                        "emb_dim": 256,
                        "n_heads": 4,
                        "n_layers": 4,
                        "hidden_dim": 1024,
                        "head_dim": 64,
                        "n_kv_groups": 2,
                        "linear_key_head_dim": 64,
                        "linear_value_head_dim": 64,
                        "linear_num_key_heads": 4,
                        "linear_num_value_heads": 4,
                    },
                )
            ),
            "context_length": int(_nested_get(config_mapping, "model", "context_length") or 512),
            "stride": _optional_int(_nested_get(config_mapping, "model", "stride"), default=256),
            "tokenizer_vocab_size": int(_nested_get(config_mapping, "model", "tokenizer_vocab_size") or 512),
            "rebuild_tokenizer": _bool_value(
                _nested_get(config_mapping, "model", "rebuild_tokenizer"),
                False,
            ),
        },
        "training": {
            "num_epochs": int(_nested_get(config_mapping, "training", "num_epochs") or 1),
            "max_steps": _optional_int(_nested_get(config_mapping, "training", "max_steps")),
            "gradient_accumulation_steps": int(
                _nested_get(config_mapping, "training", "gradient_accumulation_steps") or 1
            ),
            "save_every_steps": _optional_int(
                _nested_get(config_mapping, "training", "save_every_steps"),
                default=100,
            ),
            "grad_clip_norm": _optional_float(
                _nested_get(config_mapping, "training", "grad_clip_norm"),
                default=1.0,
            ),
            "eval_freq": int(_nested_get(config_mapping, "training", "eval_freq") or 50),
            "eval_batches": int(_nested_get(config_mapping, "training", "eval_batches") or 10),
            "save_last": _bool_value(
                _nested_get(config_mapping, "training", "save_last"),
                False,
            ),
            "save_best": _bool_value(
                _nested_get(config_mapping, "training", "save_best"),
                True,
            ),
            "save_final": _bool_value(
                _nested_get(config_mapping, "training", "save_final"),
                True,
            ),
        },
        "optimizer": _resolve_optimizer_config(config_mapping, fused_adamw),
        "runtime": {
            "device": _normalize_device(_nested_get(config_mapping, "runtime", "device") or "auto"),
            "multi_gpu_mode": str(_nested_get(config_mapping, "runtime", "multi_gpu_mode") or "auto"),
            "ddp_find_unused_parameters": _bool_value(
                _nested_get(config_mapping, "runtime", "ddp_find_unused_parameters"),
                False,
            ),
            "data_parallel_device_ids": _int_sequence_or_none(
                _nested_get(config_mapping, "runtime", "data_parallel_device_ids")
            ),
            "mixed_precision": _resolve_mixed_precision(
                _nested_get(config_mapping, "runtime", "mixed_precision")
            ),
            "preflight_vram_check": _bool_value(
                _nested_get(config_mapping, "runtime", "preflight_vram_check"),
                False,
            ),
            "target_vram_gb": _optional_float(
                _nested_get(config_mapping, "runtime", "target_vram_gb"),
                default=14.0,
            ),
            "use_fp32_fallback_linear_attention": _bool_value(
                _nested_get(config_mapping, "runtime", "use_fp32_fallback_linear_attention"),
                True,
            ),
        },
        "resume": {
            "checkpoint_path": _as_project_path(
                _nested_get(config_mapping, "resume", "checkpoint_path")
                or checkpoint_dir / "checkpoint_best.pt",
                project_root=resolved_project_root,
            ),
            "output_checkpoint_path": _as_project_path(
                _nested_get(config_mapping, "resume", "output_checkpoint_path")
                or checkpoint_dir / "checkpoint_best.pt",
                project_root=resolved_project_root,
            ),
            "best_checkpoint_path": _as_project_path(
                _nested_get(config_mapping, "resume", "best_checkpoint_path")
                or checkpoint_dir / "checkpoint_best.pt",
                project_root=resolved_project_root,
            ),
            "final_checkpoint_path": _as_project_path(
                _nested_get(config_mapping, "resume", "final_checkpoint_path")
                or checkpoint_dir / "checkpoint_final.pt",
                project_root=resolved_project_root,
            ),
            "restore_optimizer_state": _bool_value(
                _nested_get(config_mapping, "resume", "restore_optimizer_state"),
                True,
            ),
            "override_optimizer_hyperparameters": _bool_value(
                _nested_get(config_mapping, "resume", "override_optimizer_hyperparameters"),
                True,
            ),
            "resume_state_path": _as_project_path(
                _nested_get(config_mapping, "resume", "resume_state_path")
                or resume_state_path,
                project_root=resolved_project_root,
            ),
        },
        "minio": {
            "train_parts_prefix_uri": str(_nested_get(config_mapping, "minio", "train_parts_prefix_uri") or ""),
            "train_part_uris": _string_sequence(_nested_get(config_mapping, "minio", "train_part_uris")),
            "manifest_uri": _optional_string(_nested_get(config_mapping, "minio", "manifest_uri")),
            "endpoint_url": _optional_string(_nested_get(config_mapping, "minio", "endpoint_url")),
            "access_key": _optional_string(_nested_get(config_mapping, "minio", "access_key")),
            "secret_key": _optional_string(_nested_get(config_mapping, "minio", "secret_key")),
            "bucket_name": _optional_string(_nested_get(config_mapping, "minio", "bucket_name")),
            "region_name": _optional_string(_nested_get(config_mapping, "minio", "region_name")),
            "secure": _optional_bool(_nested_get(config_mapping, "minio", "secure")),
        },
    }

    _validate_config(resolved_config)
    return resolved_config


def build_protein_training_data_config(
    project_root: Path | str,
    training_config: Mapping[str, Any],
) -> DataConfig | None:
    minio_config = _mapping_value(training_config.get("minio"), default={})
    explicit_overrides = {
        key: value
        for key, value in minio_config.items()
        if key not in {"train_parts_prefix_uri", "train_part_uris"} and value is not None
    }
    if not explicit_overrides:
        return None

    resolved_project_root = Path(project_root).resolve()
    base_config = DataConfig.load(
        resolved_project_root / "config.yaml",
        resolved_project_root / ".env",
    )
    base_minio = base_config.minio
    return DataConfig(
        storage_mode=base_config.storage_mode,
        data_root=base_config.data_root,
        default_batch_size=base_config.default_batch_size,
        minio=MinioConfig(
            endpoint_url=str(explicit_overrides.get("endpoint_url", base_minio.endpoint_url)),
            access_key=str(explicit_overrides.get("access_key", base_minio.access_key)),
            secret_key=str(explicit_overrides.get("secret_key", base_minio.secret_key)),
            bucket_name=str(explicit_overrides.get("bucket_name", base_minio.bucket_name)),
            region_name=_optional_string(explicit_overrides.get("region_name")) or base_minio.region_name,
            secure=base_minio.secure if explicit_overrides.get("secure") is None else bool(explicit_overrides["secure"]),
            root_prefix=base_minio.root_prefix,
        ),
    )


def create_protein_training_optimizer(
    model: torch.nn.Module,
    optimizer_config: Mapping[str, Any],
    *,
    device: torch.device | str,
) -> torch.optim.Optimizer | list[torch.optim.Optimizer]:
    optimizer_type = _normalize_optimizer_type(optimizer_config.get("type") or "adamw")
    learning_rate = float(optimizer_config.get("learning_rate") or 3e-4)
    weight_decay = float(optimizer_config.get("weight_decay") or 0.1)

    if optimizer_type == "muon":
        return create_muon_optimizers(
            model,
            adamw_learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

    resolved_device = torch.device(device)
    optimizer_kwargs: dict[str, Any] = {}
    if _should_use_fused_adamw(optimizer_config.get("fused"), resolved_device):
        optimizer_kwargs["fused"] = True
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        **optimizer_kwargs,
    )


def apply_protein_training_optimizer_settings(
    optimizer: torch.optim.Optimizer | Sequence[torch.optim.Optimizer],
    optimizer_config: Mapping[str, Any],
) -> None:
    adamw_learning_rate = float(optimizer_config.get("learning_rate") or 3e-4)
    weight_decay = float(optimizer_config.get("weight_decay") or 0.1)

    for opt in _optimizer_sequence(optimizer):
        resolved_learning_rate = adamw_learning_rate
        for group in opt.param_groups:
            group["lr"] = resolved_learning_rate
            if "weight_decay" in group:
                group["weight_decay"] = weight_decay


def describe_protein_training_optimizers(
    optimizer: torch.optim.Optimizer | Sequence[torch.optim.Optimizer],
) -> list[str]:
    return [type(opt).__name__ for opt in _optimizer_sequence(optimizer)]


def _resolve_config_path(project_root: Path, config_path: Path | str | None) -> Path:
    if config_path is None:
        preferred_path = project_root / DEFAULT_PROTEIN_TRAIN_CONFIG_PATH
        if preferred_path.exists():
            return preferred_path
        legacy_path = project_root / LEGACY_PROTEIN_TRAIN_CONFIG_PATH
        if legacy_path.exists():
            return legacy_path
        return preferred_path

    resolved_path = Path(config_path)
    if resolved_path.is_absolute():
        return resolved_path
    project_path = (project_root / resolved_path).resolve()
    if project_path.exists():
        return project_path
    config_path_candidate = (project_root / "config" / resolved_path).resolve()
    if config_path_candidate.exists():
        return config_path_candidate
    return project_path


def _load_yaml_mapping(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Missing protein training config: {config_path}")

    raw_text = config_path.read_text(encoding="utf-8")
    if not raw_text.strip():
        return {}

    loaded = yaml.safe_load(raw_text)
    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise ValueError("training config YAML must contain a top-level mapping")
    return dict(loaded)


def _nested_get(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _mapping_value(value: Any, *, default: Mapping[str, Any]) -> Mapping[str, Any]:
    if value is None:
        return default
    if not isinstance(value, Mapping):
        raise ValueError("Expected a mapping value in training config YAML")
    return value


def _as_project_path(value: Any, *, project_root: Path) -> Path:
    resolved_path = Path(str(value))
    if resolved_path.is_absolute():
        return resolved_path
    return (project_root / resolved_path).resolve()


def _bool_value(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return _bool_value(value, False)


def _optional_int(value: Any, *, default: int | None = None) -> int | None:
    if value in {None, ""}:
        return default
    return int(value)


def _optional_float(value: Any, *, default: float | None = None) -> float | None:
    if value in {None, ""}:
        return default
    return float(value)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _resolve_optimizer_config(config_mapping: Mapping[str, Any], fused_adamw: bool) -> dict[str, Any]:
    optimizer_type = _normalize_optimizer_type(_nested_get(config_mapping, "optimizer", "type") or "adamw")
    optimizer_config: dict[str, Any] = {
        "type": optimizer_type,
        "learning_rate": float(_nested_get(config_mapping, "optimizer", "learning_rate") or 3e-4),
        "weight_decay": float(_nested_get(config_mapping, "optimizer", "weight_decay") or 0.1),
        "fused": fused_adamw,
    }
    return optimizer_config


def _resolve_auto_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str) and value.strip().lower() == "auto":
        return default
    return _bool_value(value, default)


def _normalize_device(value: Any) -> str:
    resolved = str(value).strip().lower()
    if resolved not in {"auto", "cpu", "cuda"}:
        raise ValueError("runtime.device must be one of: auto, cpu, cuda")
    return resolved


def _resolve_mixed_precision(value: Any) -> str:
    if value is None:
        return "auto"
    if isinstance(value, bool):
        return "no" if not value else "auto"
    resolved = str(value).strip().lower()
    if resolved not in {"auto", "no", "fp16", "bf16"}:
        raise ValueError("runtime.mixed_precision must be one of: auto, no, fp16, bf16")
    return resolved


def _normalize_optimizer_type(value: Any) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {"adamw", "muon"}:
        raise ValueError("optimizer.type must be either 'adamw' or 'muon'")
    return normalized


def _string_sequence(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) and not value:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence):
        raise ValueError("Expected a sequence of strings in training config YAML")
    return tuple(str(item) for item in value)


def _int_sequence_or_none(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, str) and not value:
        return None
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError("runtime.data_parallel_device_ids must be a list of integers or null")
    return tuple(int(item) for item in value)


def _optimizer_sequence(
    optimizer: torch.optim.Optimizer | Sequence[torch.optim.Optimizer],
) -> list[torch.optim.Optimizer]:
    if isinstance(optimizer, torch.optim.Optimizer):
        return [optimizer]
    return list(optimizer)


def _should_use_fused_adamw(fused_value: Any, device: torch.device) -> bool:
    if device.type != "cuda":
        return False

    resolved_fused = _resolve_auto_bool(fused_value, default=True)
    return bool(resolved_fused) and "fused" in inspect.signature(torch.optim.AdamW).parameters


def _validate_config(training_config: Mapping[str, Any]) -> None:
    data_config = _mapping_value(training_config.get("data"), default={})
    model_config = _mapping_value(training_config.get("model"), default={})
    optimizer_config = _mapping_value(training_config.get("optimizer"), default={})
    training_settings = _mapping_value(training_config.get("training"), default={})
    runtime_config = _mapping_value(training_config.get("runtime"), default={})
    resume_config = _mapping_value(training_config.get("resume"), default={})

    train_ratio = float(data_config.get("train_ratio", 0.9))
    if not 0 < train_ratio < 1:
        raise ValueError("data.train_ratio must be between 0 and 1")
    if int(data_config.get("batch_size", 1)) <= 0:
        raise ValueError("data.batch_size must be greater than 0")
    if int(data_config.get("num_workers", 0)) < 0:
        raise ValueError("data.num_workers must be greater than or equal to 0")

    if int(model_config.get("context_length", 0)) <= 0:
        raise ValueError("model.context_length must be greater than 0")
    stride = model_config.get("stride")
    if stride is not None and int(stride) <= 0:
        raise ValueError("model.stride must be greater than 0 when provided")
    if int(model_config.get("tokenizer_vocab_size", 0)) <= 0:
        raise ValueError("model.tokenizer_vocab_size must be greater than 0")

    if int(training_settings.get("num_epochs", 0)) <= 0:
        raise ValueError("training.num_epochs must be greater than 0")
    if int(training_settings.get("eval_batches", 0)) <= 0:
        raise ValueError("training.eval_batches must be greater than 0")
    if _bool_value(training_settings.get("save_last"), False):
        raise ValueError("training.save_last must be false; use checkpoint_best.pt as the model artifact")
    if not _bool_value(training_settings.get("save_best"), True):
        raise ValueError("training.save_best must be true so checkpoint_best.pt is available")

    if float(optimizer_config.get("learning_rate", 0.0)) <= 0:
        raise ValueError("optimizer.learning_rate must be greater than 0")
    if float(optimizer_config.get("weight_decay", 0.0)) < 0:
        raise ValueError("optimizer.weight_decay must be greater than or equal to 0")

    if runtime_config.get("multi_gpu_mode") not in {"auto", "none", "data_parallel", "ddp"}:
        raise ValueError("runtime.multi_gpu_mode must be one of: auto, none, data_parallel, ddp")

    checkpoint_path = Path(resume_config.get("checkpoint_path"))
    if checkpoint_path.name == "":
        raise ValueError("resume.checkpoint_path must point to a checkpoint file")
    if checkpoint_path.name != "checkpoint_best.pt":
        raise ValueError("resume.checkpoint_path must point to checkpoint_best.pt")
