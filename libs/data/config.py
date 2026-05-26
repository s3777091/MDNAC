from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_FILE = Path(__file__).resolve().parents[2] / "config.yaml"
DEFAULT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
MODELS_DIR_NAME = "models"
CATALOG_DIR_NAME = "catalog"
DATASETS_DIR_NAME = "datasets"
TRASH_DIR_NAME = "trash"
SESSIONS_DIR_NAME = "sessions"
MINIO_ROOT_PREFIX = "libs/data/models"


def _env_flag(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _bool_value(raw_value: object, default: bool) -> bool:
    if raw_value is None:
        return default
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        return raw_value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(raw_value)


def _nested_get(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _load_config_mapping(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}

    raw_text = config_path.read_text(encoding="utf-8")
    if not raw_text.strip():
        return {}

    loaded = yaml.safe_load(raw_text)
    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise ValueError("config.yaml must contain a top-level mapping")
    return dict(loaded)


def _load_dotenv_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue

        cleaned_value = value.strip()
        if len(cleaned_value) >= 2 and cleaned_value[0] == cleaned_value[-1] and cleaned_value[0] in {"'", '"'}:
            cleaned_value = cleaned_value[1:-1]
        os.environ[key] = cleaned_value


def _as_path(value: object, *, base_dir: Path | None = None) -> Path:
    path = Path(str(value))
    if path.is_absolute() or base_dir is None:
        return path
    return base_dir / path


def _config_value(
    config_mapping: Mapping[str, Any],
    env_name: str,
    key_path: tuple[str, ...],
    default: Any,
) -> Any:
    env_value = os.getenv(env_name)
    if env_value is not None:
        return env_value

    yaml_value = _nested_get(config_mapping, *key_path)
    if yaml_value is not None:
        return yaml_value

    return default


def _clean_minio_root_prefix(value: object) -> str:
    cleaned = str(value or MINIO_ROOT_PREFIX).strip().strip("/")
    return cleaned or MINIO_ROOT_PREFIX


@dataclass(slots=True, frozen=True)
class MinioConfig:
    endpoint_url: str = field(default_factory=lambda: os.getenv("MICROBIAL_DATA_MINIO_ENDPOINT", "http://localhost:9000"))
    access_key: str = field(default_factory=lambda: os.getenv("MICROBIAL_DATA_MINIO_ACCESS_KEY", "minioadmin"))
    secret_key: str = field(default_factory=lambda: os.getenv("MICROBIAL_DATA_MINIO_SECRET_KEY", "minioadmin"))
    bucket_name: str = field(default_factory=lambda: os.getenv("MICROBIAL_DATA_MINIO_BUCKET", "microbial-dna-compiler"))
    region_name: str | None = field(default_factory=lambda: os.getenv("MICROBIAL_DATA_MINIO_REGION") or None)
    secure: bool = field(default_factory=lambda: _env_flag("MICROBIAL_DATA_MINIO_SECURE", True))
    root_prefix: str = field(default_factory=lambda: _clean_minio_root_prefix(os.getenv("MICROBIAL_DATA_MINIO_PREFIX")))

    @property
    def normalized_endpoint_url(self) -> str:
        if self.endpoint_url.startswith("http://") or self.endpoint_url.startswith("https://"):
            return self.endpoint_url
        scheme = "https" if self.secure else "http"
        return f"{scheme}://{self.endpoint_url}"


@dataclass(slots=True, frozen=True)
class DataConfig:
    storage_mode: str = field(default_factory=lambda: os.getenv("MICROBIAL_DATA_STORAGE_MODE", "local").strip().lower())
    data_root: Path = field(default_factory=lambda: Path(os.getenv("MICROBIAL_DATA_ROOT", "libs/data")))
    default_batch_size: int = field(default_factory=lambda: int(os.getenv("MICROBIAL_DATA_DEFAULT_BATCH_SIZE", "25")))
    minio: MinioConfig = field(default_factory=MinioConfig)

    def __post_init__(self) -> None:
        normalized_storage_mode = self.storage_mode.strip().lower()
        if normalized_storage_mode == "global":
            normalized_storage_mode = "minio"

        object.__setattr__(self, "storage_mode", normalized_storage_mode)
        object.__setattr__(self, "data_root", Path(self.data_root))

        if self.storage_mode not in {"local", "minio"}:
            raise ValueError("storage_mode must be either 'local' or 'minio'")
        if self.default_batch_size <= 0:
            raise ValueError("default_batch_size must be greater than 0")

    @classmethod
    def load(
        cls,
        config_path: Path | str | None = None,
        env_path: Path | str | None = None,
    ) -> "DataConfig":
        resolved_config_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_FILE
        resolved_env_path = Path(env_path) if env_path is not None else DEFAULT_ENV_FILE
        _load_dotenv_file(resolved_env_path)
        config_mapping = _load_config_mapping(resolved_config_path)
        base_dir = resolved_config_path.parent

        minio = MinioConfig(
            endpoint_url=str(
                _config_value(
                    config_mapping,
                    "MICROBIAL_DATA_MINIO_ENDPOINT",
                    ("minio", "endpoint_url"),
                    os.getenv("MICROBIAL_DATA_MINIO_ENDPOINT", "http://localhost:9000"),
                )
            ),
            access_key=str(
                _config_value(
                    config_mapping,
                    "MICROBIAL_DATA_MINIO_ACCESS_KEY",
                    ("minio", "access_key"),
                    os.getenv("MICROBIAL_DATA_MINIO_ACCESS_KEY", "minioadmin"),
                )
            ),
            secret_key=str(
                _config_value(
                    config_mapping,
                    "MICROBIAL_DATA_MINIO_SECRET_KEY",
                    ("minio", "secret_key"),
                    os.getenv("MICROBIAL_DATA_MINIO_SECRET_KEY", "minioadmin"),
                )
            ),
            bucket_name=str(
                _config_value(
                    config_mapping,
                    "MICROBIAL_DATA_MINIO_BUCKET",
                    ("minio", "bucket_name"),
                    os.getenv("MICROBIAL_DATA_MINIO_BUCKET", "microbial-dna-compiler"),
                )
            ),
            region_name=(
                str(region_name)
                if (region_name := _config_value(
                    config_mapping,
                    "MICROBIAL_DATA_MINIO_REGION",
                    ("minio", "region_name"),
                    os.getenv("MICROBIAL_DATA_MINIO_REGION"),
                ))
                not in {None, ""}
                else None
            ),
            secure=_bool_value(
                _config_value(
                    config_mapping,
                    "MICROBIAL_DATA_MINIO_SECURE",
                    ("minio", "secure"),
                    _env_flag("MICROBIAL_DATA_MINIO_SECURE", True),
                ),
                True,
            ),
            root_prefix=_clean_minio_root_prefix(
                _config_value(
                    config_mapping,
                    "MICROBIAL_DATA_MINIO_PREFIX",
                    ("minio", "root_prefix"),
                    os.getenv("MICROBIAL_DATA_MINIO_PREFIX", MINIO_ROOT_PREFIX),
                )
            ),
        )

        return cls(
            storage_mode=str(
                _config_value(
                    config_mapping,
                    "MICROBIAL_DATA_STORAGE_MODE",
                    ("storage_mode",),
                    os.getenv("MICROBIAL_DATA_STORAGE_MODE", "local"),
                )
            ).strip().lower(),
            data_root=_as_path(
                _config_value(
                    config_mapping,
                    "MICROBIAL_DATA_ROOT",
                    ("data_root",),
                    os.getenv("MICROBIAL_DATA_ROOT", "libs/data"),
                ),
                base_dir=base_dir,
            ),
            default_batch_size=int(
                _config_value(
                    config_mapping,
                    "MICROBIAL_DATA_DEFAULT_BATCH_SIZE",
                    ("default_batch_size",),
                    os.getenv("MICROBIAL_DATA_DEFAULT_BATCH_SIZE", "25"),
                )
            ),
            minio=minio,
        )

    @property
    def models_root(self) -> Path:
        return self.data_root / MODELS_DIR_NAME

    @property
    def catalog_root(self) -> Path:
        return self.models_root / CATALOG_DIR_NAME

    @property
    def datasets_root(self) -> Path:
        return self.models_root / DATASETS_DIR_NAME

    @property
    def trash_root(self) -> Path:
        return self.models_root / TRASH_DIR_NAME

    @property
    def sessions_root(self) -> Path:
        return self.models_root / SESSIONS_DIR_NAME


DATA_CONFIG = DataConfig.load()
