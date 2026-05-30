from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import re
import shutil
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib import error as urllib_error
from urllib.parse import urlparse
from urllib import request as urllib_request

import boto3
import botocore.auth as botocore_auth
from botocore.config import Config

from libs.data.config import DATA_CONFIG, DataConfig


DEFAULT_TEXT_PART_SUFFIXES = (".txt", ".jsonl")
DEFAULT_EXCLUDED_PART_NAMES = (
    "tokenizer_map.json",
    "manifest.json",
    "summary.json",
    "history.json",
)


@dataclass(slots=True, frozen=True)
class S3TextPart:
    uri: str
    bucket: str
    key: str
    size: int | None = None
    etag: str | None = None


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected an s3://bucket/key URI, got: {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def build_minio_s3_client(config: DataConfig | None = None):
    resolved_config = config or DATA_CONFIG
    _apply_s3_endpoint_clock_offset(
        resolved_config.minio.normalized_endpoint_url,
        botocore_auth=botocore_auth,
    )

    return boto3.client(
        "s3",
        endpoint_url=resolved_config.minio.normalized_endpoint_url,
        aws_access_key_id=resolved_config.minio.access_key,
        aws_secret_access_key=resolved_config.minio.secret_key,
        region_name=resolved_config.minio.region_name,
        use_ssl=resolved_config.minio.secure,
        config=Config(
            s3={"addressing_style": "path"},
            retries={"max_attempts": 10, "mode": "standard"},
        ),
    )


def _apply_s3_endpoint_clock_offset(endpoint_url: str, *, botocore_auth) -> None:
    try:
        request = urllib_request.Request(endpoint_url, method="GET")
        try:
            response = urllib_request.urlopen(request, timeout=20)
            date_header = response.headers.get("Date")
        except urllib_error.HTTPError as exc:
            date_header = exc.headers.get("Date")
        if not date_header:
            return

        server_time = parsedate_to_datetime(date_header)
        if server_time.tzinfo is None:
            server_time = server_time.replace(tzinfo=timezone.utc)
        server_time = server_time.astimezone(timezone.utc)
        offset = server_time - datetime.now(timezone.utc)
    except Exception:
        return

    def adjusted_datetime(remove_tzinfo: bool = True):
        adjusted = datetime.now(timezone.utc) + offset
        if remove_tzinfo:
            return adjusted.replace(tzinfo=None)
        return adjusted

    botocore_auth.get_current_datetime = adjusted_datetime


def list_minio_text_parts(
    *,
    prefix_uri: str | None = None,
    part_uris: Sequence[str] | None = None,
    s3_client=None,
    config: DataConfig | None = None,
    suffixes: Sequence[str] | None = DEFAULT_TEXT_PART_SUFFIXES,
    excluded_file_names: Sequence[str] = DEFAULT_EXCLUDED_PART_NAMES,
) -> tuple[S3TextPart, ...]:
    if bool(prefix_uri) == bool(part_uris):
        raise ValueError("Provide exactly one of prefix_uri or part_uris.")

    if part_uris is not None:
        parts: list[S3TextPart] = []
        for uri in part_uris:
            bucket, key = parse_s3_uri(uri)
            if not key:
                raise ValueError(f"Part URI must include an object key: {uri!r}")
            parts.append(S3TextPart(uri=uri, bucket=bucket, key=key))
        return tuple(parts)

    assert prefix_uri is not None
    bucket, prefix_key = parse_s3_uri(prefix_uri)
    client = s3_client or build_minio_s3_client(config)
    normalized_suffixes = tuple(suffixes or ())
    excluded_names = set(excluded_file_names)

    paginator = client.get_paginator("list_objects_v2")
    parts = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix_key):
        for item in page.get("Contents", []):
            key = str(item["Key"])
            if not _is_text_part_key(
                key,
                suffixes=normalized_suffixes,
                excluded_file_names=excluded_names,
            ):
                continue
            parts.append(
                S3TextPart(
                    uri=f"s3://{bucket}/{key}",
                    bucket=bucket,
                    key=key,
                    size=_optional_int(item.get("Size")),
                    etag=str(item["ETag"]) if item.get("ETag") is not None else None,
                )
            )

    return tuple(sorted(parts, key=lambda part: _natural_key(part.key)))


@contextmanager
def downloaded_minio_text_part(
    part: S3TextPart,
    *,
    s3_client=None,
    config: DataConfig | None = None,
    cache_dir: Path | str | None = None,
    keep_downloaded_parts: bool = False,
    validate_cached: bool = True,
) -> Iterator[Path]:
    if keep_downloaded_parts and cache_dir is None:
        raise ValueError("cache_dir is required when keep_downloaded_parts=True.")

    client = s3_client or build_minio_s3_client(config)
    temp_dir: Path | None = None
    if cache_dir is None:
        temp_dir = Path(tempfile.mkdtemp(prefix="mdc-s3-part-"))
        target_dir = temp_dir
    else:
        target_dir = Path(cache_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

    target_path = _part_cache_path(part, target_dir)
    if target_path.exists():
        if validate_cached and not _validate_cached_part(part, target_path):
            target_path.unlink(missing_ok=True)
            _download_s3_object(client, part, target_path)
    else:
        _download_s3_object(client, part, target_path)

    try:
        yield target_path
    finally:
        if not keep_downloaded_parts:
            target_path.unlink(missing_ok=True)
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)


def _validate_cached_part(part: S3TextPart, local_path: Path) -> bool:
    if part.size is not None:
        local_size = local_path.stat().st_size
        if local_size != part.size:
            return False
    return True


def _is_text_part_key(
    key: str,
    *,
    suffixes: Sequence[str],
    excluded_file_names: set[str],
) -> bool:
    if not key or key.endswith("/"):
        return False
    file_name = key.rsplit("/", 1)[-1]
    if file_name in excluded_file_names:
        return False
    if suffixes and not key.endswith(tuple(suffixes)):
        return False
    return True


def _download_s3_object(client, part: S3TextPart, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = target_path.with_name(f"{target_path.name}.downloading")
    temporary_path.unlink(missing_ok=True)
    try:
        download_file = getattr(client, "download_file", None)
        if callable(download_file):
            download_file(part.bucket, part.key, str(temporary_path))
        else:
            response = client.get_object(Bucket=part.bucket, Key=part.key)
            with temporary_path.open("wb") as handle:
                shutil.copyfileobj(response["Body"], handle, length=16 * 1024 * 1024)
        temporary_path.replace(target_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _part_cache_path(part: S3TextPart, cache_dir: Path) -> Path:
    digest = hashlib.sha256(part.uri.encode("utf-8")).hexdigest()[:16]
    file_name = _safe_file_name(Path(part.key).name or "part.txt")
    return cache_dir / f"{digest}-{file_name}"


def _safe_file_name(value: str) -> str:
    cleaned = []
    for character in value:
        if character.isalnum() or character in {".", "-", "_"}:
            cleaned.append(character)
        else:
            cleaned.append("_")
    return "".join(cleaned) or "part.txt"


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _natural_key(value: str) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", value)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)
