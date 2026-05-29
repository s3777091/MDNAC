"""Split instruction.slim.jsonl into N parts and upload to S3.

This temp script:
  1. Splits the slimmed instruction.jsonl into N equal-ish parts
  2. Uploads each part to s3://microbial-dna-compiler/data/instruction/parts/
  3. Creates and uploads a manifest.json

Usage:
    python temp/split_and_upload_instruction.py temp/instruction.slim.jsonl \
        --parts 5 \
        --s3-prefix data/instruction/parts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path so we can import libs
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.data.config import DataConfig
from libs.data.training.streaming import build_minio_s3_client


def count_lines(path: Path) -> int:
    """Count non-empty lines in a file."""
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def split_jsonl(
    input_path: Path,
    output_dir: Path,
    *,
    num_parts: int = 5,
    part_prefix: str = "instruction_part",
) -> list[Path]:
    """Split a JSONL file into N roughly equal parts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # First pass: count lines
    print(f"[split] Counting lines...", flush=True)
    total_lines = count_lines(input_path)
    lines_per_part = total_lines // num_parts
    remainder = total_lines % num_parts
    print(f"[split] Total lines: {total_lines:,}")
    print(f"[split] Lines per part: ~{lines_per_part:,} (+{remainder} remainder)")

    # Second pass: write parts
    part_paths: list[Path] = []
    current_part = 0
    current_lines_written = 0
    current_part_limit = lines_per_part + (1 if current_part < remainder else 0)
    current_part_path = output_dir / f"{part_prefix}_{current_part + 1}.jsonl"
    part_paths.append(current_part_path)
    current_handle = current_part_path.open("w", encoding="utf-8")

    lines_processed = 0
    start_time = time.time()
    last_report = start_time

    try:
        with input_path.open("r", encoding="utf-8") as source:
            for raw_line in source:
                if not raw_line.strip():
                    continue

                current_handle.write(raw_line if raw_line.endswith("\n") else f"{raw_line}\n")
                current_lines_written += 1
                lines_processed += 1

                if current_lines_written >= current_part_limit:
                    current_handle.close()
                    size_mb = current_part_path.stat().st_size / (1024**2)
                    print(
                        f"[split] Part {current_part + 1}/{num_parts}: "
                        f"{current_lines_written:,} lines, {size_mb:.0f} MB",
                        flush=True,
                    )

                    current_part += 1
                    if current_part < num_parts:
                        current_lines_written = 0
                        current_part_limit = lines_per_part + (1 if current_part < remainder else 0)
                        current_part_path = output_dir / f"{part_prefix}_{current_part + 1}.jsonl"
                        part_paths.append(current_part_path)
                        current_handle = current_part_path.open("w", encoding="utf-8")

                # Progress
                now = time.time()
                if now - last_report >= 30.0:
                    pct = lines_processed / total_lines * 100
                    print(f"  [{now - start_time:.0f}s] {lines_processed:,}/{total_lines:,} ({pct:.1f}%)", flush=True)
                    last_report = now
    finally:
        if not current_handle.closed:
            current_handle.close()
            if current_lines_written > 0:
                size_mb = current_part_path.stat().st_size / (1024**2)
                print(
                    f"[split] Part {current_part + 1}/{num_parts}: "
                    f"{current_lines_written:,} lines, {size_mb:.0f} MB",
                    flush=True,
                )

    elapsed = time.time() - start_time
    print(f"[split] Done splitting in {elapsed:.1f}s")
    return part_paths


def compute_md5(path: Path) -> str:
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def upload_parts_to_s3(
    part_paths: list[Path],
    *,
    s3_prefix: str,
    bucket: str,
    s3_client,
) -> list[dict]:
    """Upload all parts to S3 and return metadata for manifest."""
    parts_metadata = []

    for i, part_path in enumerate(part_paths):
        s3_key = f"{s3_prefix}/{part_path.name}"
        file_size = part_path.stat().st_size
        size_gb = file_size / (1024**3)

        print(f"\n[upload] Uploading {part_path.name} ({size_gb:.2f} GB) -> s3://{bucket}/{s3_key}")
        start_time = time.time()

        # Use multipart upload for large files
        from boto3.s3.transfer import TransferConfig
        transfer_config = TransferConfig(
            multipart_threshold=64 * 1024 * 1024,   # 64MB
            multipart_chunksize=64 * 1024 * 1024,    # 64MB
            max_concurrency=4,
            use_threads=True,
        )

        # Upload with progress callback
        bytes_uploaded = [0]
        last_report_time = [time.time()]

        def progress_callback(bytes_transferred):
            bytes_uploaded[0] += bytes_transferred
            now = time.time()
            if now - last_report_time[0] >= 15.0:
                pct = bytes_uploaded[0] / file_size * 100
                speed_mbps = bytes_uploaded[0] / (now - start_time) / (1024**2)
                print(
                    f"  {bytes_uploaded[0] / (1024**3):.2f}/{size_gb:.2f} GB "
                    f"({pct:.1f}%, {speed_mbps:.1f} MB/s)",
                    flush=True,
                )
                last_report_time[0] = now

        s3_client.upload_file(
            str(part_path),
            bucket,
            s3_key,
            Config=transfer_config,
            Callback=progress_callback,
        )

        elapsed = time.time() - start_time
        speed_mbps = file_size / elapsed / (1024**2) if elapsed > 0 else 0
        print(f"  Done in {elapsed:.1f}s ({speed_mbps:.1f} MB/s)")

        # Count lines for manifest
        line_count = count_lines(part_path)

        md5 = compute_md5(part_path)

        parts_metadata.append({
            "filename": part_path.name,
            "s3_key": s3_key,
            "size_bytes": file_size,
            "line_count": line_count,
            "md5": md5,
        })

    return parts_metadata


def create_and_upload_manifest(
    parts_metadata: list[dict],
    *,
    s3_prefix: str,
    bucket: str,
    s3_client,
    input_path: str,
    local_manifest_path: Path | None = None,
) -> dict:
    """Create manifest.json and upload to S3."""
    total_lines = sum(p["line_count"] for p in parts_metadata)
    total_bytes = sum(p["size_bytes"] for p in parts_metadata)

    manifest = {
        "format": "instruction_jsonl_slim",
        "version": "2.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(input_path),
        "total_records": total_lines,
        "total_size_bytes": total_bytes,
        "num_parts": len(parts_metadata),
        "fields": [
            "instruction",
            "input",
            "output",
            "accession",
            "organism",
            "product",
            "taxonomy",
            "keywords",
            "derived_labels",
        ],
        "parts": [
            {
                "index": i + 1,
                "filename": p["filename"],
                "s3_key": p["s3_key"],
                "s3_uri": f"s3://{bucket}/{p['s3_key']}",
                "size_bytes": p["size_bytes"],
                "line_count": p["line_count"],
                "md5": p["md5"],
            }
            for i, p in enumerate(parts_metadata)
        ],
    }

    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"

    # Save locally
    if local_manifest_path is not None:
        local_manifest_path.write_text(manifest_text, encoding="utf-8")
        print(f"\n[manifest] Saved locally: {local_manifest_path}")

    # Upload to S3
    manifest_key = f"{s3_prefix}/manifest.json"
    s3_client.put_object(
        Bucket=bucket,
        Key=manifest_key,
        Body=manifest_text.encode("utf-8"),
        ContentType="application/json",
    )
    print(f"[manifest] Uploaded: s3://{bucket}/{manifest_key}")

    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split instruction.slim.jsonl into parts and upload to S3.",
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Source instruction.slim.jsonl file.",
    )
    parser.add_argument(
        "--parts",
        type=int,
        default=5,
        help="Number of parts to split into. Default: 5",
    )
    parser.add_argument(
        "--s3-prefix",
        type=str,
        default="data/instruction/parts",
        help="S3 key prefix for upload. Default: data/instruction/parts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Local directory for split parts. Default: temp/parts",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Only split, don't upload to S3.",
    )
    parser.add_argument(
        "--skip-split",
        action="store_true",
        help="Skip splitting, use existing parts in output-dir.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_path: Path = args.input_path
    output_dir: Path = args.output_dir or Path("temp/parts")
    num_parts: int = args.parts
    s3_prefix: str = args.s3_prefix.strip("/")

    if not input_path.is_file() and not args.skip_split:
        print(f"[error] Input file not found: {input_path}")
        return 1

    print(f"[config] Input:     {input_path}")
    print(f"[config] Parts:     {num_parts}")
    print(f"[config] Output:    {output_dir}")
    print(f"[config] S3 prefix: {s3_prefix}")
    print()

    # Step 1: Split
    if not args.skip_split:
        part_paths = split_jsonl(input_path, output_dir, num_parts=num_parts)
    else:
        part_paths = sorted(output_dir.glob("instruction_part_*.jsonl"))
        if not part_paths:
            print(f"[error] No parts found in {output_dir}")
            return 1
        print(f"[split] Skipped, using {len(part_paths)} existing parts")

    if args.skip_upload:
        print("\n[upload] Skipped (--skip-upload)")
        return 0

    # Step 2: Upload
    print(f"\n[upload] Connecting to S3...")
    config = DataConfig.load()
    s3_client = build_minio_s3_client(config)
    bucket = config.minio.bucket_name
    print(f"[upload] Bucket: {bucket}")
    print(f"[upload] Endpoint: {config.minio.normalized_endpoint_url}")

    parts_metadata = upload_parts_to_s3(
        part_paths,
        s3_prefix=s3_prefix,
        bucket=bucket,
        s3_client=s3_client,
    )

    # Step 3: Manifest
    manifest = create_and_upload_manifest(
        parts_metadata,
        s3_prefix=s3_prefix,
        bucket=bucket,
        s3_client=s3_client,
        input_path=str(input_path),
        local_manifest_path=output_dir / "manifest.json",
    )

    print(f"\n{'='*60}")
    print(f"[done] All {num_parts} parts uploaded successfully!")
    print(f"[done] Total records: {manifest['total_records']:,}")
    print(f"[done] Total size:    {manifest['total_size_bytes'] / (1024**3):.2f} GB")
    print(f"[done] Manifest:      s3://{bucket}/{s3_prefix}/manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
