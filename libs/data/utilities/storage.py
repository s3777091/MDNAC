from __future__ import annotations

import csv
import io
import json
from collections.abc import Sequence
from datetime import datetime, timezone

from libs.data.entities import FetchRequest, ManagedDataset, SequenceRecord
from libs.data.training.tokenizer import SequenceTokenizer
from libs.data.utilities.parsers import parse_csv_rows

DATASET_FILE_NAMES = {
    "txt": "train.txt",
    "tokenizer_map": "tokenizer_map.json",
}
DATASET_BUNDLE_FILENAMES = tuple(DATASET_FILE_NAMES.values())

CATALOG_FIELDS = (
    "source_name",
    "dataset_name",
    "storage_mode",
    "record_count",
    "updated_at_utc",
    "snapshot_id",
    "current_location",
)


def utc_snapshot_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def build_dataset_bundle(
    source_name: str,
    request: FetchRequest,
    records: Sequence[SequenceRecord],
    storage_mode: str,
    snapshot_id: str,
    merge_strategy: str,
) -> dict[str, str]:
    del source_name, request, storage_mode, snapshot_id, merge_strategy
    return {
        DATASET_FILE_NAMES["txt"]: render_train_txt(records),
        DATASET_FILE_NAMES["tokenizer_map"]: render_tokenizer_map(records),
    }


def build_prebuilt_dataset_bundle(train_text: str, tokenizer_map_text: str) -> dict[str, str]:
    return {
        DATASET_FILE_NAMES["txt"]: train_text,
        DATASET_FILE_NAMES["tokenizer_map"]: tokenizer_map_text,
    }


def render_train_txt(records: Sequence[SequenceRecord]) -> str:
    return "\n".join(record.to_training_line() for record in records) + "\n"


def render_tokenizer_map(records: Sequence[SequenceRecord]) -> str:
    tokenizer = SequenceTokenizer.from_records(records)
    return render_tokenizer_map_payload(
        source_name=records[0].source_name if records else "",
        record_count=len(records),
        tokenizer=tokenizer,
    )


def render_tokenizer_map_payload(
    source_name: str,
    record_count: int,
    tokenizer: SequenceTokenizer,
) -> str:
    payload = {
        "source_name": source_name,
        "record_count": record_count,
        "tokenizer": json.loads(tokenizer.to_json()),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def parse_catalog_csv(raw_text: str) -> list[dict[str, str]]:
    return parse_csv_rows(raw_text)


def render_catalog_csv(rows: Sequence[dict[str, str]]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(CATALOG_FIELDS))
    writer.writeheader()
    for row in rows:
        writer.writerow({field_name: row.get(field_name, "") for field_name in CATALOG_FIELDS})
    return output.getvalue()


def managed_dataset_from_row(row: dict[str, str]) -> ManagedDataset:
    return ManagedDataset(
        source_name=row.get("source_name", ""),
        dataset_name=row.get("dataset_name", ""),
        storage_mode=row.get("storage_mode", "local"),
        current_location=row.get("current_location", ""),
        record_count=int(row.get("record_count", "0") or 0),
        updated_at_utc=row.get("updated_at_utc", ""),
        snapshot_id=row.get("snapshot_id", ""),
    )
