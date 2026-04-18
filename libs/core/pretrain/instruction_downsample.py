from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping


DEFAULT_KEEP_RATIO = 0.5
DEFAULT_SUBLINEAR_ALPHA = 0.8
SYSTEMATIC_PHASE_DENOMINATOR = 1 << 20

StratumKey = tuple[str, str]


@dataclass(slots=True)
class InstructionJsonlDownsampleSummary:
    input_path: str
    output_path: str
    total_line_count: int
    target_line_count: int
    written_line_count: int
    keep_ratio: float
    alpha: float
    dataset_group_count: int
    unique_stratum_count: int


@dataclass(slots=True, frozen=True)
class _StratumPlan:
    total_count: int
    keep_count: int
    phase_numerator: int


def downsample_instruction_jsonl(
    input_path: Path | str,
    *,
    output_path: Path | str,
    keep_ratio: float = DEFAULT_KEEP_RATIO,
    alpha: float = DEFAULT_SUBLINEAR_ALPHA,
    overwrite: bool = False,
    dry_run: bool = False,
) -> InstructionJsonlDownsampleSummary:
    resolved_input_path = Path(input_path)
    resolved_output_path = Path(output_path)
    if not resolved_input_path.is_file():
        raise FileNotFoundError(f"instruction.jsonl was not found: {resolved_input_path}")
    if resolved_input_path.resolve() == resolved_output_path.resolve():
        raise ValueError("output_path must be different from input_path.")
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("keep_ratio must be greater than 0 and less than or equal to 1.")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be greater than 0 and less than or equal to 1.")
    if resolved_output_path.exists() and not overwrite and not dry_run:
        raise FileExistsError(f"Refusing to overwrite existing output file: {resolved_output_path}")

    total_line_count, counts_by_group = _collect_stratum_counts(resolved_input_path)
    if total_line_count == 0:
        raise ValueError("instruction.jsonl does not contain any non-empty JSONL records.")

    target_line_count = int(round(total_line_count * keep_ratio))
    group_targets = _allocate_group_targets(counts_by_group, target_line_count)
    plan_by_stratum = _build_stratum_plan(
        counts_by_group,
        group_targets=group_targets,
        alpha=alpha,
    )
    written_line_count = sum(plan.keep_count for plan in plan_by_stratum.values())

    if not dry_run:
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_downsampled_jsonl(
            resolved_input_path,
            resolved_output_path,
            plan_by_stratum=plan_by_stratum,
        )

    return InstructionJsonlDownsampleSummary(
        input_path=str(resolved_input_path),
        output_path=str(resolved_output_path),
        total_line_count=total_line_count,
        target_line_count=target_line_count,
        written_line_count=written_line_count,
        keep_ratio=keep_ratio,
        alpha=alpha,
        dataset_group_count=len(counts_by_group),
        unique_stratum_count=len(plan_by_stratum),
    )


def _collect_stratum_counts(input_path: Path) -> tuple[int, dict[str, dict[str, int]]]:
    total_line_count = 0
    counts_by_group: dict[str, dict[str, int]] = defaultdict(dict)
    for _, payload, _ in _iter_instruction_payloads(input_path):
        dataset_group, protein_bucket = _instruction_stratum_key(payload)
        group_counts = counts_by_group[dataset_group]
        group_counts[protein_bucket] = group_counts.get(protein_bucket, 0) + 1
        total_line_count += 1
    return total_line_count, {group: dict(counts) for group, counts in counts_by_group.items()}


def _build_stratum_plan(
    counts_by_group: Mapping[str, Mapping[str, int]],
    *,
    group_targets: Mapping[str, int],
    alpha: float,
) -> dict[StratumKey, _StratumPlan]:
    plan_by_stratum: dict[StratumKey, _StratumPlan] = {}
    for dataset_group, counts_by_bucket in counts_by_group.items():
        quotas = _allocate_stratum_quotas(
            counts_by_bucket,
            target_count=group_targets[dataset_group],
            alpha=alpha,
        )
        for protein_bucket, total_count in counts_by_bucket.items():
            stratum_key = (dataset_group, protein_bucket)
            plan_by_stratum[stratum_key] = _StratumPlan(
                total_count=total_count,
                keep_count=quotas[protein_bucket],
                phase_numerator=_stable_phase_numerator(dataset_group, protein_bucket),
            )
    return plan_by_stratum


def _write_downsampled_jsonl(
    input_path: Path,
    output_path: Path,
    *,
    plan_by_stratum: Mapping[StratumKey, _StratumPlan],
) -> None:
    seen_by_stratum: dict[StratumKey, int] = defaultdict(int)
    written_line_count = 0
    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line_number, raw_line in enumerate(source, 1):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number} in {input_path}.") from exc
            stratum_key = _instruction_stratum_key(payload)
            plan = plan_by_stratum[stratum_key]
            seen_count = seen_by_stratum[stratum_key]
            if _should_keep_occurrence(
                seen_count=seen_count,
                total_count=plan.total_count,
                keep_count=plan.keep_count,
                phase_numerator=plan.phase_numerator,
            ):
                target.write(raw_line if raw_line.endswith("\n") else f"{raw_line}\n")
                written_line_count += 1
            seen_by_stratum[stratum_key] = seen_count + 1

    expected_line_count = sum(plan.keep_count for plan in plan_by_stratum.values())
    if written_line_count != expected_line_count:
        raise RuntimeError(
            "Downsampled instruction.jsonl line count drifted during write. "
            f"Expected {expected_line_count}, wrote {written_line_count}."
        )


def _iter_instruction_payloads(input_path: Path) -> Iterator[tuple[int, dict[str, object], str]]:
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number} in {input_path}.") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected a JSON object at line {line_number} in {input_path}.")
            yield line_number, payload, raw_line


def _instruction_stratum_key(payload: Mapping[str, object]) -> StratumKey:
    metadata = payload.get("metadata")
    normalized_metadata = metadata if isinstance(metadata, Mapping) else {}
    dataset_group = _normalize_bucket_value(normalized_metadata.get("dataset_group")) or "unknown"
    protein_bucket = (
        _normalize_bucket_value(normalized_metadata.get("product"))
        or _normalize_bucket_value(payload.get("product"))
        or _normalize_bucket_value(payload.get("description"))
        or _normalize_bucket_value(payload.get("accession"))
        or "unknown-protein"
    )
    return dataset_group, protein_bucket


def _normalize_bucket_value(value: object) -> str:
    if value is None:
        return ""
    normalized = " ".join(str(value).strip().split())
    return normalized.lower()


def _allocate_group_targets(
    counts_by_group: Mapping[str, Mapping[str, int]],
    target_line_count: int,
) -> dict[str, int]:
    group_totals = {group: sum(counts.values()) for group, counts in counts_by_group.items()}
    global_total = sum(group_totals.values())
    min_group_targets = {group: len(counts) for group, counts in counts_by_group.items()}
    minimum_required = sum(min_group_targets.values())
    if target_line_count < minimum_required:
        raise ValueError(
            "keep_ratio is too small to preserve at least one example for every dataset_group/product bucket. "
            f"Need at least {minimum_required} lines but target is {target_line_count}."
        )

    raw_group_targets = {
        group: group_totals[group] * target_line_count / global_total
        for group in counts_by_group
    }
    group_targets = {
        group: min(group_totals[group], max(min_group_targets[group], int(math.floor(raw_group_targets[group]))))
        for group in counts_by_group
    }
    current_total = sum(group_targets.values())
    ordered_groups = sorted(counts_by_group)

    while current_total < target_line_count:
        progress = False
        for group in sorted(
            ordered_groups,
            key=lambda name: (
                raw_group_targets[name] - math.floor(raw_group_targets[name]),
                group_totals[name] - group_targets[name],
                name,
            ),
            reverse=True,
        ):
            if group_targets[group] >= group_totals[group]:
                continue
            group_targets[group] += 1
            current_total += 1
            progress = True
            if current_total == target_line_count:
                break
        if not progress:
            break

    while current_total > target_line_count:
        progress = False
        for group in sorted(
            ordered_groups,
            key=lambda name: (
                raw_group_targets[name] - math.floor(raw_group_targets[name]),
                group_targets[name] - min_group_targets[name],
                name,
            ),
        ):
            if group_targets[group] <= min_group_targets[group]:
                continue
            group_targets[group] -= 1
            current_total -= 1
            progress = True
            if current_total == target_line_count:
                break
        if not progress:
            raise RuntimeError("Unable to reconcile dataset-group targets with the requested keep_ratio.")

    return group_targets


def _allocate_stratum_quotas(
    counts_by_bucket: Mapping[str, int],
    *,
    target_count: int,
    alpha: float,
) -> dict[str, int]:
    total_count = sum(counts_by_bucket.values())
    minimum_required = len(counts_by_bucket)
    if target_count < minimum_required:
        raise ValueError(
            "target_count is too small to preserve one example per protein bucket. "
            f"Need at least {minimum_required} but received {target_count}."
        )
    if target_count >= total_count:
        return dict(counts_by_bucket)

    continuous_target = float(target_count)

    def continuous_total(scale: float) -> float:
        total = 0.0
        for count in counts_by_bucket.values():
            raw = scale * math.pow(count, alpha)
            if raw < 1.0:
                raw = 1.0
            elif raw > count:
                raw = float(count)
            total += raw
        return total

    lower_bound = 0.0
    upper_bound = 1.0
    while continuous_total(upper_bound) < continuous_target:
        upper_bound *= 2.0

    for _ in range(80):
        midpoint = (lower_bound + upper_bound) / 2.0
        if continuous_total(midpoint) >= continuous_target:
            upper_bound = midpoint
        else:
            lower_bound = midpoint

    quotas: dict[str, int] = {}
    fractional_remainders: list[tuple[float, int, str]] = []
    base_total = 0
    for protein_bucket, count in counts_by_bucket.items():
        raw = upper_bound * math.pow(count, alpha)
        if raw < 1.0:
            raw = 1.0
        elif raw > count:
            raw = float(count)
        quota = max(1, min(count, int(math.floor(raw))))
        quotas[protein_bucket] = quota
        base_total += quota
        if quota < count:
            fractional_remainders.append((raw - quota, count, protein_bucket))

    remaining = target_count - base_total
    if remaining < 0:
        raise RuntimeError("Base quota allocation exceeded the target_count.")
    if remaining > len(fractional_remainders):
        raise RuntimeError("Quota allocation ran out of expandable protein buckets.")

    fractional_remainders.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    for _, _, protein_bucket in fractional_remainders[:remaining]:
        quotas[protein_bucket] += 1

    if sum(quotas.values()) != target_count:
        raise RuntimeError("Failed to match the requested target_count during quota allocation.")
    return quotas


def _stable_phase_numerator(dataset_group: str, protein_bucket: str) -> int:
    digest = hashlib.blake2b(
        f"{dataset_group}\u241f{protein_bucket}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big") % SYSTEMATIC_PHASE_DENOMINATOR


def _should_keep_occurrence(
    *,
    seen_count: int,
    total_count: int,
    keep_count: int,
    phase_numerator: int,
) -> bool:
    if keep_count <= 0:
        return False
    if keep_count >= total_count:
        return True
    denominator = total_count * SYSTEMATIC_PHASE_DENOMINATOR
    previous_bucket = (
        seen_count * keep_count * SYSTEMATIC_PHASE_DENOMINATOR + phase_numerator * total_count
    ) // denominator
    current_bucket = (
        (seen_count + 1) * keep_count * SYSTEMATIC_PHASE_DENOMINATOR + phase_numerator * total_count
    ) // denominator
    return current_bucket > previous_bucket
