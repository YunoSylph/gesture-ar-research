from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from research_pipeline.data.schema import ManifestRecord, manifest_record_from_dict
from research_pipeline.utils.errors import SchemaError


def read_jsonl(path: str | Path, *, strict: bool = True) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise SchemaError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            try:
                records.append(manifest_record_from_dict(payload, strict=strict))
            except SchemaError as exc:
                raise SchemaError(f"{path}:{line_number}: {exc}") from exc
    return records


def write_jsonl(path: str | Path, records: Iterable[ManifestRecord]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def read_capture_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def ensure_unique_sample_ids(records: Iterable[ManifestRecord]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for record in records:
        if record.sample_id in seen:
            duplicates.add(record.sample_id)
        seen.add(record.sample_id)
    if duplicates:
        raise SchemaError(f"Duplicate sample_id values: {sorted(duplicates)}")

