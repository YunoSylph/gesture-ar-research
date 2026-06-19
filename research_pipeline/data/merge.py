from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Any

from research_pipeline.data.manifest import ensure_unique_sample_ids
from research_pipeline.data.schema import ManifestRecord
from research_pipeline.labels import TARGET_LABELS


@dataclass(frozen=True)
class MergeReport:
    """Coverage / balance summary of a merged 7-class manifest."""

    total: int
    by_target: dict[str, int]
    by_source: dict[str, int]
    by_target_source: dict[str, dict[str, int]]
    missing_targets: tuple[str, ...]
    dropped: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "by_target": self.by_target,
            "by_source": self.by_source,
            "by_target_source": self.by_target_source,
            "missing_targets": list(self.missing_targets),
            "dropped": self.dropped,
        }


def _cap_subset(
    records: list[ManifestRecord], limit: int | None, rng: random.Random
) -> tuple[list[ManifestRecord], int]:
    """Deterministically keep at most ``limit`` records, preserving their order."""

    if limit is None or len(records) <= limit:
        return records, 0
    indices = list(range(len(records)))
    rng.shuffle(indices)
    keep = set(indices[:limit])
    kept = [record for index, record in enumerate(records) if index in keep]
    return kept, len(records) - limit


def merge_manifests(
    records: list[ManifestRecord],
    *,
    max_per_class: int | None = None,
    max_per_class_per_source: int | None = None,
    seed: int = 13,
) -> tuple[list[ManifestRecord], MergeReport]:
    """Merge records into one 7-class manifest with optional class balancing.

    ``max_per_class_per_source`` caps each (target_label, source_dataset) group so
    a large source (e.g. Jester) cannot dominate a class; ``max_per_class`` then
    caps the total per target_label. Both subsample deterministically for a given
    ``seed`` while preserving the input concatenation order. Domain metadata on
    each record (source_dataset, capture_domain, ...) is carried through unchanged.
    """

    order = {id(record): index for index, record in enumerate(records)}
    dropped = 0
    working = records

    if max_per_class_per_source is not None:
        per_source: dict[tuple[str, str], list[ManifestRecord]] = {}
        for record in working:
            per_source.setdefault((record.target_label, record.source_dataset), []).append(record)
        kept: list[ManifestRecord] = []
        for key in sorted(per_source):
            rng = random.Random(f"{seed}:{key[0]}:{key[1]}")
            subset, removed = _cap_subset(per_source[key], max_per_class_per_source, rng)
            kept.extend(subset)
            dropped += removed
        working = kept

    if max_per_class is not None:
        per_target: dict[str, list[ManifestRecord]] = {}
        for record in working:
            per_target.setdefault(record.target_label, []).append(record)
        kept = []
        for label in sorted(per_target):
            rng = random.Random(f"{seed}:{label}")
            subset, removed = _cap_subset(per_target[label], max_per_class, rng)
            kept.extend(subset)
            dropped += removed
        working = kept

    working = sorted(working, key=lambda record: order[id(record)])
    ensure_unique_sample_ids(working)

    by_target = dict(Counter(record.target_label for record in working))
    by_source = dict(Counter(record.source_dataset for record in working))
    by_target_source: dict[str, dict[str, int]] = {}
    for record in working:
        bucket = by_target_source.setdefault(record.target_label, {})
        bucket[record.source_dataset] = bucket.get(record.source_dataset, 0) + 1
    missing = tuple(label for label in TARGET_LABELS if label not in by_target)

    report = MergeReport(
        total=len(working),
        by_target=by_target,
        by_source=by_source,
        by_target_source=by_target_source,
        missing_targets=missing,
        dropped=dropped,
    )
    return working, report
