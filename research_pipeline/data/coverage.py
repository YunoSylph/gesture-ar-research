from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from research_pipeline.data.schema import ManifestRecord
from research_pipeline.data.tensors import LandmarkTensor


@dataclass(frozen=True)
class CoverageReport:
    """Summary of a coverage-based manifest filter."""

    min_coverage: float
    total: int
    kept: int
    dropped: int
    mean_coverage: float
    kept_by_target: dict[str, int]
    dropped_by_target: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_coverage": self.min_coverage,
            "total": self.total,
            "kept": self.kept,
            "dropped": self.dropped,
            "mean_coverage": self.mean_coverage,
            "kept_by_target": self.kept_by_target,
            "dropped_by_target": self.dropped_by_target,
        }


def tensor_coverage(tensor: LandmarkTensor) -> float:
    """Fraction of frames with a detected hand (valid landmark mask)."""

    mask = np.asarray(tensor.sequence_mask, dtype=bool)
    return float(mask.mean()) if mask.size else 0.0


def filter_records_by_coverage(
    records: list[ManifestRecord],
    coverage_of: Callable[[ManifestRecord], float],
    *,
    min_coverage: float = 0.85,
) -> tuple[list[ManifestRecord], CoverageReport]:
    """Keep records whose detection coverage meets ``min_coverage``.

    ``coverage_of`` maps a record to its coverage (so the pure filtering/report
    logic stays independent of how tensors are loaded). Low-coverage clips (hand
    out of frame / motion blur) produce mostly-empty tensors and are poor
    training data, so they are dropped here before merge/training.
    """

    kept: list[ManifestRecord] = []
    kept_by: Counter[str] = Counter()
    dropped_by: Counter[str] = Counter()
    coverages: list[float] = []
    for record in records:
        coverage = coverage_of(record)
        coverages.append(coverage)
        if coverage >= min_coverage:
            kept.append(record)
            kept_by[record.target_label] += 1
        else:
            dropped_by[record.target_label] += 1
    report = CoverageReport(
        min_coverage=min_coverage,
        total=len(records),
        kept=len(kept),
        dropped=len(records) - len(kept),
        mean_coverage=float(np.mean(coverages)) if coverages else 0.0,
        kept_by_target=dict(kept_by),
        dropped_by_target=dict(dropped_by),
    )
    return kept, report
