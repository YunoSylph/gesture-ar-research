"""Aggregate many live-session summaries into one reproducible live-evaluation report.

The per-session view (``summarize_task_records``) is a quick triage of a single
webcam run. It is not, on its own, scientific evidence: a single session is noisy
and not reproducible. This module aggregates a *set* of logged sessions into the
same action-level metrics used by the offline replay (cost-weighted action
precision/recall, required-action recall, decision latency, task-success rate),
plus session-quality stats (FPS, processing latency, detection coverage), so that
live behaviour becomes a trackable, reproducible measurement rather than a visual
impression. The live protocol complements -- it does not replace -- the replay
ablation that carries the primary thesis claim.
"""

from __future__ import annotations

from statistics import median
from typing import Any


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    return float(median(values)) if values else 0.0


def _ground_truth_reports(reports: list[dict[str, Any]], task: str) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for report in reports:
        task_report = report.get("tasks", {}).get(task)
        if task_report and isinstance(task_report.get("ground_truth"), dict):
            found.append(task_report["ground_truth"])
    return found


def aggregate_session_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate ``summarize_task_records`` outputs across sessions.

    Each entry in ``reports`` is one session report (the dict returned by
    ``summarize_task_records``). Action-level metrics are aggregated only over the
    sessions that carry a ground-truth scenario for the task; session-quality
    metrics are aggregated over all sessions.
    """

    num_sessions = len(reports)
    sessions = [report.get("session", {}) for report in reports]

    quality = {
        "fps_mean": round(_mean([float(s.get("fps", {}).get("mean", 0.0)) for s in sessions]), 3),
        "processing_ms_p95_mean": round(_mean([float(s.get("processing_ms", {}).get("p95", 0.0)) for s in sessions]), 3),
        "detection_rate_mean": round(_mean([float(s.get("detection_rate_mean", 0.0)) for s in sessions]), 4),
        "confidence_mean": round(_mean([float(s.get("confidence_mean", 0.0)) for s in sessions]), 4),
        "total_frames": int(sum(int(s.get("frames", 0)) for s in sessions)),
    }

    task_ids = sorted({task for report in reports for task in report.get("tasks", {})})
    per_task: dict[str, Any] = {}
    all_ground_truth: list[dict[str, Any]] = []
    for task in task_ids:
        ground_truth = _ground_truth_reports(reports, task)
        if not ground_truth:
            continue
        all_ground_truth.extend(ground_truth)
        per_task[task] = {
            "sessions": len(ground_truth),
            "task_success_rate": round(_mean([1.0 if g.get("task_success") else 0.0 for g in ground_truth]), 4),
            "action_precision_mean": round(_mean([float(g.get("action_precision", 0.0)) for g in ground_truth]), 4),
            "action_recall_mean": round(_mean([float(g.get("action_recall", 0.0)) for g in ground_truth]), 4),
            "required_action_recall_mean": round(
                _mean([float(g.get("required_action_recall", 0.0)) for g in ground_truth]), 4
            ),
            "latency_abs_ms_median": round(
                _median([float(g.get("latency_abs_ms", {}).get("median", 0.0)) for g in ground_truth]), 3
            ),
        }

    overall = {
        "scored_task_runs": len(all_ground_truth),
        "task_success_rate": round(_mean([1.0 if g.get("task_success") else 0.0 for g in all_ground_truth]), 4),
        "action_precision_mean": round(_mean([float(g.get("action_precision", 0.0)) for g in all_ground_truth]), 4),
        "action_recall_mean": round(_mean([float(g.get("action_recall", 0.0)) for g in all_ground_truth]), 4),
    }

    return {
        "protocol": "aggregated_live_session_metrics",
        "num_sessions": num_sessions,
        "quality": quality,
        "tasks": per_task,
        "overall": overall,
        "notes": [
            "Aggregated over real webcam sessions; complements the replay ablation, not a replacement.",
            "Action-level metrics require a ground-truth task scenario per session.",
            "Single-session numbers are triage only; report aggregated metrics over several sessions.",
        ],
    }
