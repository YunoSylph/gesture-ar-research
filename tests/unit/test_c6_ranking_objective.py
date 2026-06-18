from __future__ import annotations

import numpy as np
import pytest

from research_pipeline.cli.run_c5_calibrated_recognition import ScorePack, _rank_candidates
from research_pipeline.labels import TARGET_LABELS, label_to_index
from research_pipeline.models.calibrated import CalibratedFusionConfig

NUM_LABELS = len(TARGET_LABELS)


def _confident_but_half_wrong_pack() -> ScorePack:
    # Every sample is predicted no_gesture with confidence 0.9, but only half are
    # actually no_gesture -> a deliberately miscalibrated pack with ECE ~ 0.4.
    n = 40
    scores = np.full((n, NUM_LABELS), 0.1 / (NUM_LABELS - 1), dtype=np.float64)
    scores[:, label_to_index("no_gesture")] = 0.9
    y_true = ["no_gesture"] * (n // 2) + ["point_2f"] * (n // 2)
    return ScorePack(y_true=y_true, c1_scores=scores, c3_scores=scores.copy(), latencies_ms=[])


def test_ranking_reports_ece_and_penalty_lowers_safety_score() -> None:
    pack = _confident_but_half_wrong_pack()
    candidates = [CalibratedFusionConfig()]

    base = _rank_candidates(pack, candidates, {"ece_penalty": 0.0})
    penalised = _rank_candidates(pack, candidates, {"ece_penalty": 1.0})

    ece = base[0]["expected_calibration_error"]
    assert ece > 0.1  # the pack is genuinely miscalibrated
    # macro_score does not depend on the calibration penalty.
    assert penalised[0]["macro_score"] == pytest.approx(base[0]["macro_score"])
    # ece_penalty * ece is subtracted from the safety objective.
    assert penalised[0]["safety_score"] == pytest.approx(base[0]["safety_score"] - 1.0 * ece)
