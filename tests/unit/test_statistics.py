from __future__ import annotations

import numpy as np

from research_pipeline.evaluation.statistics import (
    bootstrap_ci,
    mcnemar_exact_p,
    paired_comparison,
)


def test_bootstrap_ci_brackets_mean() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(5.0, 1.0, size=200).tolist()
    estimate = bootstrap_ci(values, seed=1)
    assert estimate.n == 200
    assert estimate.low < estimate.point < estimate.high
    assert abs(estimate.point - 5.0) < 0.3


def test_bootstrap_ci_handles_empty_and_single() -> None:
    assert bootstrap_ci([]).n == 0
    single = bootstrap_ci([3.0])
    assert single.point == 3.0 and single.low == 3.0 and single.high == 3.0


def test_paired_comparison_detects_consistent_reduction() -> None:
    # Method has strictly lower cost on every paired unit -> clear improvement.
    baseline = [5.0, 6.0, 4.0, 7.0, 5.5, 6.2, 4.8, 5.9]
    method = [2.0, 2.5, 1.8, 3.0, 2.2, 2.7, 2.1, 2.4]
    result = paired_comparison(baseline, method, lower_is_better=True, seed=2)
    assert result.delta < 0
    assert result.delta_ci_high < 0  # whole CI below zero -> reduction is significant
    assert result.prob_improvement > 0.95
    assert result.p_value < 0.05


def test_paired_comparison_no_effect_is_not_significant() -> None:
    values = [3.0, 4.0, 5.0, 2.0, 6.0, 3.5]
    result = paired_comparison(values, list(values), lower_is_better=True, seed=3)
    assert result.delta == 0.0
    assert result.p_value == 1.0
    assert 0.0 <= result.prob_improvement <= 1.0


def test_paired_comparison_requires_matching_shape() -> None:
    import pytest

    with pytest.raises(ValueError):
        paired_comparison([1.0, 2.0], [1.0], lower_is_better=True)


def test_mcnemar_symmetry_and_bounds() -> None:
    assert mcnemar_exact_p(0, 0) == 1.0
    assert mcnemar_exact_p(10, 10) == 1.0
    p = mcnemar_exact_p(9, 1)
    assert 0.0 < p < 0.05
    # Order of wins/losses must not change the two-sided p-value.
    assert mcnemar_exact_p(9, 1) == mcnemar_exact_p(1, 9)
