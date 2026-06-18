from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

import numpy as np

Reducer = Callable[[np.ndarray], float]


@dataclass(slots=True)
class IntervalEstimate:
    point: float
    low: float
    high: float
    n: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PairedComparison:
    """Paired comparison of a per-unit metric between a baseline and a method.

    ``delta`` is ``mean(method) - mean(baseline)``; for cost-like metrics
    (false-action cost, false positives) a negative ``delta`` is the desired
    improvement. ``prob_improvement`` is the bootstrap fraction of resamples in
    which the method beats the baseline in the configured direction, and
    ``p_value`` is the McNemar exact p-value over per-unit win/loss outcomes.
    """

    baseline_mean: float
    method_mean: float
    delta: float
    delta_ci_low: float
    delta_ci_high: float
    prob_improvement: float
    p_value: float
    n: int
    lower_is_better: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_array(values: Sequence[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=np.float64)


def bootstrap_ci(
    values: Sequence[float],
    *,
    reducer: Reducer = np.mean,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 13,
) -> IntervalEstimate:
    """Percentile bootstrap confidence interval for a per-unit metric."""

    data = _as_array(values)
    n = int(data.shape[0])
    if n == 0:
        return IntervalEstimate(point=0.0, low=0.0, high=0.0, n=0)
    point = float(reducer(data))
    if n == 1:
        return IntervalEstimate(point=point, low=point, high=point, n=1)
    rng = np.random.default_rng(seed)
    samples = np.empty((n_resamples,), dtype=np.float64)
    for index in range(n_resamples):
        resampled = data[rng.integers(0, n, size=n)]
        samples[index] = float(reducer(resampled))
    alpha = (1.0 - confidence) / 2.0
    low = float(np.quantile(samples, alpha))
    high = float(np.quantile(samples, 1.0 - alpha))
    return IntervalEstimate(point=point, low=low, high=high, n=n)


def paired_comparison(
    baseline: Sequence[float],
    method: Sequence[float],
    *,
    lower_is_better: bool = True,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 13,
) -> PairedComparison:
    """Compare a method against a baseline on the same units (e.g. sequences).

    ``baseline[i]`` and ``method[i]`` must be the metric for the same unit ``i``
    under each pipeline, so the difference is paired and the comparison controls
    for per-unit difficulty -- exactly the ablation setting where every method
    sees the identical replay sequences.
    """

    base = _as_array(baseline)
    meth = _as_array(method)
    if base.shape != meth.shape:
        raise ValueError(f"baseline and method must be paired, got {base.shape} vs {meth.shape}.")
    n = int(base.shape[0])
    if n == 0:
        return PairedComparison(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0, lower_is_better)

    diff = meth - base
    delta = float(diff.mean())
    improvement = -diff if lower_is_better else diff

    if n == 1:
        better = float(improvement[0] > 0)
        return PairedComparison(
            baseline_mean=float(base.mean()),
            method_mean=float(meth.mean()),
            delta=delta,
            delta_ci_low=delta,
            delta_ci_high=delta,
            prob_improvement=better,
            p_value=1.0,
            n=1,
            lower_is_better=lower_is_better,
        )

    rng = np.random.default_rng(seed)
    deltas = np.empty((n_resamples,), dtype=np.float64)
    for index in range(n_resamples):
        picks = rng.integers(0, n, size=n)
        deltas[index] = float(diff[picks].mean())
    alpha = (1.0 - confidence) / 2.0
    ci_low = float(np.quantile(deltas, alpha))
    ci_high = float(np.quantile(deltas, 1.0 - alpha))
    prob_improvement = float(np.mean((-deltas if lower_is_better else deltas) > 0.0))

    wins = int(np.sum(improvement > 0))
    losses = int(np.sum(improvement < 0))
    p_value = mcnemar_exact_p(wins, losses)

    return PairedComparison(
        baseline_mean=float(base.mean()),
        method_mean=float(meth.mean()),
        delta=delta,
        delta_ci_low=ci_low,
        delta_ci_high=ci_high,
        prob_improvement=prob_improvement,
        p_value=p_value,
        n=n,
        lower_is_better=lower_is_better,
    )


def mcnemar_exact_p(wins: int, losses: int) -> float:
    """Two-sided exact McNemar p-value over discordant pairs (binomial, p=0.5).

    ``wins`` and ``losses`` are the counts of units where the method beat / lost
    to the baseline; concordant (tied) pairs are ignored, as in McNemar's test.
    Dependency-free so the evaluation stack keeps no SciPy requirement.
    """

    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    # Two-sided tail of Binomial(n, 0.5).
    tail = sum(_binom_coeff(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return float(min(1.0, 2.0 * tail))


def _binom_coeff(n: int, k: int) -> float:
    from math import comb

    return float(comb(n, k))
