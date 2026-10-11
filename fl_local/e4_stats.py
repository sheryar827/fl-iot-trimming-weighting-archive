"""Frozen Q1-E4 paired-analysis rules.

Only the seven prospective seeds may enter the 16-test decision family.  The
three previously observed seeds are allowed only in the separately labelled
ten-seed descriptive estimates.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import numpy as np


BLOCK_ORDER = ("T1", "T2", "T5", "T6")
NEW_SEEDS = (127, 131, 137, 139, 149, 151, 157)
EXISTING_SEEDS = (42, 123, 7)
ALL_SEEDS = (7, 42, 123, 127, 131, 137, 139, 149, 151, 157)
ARM_ORDER = (
    "FedAvg",
    "UnweightedFedAvg",
    "FedTrimmedAvg",
    "WeightedFedTrimmedAvg",
)
CONTRASTS = (
    ("FT-UW", "FedTrimmedAvg", "UnweightedFedAvg", "positive"),
    ("WT-FA", "WeightedFedTrimmedAvg", "FedAvg", "positive"),
    ("FA-UW", "FedAvg", "UnweightedFedAvg", "negative"),
    ("WT-FT", "WeightedFedTrimmedAvg", "FedTrimmedAvg", "negative"),
)
COMPARISON_ORDER = tuple(
    (block, name) for block in BLOCK_ORDER for name, _, _, _ in CONTRASTS
)
FAMILY_ALPHA = 0.05


def _finite_array(values: Iterable[object], label: str) -> np.ndarray:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    if array.ndim != 1 or array.size < 2:
        raise ValueError(f"{label} requires at least two paired differences")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite values")
    return array


def paired_summary(values: Iterable[object], *, include_test: bool) -> dict:
    """Return a JSON-safe paired summary in raw F1 units.

    A zero-variance nonzero vector has a degenerate point interval and p=0.  Its
    t statistic is represented as ``None`` rather than an infinity.  A vector
    of exact zeros has p=1 and CI [0, 0].
    """
    from scipy.stats import t as student_t

    delta = _finite_array(values, "paired differences")
    n = int(delta.size)
    constant = bool(np.all(delta == delta[0]))
    mean = float(delta[0]) if constant else float(np.mean(delta))
    sd = 0.0 if constant else float(np.std(delta, ddof=1))
    median = float(np.median(delta))
    minimum = float(np.min(delta))
    maximum = float(np.max(delta))
    positive = int(np.count_nonzero(delta > 0.0))
    zero = int(np.count_nonzero(delta == 0.0))
    negative = int(np.count_nonzero(delta < 0.0))
    if positive + zero + negative != n:
        raise ValueError("sign counts do not cover the paired differences")

    if sd == 0.0:
        ci_low = mean
        ci_high = mean
        if mean == 0.0:
            statistic = 0.0
            p_value = 1.0
            test_status = "zero_mean_zero_sd"
        else:
            statistic = None
            p_value = 0.0
            test_status = "nonzero_mean_zero_sd"
    else:
        standard_error = sd / math.sqrt(n)
        critical = float(student_t.ppf(0.975, df=n - 1))
        statistic = mean / standard_error
        p_value = float(2.0 * student_t.sf(abs(statistic), df=n - 1))
        ci_low = mean - critical * standard_error
        ci_high = mean + critical * standard_error
        test_status = "finite"

    out = {
        "n": n,
        "df": n - 1,
        "mean": mean,
        "sd": sd,
        "median": median,
        "minimum": minimum,
        "maximum": maximum,
        "positive": positive,
        "zero": zero,
        "negative": negative,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
    }
    if include_test:
        out.update(
            {
                "t_statistic": statistic,
                "raw_p": p_value,
                "test_status": test_status,
            }
        )
    return out


def holm_adjust(
    raw_p_by_comparison: Mapping[tuple[str, str], object]
) -> dict[tuple[str, str], float]:
    """Holm FWER adjustment with the frozen comparison order as tie-breaker."""
    if set(raw_p_by_comparison) != set(COMPARISON_ORDER):
        missing = set(COMPARISON_ORDER) - set(raw_p_by_comparison)
        extra = set(raw_p_by_comparison) - set(COMPARISON_ORDER)
        raise ValueError(f"Holm family must be exactly 16 tests; missing={missing} extra={extra}")
    canonical_index = {key: i for i, key in enumerate(COMPARISON_ORDER)}
    values: dict[tuple[str, str], float] = {}
    for key, value in raw_p_by_comparison.items():
        p_value = float(value)
        if not math.isfinite(p_value) or not 0.0 <= p_value <= 1.0:
            raise ValueError(f"invalid p-value for {key}: {value!r}")
        values[key] = p_value
    ordered = sorted(values, key=lambda key: (values[key], canonical_index[key]))
    adjusted: dict[tuple[str, str], float] = {}
    running = 0.0
    family_size = len(ordered)
    for rank, key in enumerate(ordered):
        candidate = min(1.0, (family_size - rank) * values[key])
        running = max(running, candidate)
        adjusted[key] = running
    return {key: adjusted[key] for key in COMPARISON_ORDER}


def decision(mean: object, adjusted_p: object, desired: str) -> str:
    effect = float(mean)
    p_value = float(adjusted_p)
    if not math.isfinite(effect) or not math.isfinite(p_value):
        raise ValueError("decision inputs must be finite")
    if desired not in {"positive", "negative"}:
        raise ValueError(f"unknown desired direction: {desired}")
    significant = p_value < FAMILY_ALPHA
    desired_sign = effect > 0.0 if desired == "positive" else effect < 0.0
    opposite_sign = effect < 0.0 if desired == "positive" else effect > 0.0
    if significant and desired_sign:
        return "supported"
    if significant and opposite_sign:
        return "reversal"
    return "inconclusive"


def paired_differences(
    final_f1: Mapping[tuple[str, str, int], object],
    *,
    seeds: Iterable[int],
) -> dict[tuple[str, str], list[float]]:
    """Require complete same-seed quartets and return canonical contrasts."""
    seed_order = tuple(int(seed) for seed in seeds)
    expected = {
        (block, algorithm, seed)
        for block in BLOCK_ORDER
        for algorithm in ARM_ORDER
        for seed in seed_order
    }
    if set(final_f1) != expected:
        missing = expected - set(final_f1)
        extra = set(final_f1) - expected
        raise ValueError(f"incomplete or out-of-scope quartets; missing={missing} extra={extra}")
    numeric = {
        key: float(value)
        for key, value in final_f1.items()
    }
    if not all(math.isfinite(value) for value in numeric.values()):
        raise ValueError("Final F1 inputs must all be finite")
    out: dict[tuple[str, str], list[float]] = {}
    for block in BLOCK_ORDER:
        for name, left, right, _ in CONTRASTS:
            out[(block, name)] = [
                numeric[(block, left, seed)] - numeric[(block, right, seed)]
                for seed in seed_order
            ]
    return out


def prospective_analysis(
    final_f1: Mapping[tuple[str, str, int], object]
) -> list[dict]:
    """Run only the frozen seven-new-seed, 16-comparison decision family."""
    differences = paired_differences(final_f1, seeds=NEW_SEEDS)
    summaries = {
        key: paired_summary(values, include_test=True)
        for key, values in differences.items()
    }
    adjusted = holm_adjust({key: row["raw_p"] for key, row in summaries.items()})
    direction = {(block, name): desired for block in BLOCK_ORDER for name, _, _, desired in CONTRASTS}
    rows = []
    for key in COMPARISON_ORDER:
        row = dict(summaries[key])
        row.update(
            {
                "block": key[0],
                "contrast": key[1],
                "desired_direction": direction[key],
                "holm_p": adjusted[key],
                "decision": decision(row["mean"], adjusted[key], direction[key]),
                "deltas": list(differences[key]),
            }
        )
        rows.append(row)
    return rows


def combined_estimates(
    final_f1: Mapping[tuple[str, str, int], object]
) -> list[dict]:
    """Ten-seed descriptive estimates; deliberately emits no p-values."""
    differences = paired_differences(final_f1, seeds=ALL_SEEDS)
    rows = []
    for key in COMPARISON_ORDER:
        summary = paired_summary(differences[key], include_test=False)
        forbidden = {"raw_p", "holm_p", "t_statistic", "decision"}
        if forbidden.intersection(summary):
            raise RuntimeError("combined estimates must not emit decision-test fields")
        summary.update(
            {
                "block": key[0],
                "contrast": key[1],
                "label": "combined data-informed estimate",
                "deltas": list(differences[key]),
            }
        )
        rows.append(summary)
    return rows
