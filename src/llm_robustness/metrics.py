"""Evaluation and calibration metrics, implemented in pure Python.

Everything here depends only on the standard library so that analysis can run
in any environment (no numpy / scikit-learn / torch required). The functions
operate on integer-encoded gold/prediction lists (see :mod:`labels`).

Two families of metric are provided:

* **Classification** -- accuracy, per-class and macro F1, plus trivial
  baselines (majority class, stratified random). The baselines matter a great
  deal for this project: on RumourEval the ``comment`` class alone is ~80% of
  the dev set, so a model must be compared against a "always predict comment"
  baseline before any accuracy number can be interpreted.
* **Calibration** -- Expected Calibration Error (ECE) and reliability-diagram
  bins, computed from (confidence, correct) pairs.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Dict, List, Sequence, Tuple

# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


def accuracy(golds: Sequence[int], preds: Sequence[int]) -> float:
    if not golds:
        return 0.0
    return sum(int(g == p) for g, p in zip(golds, preds)) / len(golds)


def per_class_f1(golds: Sequence[int], preds: Sequence[int], num_classes: int) -> List[float]:
    """F1 for each class id in ``range(num_classes)`` (0 when unsupported)."""
    scores = []
    for c in range(num_classes):
        tp = sum(int(g == c and p == c) for g, p in zip(golds, preds))
        fp = sum(int(g != c and p == c) for g, p in zip(golds, preds))
        fn = sum(int(g == c and p != c) for g, p in zip(golds, preds))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        scores.append(f1)
    return scores


def macro_f1(golds: Sequence[int], preds: Sequence[int], num_classes: int) -> float:
    scores = per_class_f1(golds, preds, num_classes)
    return sum(scores) / len(scores) if scores else 0.0


def majority_baseline(golds: Sequence[int], num_classes: int) -> Dict[str, float]:
    """Metrics for always predicting the most frequent gold class."""
    if not golds:
        return {"accuracy": 0.0, "macro_f1": 0.0, "class": -1}
    most_common = Counter(golds).most_common(1)[0][0]
    preds = [most_common] * len(golds)
    return {
        "accuracy": accuracy(golds, preds),
        "macro_f1": macro_f1(golds, preds, num_classes),
        "class": most_common,
    }


def random_baseline(golds: Sequence[int], num_classes: int, seed: int = 0,
                    stratified: bool = True) -> Dict[str, float]:
    """Metrics for a random classifier.

    ``stratified`` draws predictions from the empirical gold distribution;
    otherwise predictions are uniform over classes.
    """
    if not golds:
        return {"accuracy": 0.0, "macro_f1": 0.0}
    rng = random.Random(seed)
    if stratified:
        population = list(golds)
        preds = [rng.choice(population) for _ in golds]
    else:
        preds = [rng.randrange(num_classes) for _ in golds]
    return {
        "accuracy": accuracy(golds, preds),
        "macro_f1": macro_f1(golds, preds, num_classes),
    }


def prediction_distribution(preds: Sequence[int], num_classes: int) -> List[int]:
    counts = Counter(preds)
    return [counts.get(c, 0) for c in range(num_classes)]


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------


def reliability_bins(confidences: Sequence[float], correct: Sequence[int],
                     n_bins: int = 10) -> List[dict]:
    """Bin (confidence, correct) pairs for a reliability diagram.

    Returns one dict per non-empty bin with keys: ``lo``, ``hi``, ``count``,
    ``avg_confidence``, ``accuracy``.
    """
    bins = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [
            i for i, c in enumerate(confidences)
            # last bin is closed on the right so conf == 1.0 lands somewhere
            if (lo <= c < hi) or (b == n_bins - 1 and c == hi)
        ]
        if not idx:
            continue
        conf = sum(confidences[i] for i in idx) / len(idx)
        acc = sum(correct[i] for i in idx) / len(idx)
        bins.append({
            "lo": lo, "hi": hi, "count": len(idx),
            "avg_confidence": conf, "accuracy": acc,
        })
    return bins


def expected_calibration_error(confidences: Sequence[float], correct: Sequence[int],
                               n_bins: int = 10) -> float:
    """Expected Calibration Error: sum_b (n_b / N) * |acc_b - conf_b|."""
    n = len(confidences)
    if n == 0:
        return 0.0
    ece = 0.0
    for b in reliability_bins(confidences, correct, n_bins):
        ece += (b["count"] / n) * abs(b["accuracy"] - b["avg_confidence"])
    return ece


def mcnemar_test(correct_a: Sequence[int], correct_b: Sequence[int],
                 continuity: bool = True) -> Dict[str, float]:
    """Paired McNemar test between two classifiers on the *same* examples.

    ``correct_a`` / ``correct_b`` are aligned 0/1 correctness indicators (same
    example at each index). Returns the discordant counts, the chi-square
    statistic (with continuity correction by default) and a p-value. For small
    discordant totals (b + c < 25) an exact two-sided binomial p-value is used
    instead, which is the standard recommendation.
    """
    if len(correct_a) != len(correct_b):
        raise ValueError("inputs must be aligned and equal length")
    b = sum(int(a == 1 and bb == 0) for a, bb in zip(correct_a, correct_b))
    c = sum(int(a == 0 and bb == 1) for a, bb in zip(correct_a, correct_b))
    n_disc = b + c

    if n_disc == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0, "method": "none"}

    if n_disc < 25:
        # Exact two-sided binomial test, p = 0.5.
        k = min(b, c)
        tail = sum(_binom_pmf(n_disc, i, 0.5) for i in range(0, k + 1))
        p = min(1.0, 2.0 * tail)
        return {"b": b, "c": c, "statistic": float(min(b, c)),
                "p_value": p, "method": "exact_binomial"}

    diff = abs(b - c) - (1.0 if continuity else 0.0)
    stat = (diff * diff) / n_disc
    return {"b": b, "c": c, "statistic": stat,
            "p_value": _chi2_sf_1df(stat), "method": "chi2_continuity"}


def _binom_pmf(n: int, k: int, p: float) -> float:
    from math import comb
    return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def _chi2_sf_1df(x: float) -> float:
    """Survival function of chi-square with 1 dof = erfc(sqrt(x/2))."""
    import math
    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x / 2.0))


def confidence_when_right_vs_wrong(confidences: Sequence[float],
                                   correct: Sequence[int]) -> Tuple[float, float]:
    """Mean confidence on correct vs incorrect predictions.

    A well-calibrated classifier is *more* confident when right. The reverse
    (more confident when wrong) is the miscalibration pattern this project set
    out to probe.
    """
    right = [c for c, ok in zip(confidences, correct) if ok]
    wrong = [c for c, ok in zip(confidences, correct) if not ok]
    mean_right = sum(right) / len(right) if right else float("nan")
    mean_wrong = sum(wrong) / len(wrong) if wrong else float("nan")
    return mean_right, mean_wrong
