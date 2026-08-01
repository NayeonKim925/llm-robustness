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
