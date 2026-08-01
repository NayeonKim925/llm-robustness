"""Unit tests for the pure-stdlib metrics (run: python -m pytest -q)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llm_robustness import metrics


def test_accuracy_perfect_and_zero():
    assert metrics.accuracy([0, 1, 2], [0, 1, 2]) == 1.0
    assert metrics.accuracy([0, 1, 2], [1, 2, 0]) == 0.0


def test_macro_f1_perfect():
    golds = [0, 1, 2, 3]
    assert abs(metrics.macro_f1(golds, golds, 4) - 1.0) < 1e-9


def test_per_class_f1_unpredicted_class_is_zero():
    # Class 3 (comment) is the majority but never predicted -> F1 == 0.
    golds = [3, 3, 3, 1]
    preds = [1, 1, 1, 1]
    f1 = metrics.per_class_f1(golds, preds, 4)
    assert f1[3] == 0.0


def test_majority_baseline_matches_class_frequency():
    golds = [3, 3, 3, 1]  # 3 is majority -> acc 0.75
    base = metrics.majority_baseline(golds, 4)
    assert base["class"] == 3
    assert abs(base["accuracy"] - 0.75) < 1e-9


def test_ece_perfectly_calibrated_is_zero():
    # Confidence exactly equals accuracy within each bin -> ECE 0.
    confs = [1.0, 1.0, 0.0, 0.0]
    correct = [1, 1, 0, 0]  # bin@1.0 acc 1.0; bin@0.0 acc 0.0
    assert metrics.expected_calibration_error(confs, correct, n_bins=10) < 1e-9


def test_ece_detects_overconfidence():
    confs = [0.99, 0.99, 0.99, 0.99]
    correct = [1, 0, 0, 0]  # 99% confident, 25% accurate
    ece = metrics.expected_calibration_error(confs, correct, n_bins=10)
    assert abs(ece - (0.99 - 0.25)) < 1e-6


def test_confidence_right_vs_wrong():
    confs = [0.9, 0.8, 0.7, 0.6]
    correct = [1, 1, 0, 0]
    right, wrong = metrics.confidence_when_right_vs_wrong(confs, correct)
    assert abs(right - 0.85) < 1e-9
    assert abs(wrong - 0.65) < 1e-9
