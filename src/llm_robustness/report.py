"""Baseline-anchored analysis of prediction dumps (no model, CPU only).

Re-reads the committed prediction files and recomputes -- against trivial
baselines -- accuracy / macro-F1 / per-class F1, the prediction distribution
(to expose class collapse), calibration (ECE, reliability, confidence
right-vs-wrong), and paired McNemar tests across conditions.

This module owns the logic; ``scripts/`` and the console entry points in
:mod:`cli` are thin wrappers over it. The original notebooks reported
accuracy/F1 with no baseline, which is exactly how a class collapse (``comment``
is ~80% of dev) stayed invisible.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Dict, List, Tuple

from . import labels, metrics

NUM_CLASSES = len(labels.LABELS)
CONDITIONS = ["reply_only", "useful", "irrelevant", "conflicting", "mixed", "lexical"]

# Classification dumps: {condition: {"golds": [...], "preds": [...]}} (int labels).
CLASSIFICATION_DUMPS: List[Tuple[str, str]] = [
    ("qwen0.5B-zeroshot", "results/raw_predictions/experiment_results_0.5B.json"),
    ("qwen1.5B-adv-FT", "results/raw_predictions/experiment_results_1.5B_adv.json"),
    ("qwen3B-zeroshot", "results/raw_predictions/experiment_results_3B_zeroshot.json"),
    ("qwen3B-adv-FT", "results/raw_predictions/experiment_results_3B_adv.json"),
]

# Confidence dumps (per-example rows). Produced by the ORIGINAL flawed pipeline
# (no chat template, wrong label-token ids, contaminated split); reported only
# to quantify the artifact.
CONFIDENCE_DUMPS: List[Tuple[str, str]] = [
    ("qwen_1.5b_ft_ORIGINAL_FLAWED", "results/raw_predictions/confidence_results_1.5b_ft.json"),
    ("qwen_1.5b_adv_ORIGINAL_FLAWED", "results/raw_predictions/confidence_results_1.5b_adv.json"),
]


def analyze_classification(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    out = {}
    for cond, obj in data.items():
        golds, preds = obj.get("golds", []), obj.get("preds", [])
        if not golds:
            continue
        maj = metrics.majority_baseline(golds, NUM_CLASSES)
        rnd = metrics.random_baseline(golds, NUM_CLASSES, seed=0, stratified=True)
        dist = metrics.prediction_distribution(preds, NUM_CLASSES)
        out[cond] = {
            "n": len(golds),
            "accuracy": metrics.accuracy(golds, preds),
            "macro_f1": metrics.macro_f1(golds, preds, NUM_CLASSES),
            "per_class_f1": metrics.per_class_f1(golds, preds, NUM_CLASSES),
            "pred_distribution": dict(zip(labels.LABELS, dist)),
            "baseline_majority_acc": maj["accuracy"],
            "baseline_majority_macro_f1": maj["macro_f1"],
            "baseline_random_acc": rnd["accuracy"],
            "n_distinct_predicted_classes": sum(1 for c in dist if c > 0),
        }
    return out


def analyze_confidence(path: str) -> dict:
    with open(path) as f:
        rows = json.load(f)
    out = {"n_examples": len(rows)}
    for cond in CONDITIONS:
        pred_key, conf_key, corr_key = f"{cond}_pred", f"{cond}_conf", f"{cond}_correct"
        if pred_key not in rows[0]:
            continue
        golds = [labels.to_id(r["true_label"]) for r in rows]
        preds = [labels.to_id(r[pred_key]) for r in rows]
        confs = [r[conf_key] for r in rows]
        correct = [int(r[corr_key]) for r in rows]
        mean_right, mean_wrong = metrics.confidence_when_right_vs_wrong(confs, correct)
        out[cond] = {
            "accuracy": metrics.accuracy(golds, preds),
            "macro_f1": metrics.macro_f1(golds, preds, NUM_CLASSES),
            "pred_distribution": dict(Counter(r[pred_key] for r in rows)),
            "avg_confidence": sum(confs) / len(confs),
            "ece": metrics.expected_calibration_error(confs, correct, n_bins=10),
            "mean_conf_when_right": mean_right,
            "mean_conf_when_wrong": mean_wrong,
            "reliability_bins": metrics.reliability_bins(confs, correct, n_bins=10),
        }
    return out


def write_baseline_table(all_cls: dict, out_dir: str) -> str:
    path = os.path.join(out_dir, "classification_vs_baselines.csv")
    with open(path, "w") as f:
        f.write("model,condition,n,accuracy,macro_f1,majority_acc,random_acc,"
                "n_classes_predicted,comment_f1\n")
        for model, conds in all_cls.items():
            for cond, m in conds.items():
                f.write(
                    f"{model},{cond},{m['n']},{m['accuracy']:.4f},{m['macro_f1']:.4f},"
                    f"{m['baseline_majority_acc']:.4f},{m['baseline_random_acc']:.4f},"
                    f"{m['n_distinct_predicted_classes']},{m['per_class_f1'][3]:.4f}\n"
                )
    return path


def run_analysis(results_dir: str = "results") -> Tuple[dict, dict]:
    """Compute + persist classification and calibration analyses; return both."""
    os.makedirs(results_dir, exist_ok=True)
    all_cls = {m: analyze_classification(p) for m, p in CLASSIFICATION_DUMPS if os.path.exists(p)}
    all_conf = {m: analyze_confidence(p) for m, p in CONFIDENCE_DUMPS if os.path.exists(p)}
    json.dump(all_cls, open(os.path.join(results_dir, "classification_analysis.json"), "w"), indent=2)
    json.dump(all_conf, open(os.path.join(results_dir, "calibration_analysis.json"), "w"), indent=2)
    write_baseline_table(all_cls, results_dir)
    return all_cls, all_conf


def compute_significance(results_dir: str = "results") -> dict:
    """Paired McNemar of reply_only vs each perturbed condition, per model.

    Uses the confidence dumps (each row has every condition's correctness for
    the same reply_id, so pairing is exact).
    """
    report = {}
    for model, path in CONFIDENCE_DUMPS:
        if not os.path.exists(path):
            continue
        rows = json.load(open(path))
        base = [int(r["reply_only_correct"]) for r in rows]
        report[model] = {
            f"reply_only_vs_{c}": metrics.mcnemar_test(base, [int(r[f"{c}_correct"]) for r in rows])
            for c in CONDITIONS if c != "reply_only"
        }
    json.dump(report, open(os.path.join(results_dir, "significance.json"), "w"), indent=2)
    return report
