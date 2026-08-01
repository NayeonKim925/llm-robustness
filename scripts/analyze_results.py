#!/usr/bin/env python3
"""Turn the raw prediction dumps into an honest, baseline-anchored report.

This script does **not** run any model. It re-reads the prediction files that
were produced on GPU (committed to the repo) and recomputes, from scratch and
against trivial baselines:

* accuracy, macro-F1 and per-class F1 per context condition;
* the majority-class and stratified-random baselines on the *same* gold set;
* the prediction distribution (to expose class collapse);
* calibration (ECE, reliability bins, confidence-when-right-vs-wrong) for the
  confidence dumps.

Outputs go to ``results/`` (CSV/JSON) and ``figures/`` (PNG). Everything is
deterministic. Figures are skipped gracefully if matplotlib is unavailable.

Why this exists
---------------
The original notebooks reported accuracy/F1 without any baseline. On this
dataset the ``comment`` class alone is ~80% of dev, so an "always predict
comment" classifier scores ~0.80 accuracy. Reading the raw numbers next to
that baseline is the whole point: it makes the class-collapse failure mode
impossible to miss.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

# Make ``src`` importable when run as a plain script.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llm_robustness import labels, metrics  # noqa: E402

NUM_CLASSES = len(labels.LABELS)

# Classification dumps: {condition: {"golds": [...], "preds": [...]}} with
# integer labels. (model_name, path)
CLASSIFICATION_DUMPS = [
    ("qwen0.5B-zeroshot", "results/raw_predictions/experiment_results_0.5B.json"),
    ("qwen1.5B-adv-FT", "results/raw_predictions/experiment_results_1.5B_adv.json"),
    ("qwen3B-zeroshot", "results/raw_predictions/experiment_results_3B_zeroshot.json"),
    ("qwen3B-adv-FT", "results/raw_predictions/experiment_results_3B_adv.json"),
]

# Confidence dumps: list of per-example rows with ``{cond}_pred`` (string),
# ``{cond}_conf`` (float), ``{cond}_correct`` (0/1), ``true_label`` (string).
# NOTE: these were produced by the original confidence notebook, which (a) fed
# the chat model a prompt WITHOUT its chat template, (b) read logits for the
# no-leading-space label tokens, and (c) evaluated on a split that mixed in
# training data. They are reported here only to *quantify* those artifacts.
CONFIDENCE_DUMPS = [
    ("qwen_1.5b_ft_ORIGINAL_FLAWED", "results/raw_predictions/confidence_results_1.5b_ft.json"),
    ("qwen_1.5b_adv_ORIGINAL_FLAWED", "results/raw_predictions/confidence_results_1.5b_adv.json"),
]

CONDITIONS = ["reply_only", "useful", "irrelevant", "conflicting", "mixed", "lexical"]


def analyze_classification(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    out = {}
    for cond, obj in data.items():
        golds = obj.get("golds", [])
        preds = obj.get("preds", [])
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


def make_figures(all_cls: dict, all_conf: dict, fig_dir: str) -> list:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[figures] matplotlib unavailable ({e}); skipping figures.")
        return []

    made = []

    # 1) Accuracy vs majority baseline, per model, reply_only condition.
    models = list(all_cls.keys())
    accs = [all_cls[m].get("reply_only", {}).get("accuracy", 0) for m in models]
    majs = [all_cls[m].get("reply_only", {}).get("baseline_majority_acc", 0) for m in models]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(models))
    ax.bar([i - 0.2 for i in x], accs, width=0.4, label="model")
    ax.bar([i + 0.2 for i in x], majs, width=0.4, label="majority baseline")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("accuracy (reply_only)")
    ax.set_title("Models collapse below the majority-class baseline")
    ax.legend()
    fig.tight_layout()
    p = os.path.join(fig_dir, "accuracy_vs_baseline.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    made.append(p)

    # 2) Prediction distribution (class collapse) for the first model.
    m0 = models[0]
    dist = all_cls[m0].get("reply_only", {}).get("pred_distribution", {})
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(list(dist.keys()), list(dist.values()), color="#c0392b")
    ax.set_ylabel("# predictions")
    ax.set_title(f"Prediction collapse ({m0}, reply_only)\n"
                 f"comment is ~80% of gold but is never predicted")
    fig.tight_layout()
    p = os.path.join(fig_dir, "prediction_collapse.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    made.append(p)

    # 3) Reliability diagram for a confidence dump (flagged as flawed).
    if all_conf:
        name, conf = next(iter(all_conf.items()))
        bins = conf.get("reply_only", {}).get("reliability_bins", [])
        if bins:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
            ax.plot([b["avg_confidence"] for b in bins],
                    [b["accuracy"] for b in bins], "o-", label="observed")
            ax.set_xlabel("confidence")
            ax.set_ylabel("accuracy")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f"Reliability diagram\n{name} (reply_only) -- ORIGINAL FLAWED pipeline")
            ax.legend()
            fig.tight_layout()
            p = os.path.join(fig_dir, "reliability_original_flawed.png")
            fig.savefig(p, dpi=130)
            plt.close(fig)
            made.append(p)
    return made


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    all_cls = {}
    for model, path in CLASSIFICATION_DUMPS:
        if os.path.exists(path):
            all_cls[model] = analyze_classification(path)
        else:
            print(f"[skip] {path} not found")

    all_conf = {}
    for model, path in CONFIDENCE_DUMPS:
        if os.path.exists(path):
            all_conf[model] = analyze_confidence(path)
        else:
            print(f"[skip] {path} not found")

    with open(os.path.join(args.results_dir, "classification_analysis.json"), "w") as f:
        json.dump(all_cls, f, indent=2)
    with open(os.path.join(args.results_dir, "calibration_analysis.json"), "w") as f:
        json.dump(all_conf, f, indent=2)
    table = write_baseline_table(all_cls, args.results_dir)

    # Console summary.
    print("\n=== Classification vs. trivial baselines (reply_only) ===")
    print(f"{'model':32s} {'acc':>7} {'macroF1':>8} {'majAcc':>7} "
          f"{'#cls':>5} {'commentF1':>10}")
    for model, conds in all_cls.items():
        m = conds.get("reply_only")
        if m:
            print(f"{model:32s} {m['accuracy']:7.3f} {m['macro_f1']:8.3f} "
                  f"{m['baseline_majority_acc']:7.3f} "
                  f"{m['n_distinct_predicted_classes']:5d} {m['per_class_f1'][3]:10.3f}")

    print("\n=== Calibration of ORIGINAL (flawed) confidence pipeline (reply_only) ===")
    for name, conf in all_conf.items():
        m = conf.get("reply_only", {})
        print(f"{name:32s} acc={m.get('accuracy', 0):.3f} "
              f"ECE={m.get('ece', 0):.3f} "
              f"conf_right={m.get('mean_conf_when_right', float('nan')):.3f} "
              f"conf_wrong={m.get('mean_conf_when_wrong', float('nan')):.3f}")

    print(f"\nWrote: {table}")
    if not args.no_figures:
        for p in make_figures(all_cls, all_conf, args.figures_dir):
            print(f"Wrote figure: {p}")


if __name__ == "__main__":
    main()
