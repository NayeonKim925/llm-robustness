"""Console entry points (CPU-only tools).

Installed as ``llmr-dataset`` / ``llmr-analyze`` / ``llmr-figures`` /
``llmr-significance`` via ``[project.scripts]`` in ``pyproject.toml``. The
``scripts/*.py`` files are thin wrappers over these same functions.
"""

from __future__ import annotations

import argparse
import json

from . import data, report, viz


def build_dataset_main() -> None:
    ap = argparse.ArgumentParser(description="Build context_conditions.json from raw trees.")
    ap.add_argument("--data-root", default="rumoureval-2019-training-data")
    ap.add_argument("--out", default="rumoureval-2019-training-data/context_conditions.json")
    ap.add_argument("--card", default="results/data_card.json")
    args = ap.parse_args()

    import os
    dataset = data.build_dataset(args.data_root)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(dataset, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    summary = data.summarize(dataset)
    os.makedirs(os.path.dirname(args.card) or ".", exist_ok=True)
    json.dump({"n_total": len(dataset), "by_split": summary}, open(args.card, "w"), indent=2)

    print(f"Wrote {len(dataset)} examples -> {args.out}")
    for split, info in summary.items():
        print(f"  {split:8s} n={info['n']:5d}  labels={info['labels']}  "
              f"valid_cc={info['valid_cc_frac']}  valid_ti={info['valid_ti_frac']}")


def analyze_main() -> None:
    ap = argparse.ArgumentParser(description="Recompute metrics + baselines from dumps.")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--figures-dir", default="figures")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    all_cls, all_conf = report.run_analysis(args.results_dir)

    print("\n=== Classification vs. trivial baselines (reply_only) ===")
    print(f"{'model':32s} {'acc':>7} {'macroF1':>8} {'majAcc':>7} {'#cls':>5} {'commentF1':>10}")
    for model, conds in all_cls.items():
        m = conds.get("reply_only")
        if m:
            print(f"{model:32s} {m['accuracy']:7.3f} {m['macro_f1']:8.3f} "
                  f"{m['baseline_majority_acc']:7.3f} "
                  f"{m['n_distinct_predicted_classes']:5d} {m['per_class_f1'][3]:10.3f}")

    print("\n=== Calibration of ORIGINAL (flawed) confidence pipeline (reply_only) ===")
    for name, conf in all_conf.items():
        m = conf.get("reply_only", {})
        print(f"{name:32s} acc={m.get('accuracy', 0):.3f} ECE={m.get('ece', 0):.3f} "
              f"conf_right={m.get('mean_conf_when_right', float('nan')):.3f} "
              f"conf_wrong={m.get('mean_conf_when_wrong', float('nan')):.3f}")

    if not args.no_figures:
        for p in viz.make_all(args.results_dir, args.figures_dir):
            print(f"Wrote figure: {p}")


def figures_main() -> None:
    ap = argparse.ArgumentParser(description="Render result figures from the analysis JSON.")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--figures-dir", default="figures")
    args = ap.parse_args()
    for p in viz.make_all(args.results_dir, args.figures_dir):
        print(f"Wrote {p}")


def significance_main() -> None:
    ap = argparse.ArgumentParser(description="Paired McNemar tests across conditions.")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()
    rep = report.compute_significance(args.results_dir)
    for model, r in rep.items():
        print(f"\n=== {model}: McNemar vs reply_only ===")
        print(f"{'comparison':28s} {'b':>5} {'c':>5} {'stat':>8} {'p':>10} {'method':>16}")
        for name, res in r.items():
            print(f"{name:28s} {res['b']:5d} {res['c']:5d} {res['statistic']:8.3f} "
                  f"{res['p_value']:10.4g} {res['method']:>16}")
