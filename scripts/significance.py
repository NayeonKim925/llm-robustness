#!/usr/bin/env python3
"""Paired McNemar tests comparing context conditions (CPU).

For each model, compares the ``reply_only`` condition against every perturbed
condition using a paired McNemar test on per-example correctness. Paired data
comes from the confidence dumps, which store every condition's prediction for
the *same* ``reply_id`` on each row (so alignment is exact).

NOTE: the available confidence dumps were produced by the original (flawed)
confidence pipeline, so the *substantive* p-values below describe those
predictions. The point of this script is the reusable, tested significance
machinery (``metrics.mcnemar_test``); re-run it on corrected predictions from
``evaluate.py`` (which keeps all examples aligned) for headline claims.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llm_robustness import metrics  # noqa: E402

DUMPS = [
    ("qwen1.5b_ft", "results/raw_predictions/confidence_results_1.5b_ft.json"),
    ("qwen1.5b_adv", "results/raw_predictions/confidence_results_1.5b_adv.json"),
]
CONDITIONS = ["reply_only", "useful", "irrelevant", "conflicting", "mixed", "lexical"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/significance.json")
    args = ap.parse_args()

    report = {}
    for model, path in DUMPS:
        if not os.path.exists(path):
            print(f"[skip] {path}")
            continue
        rows = json.load(open(path))
        base = [int(r["reply_only_correct"]) for r in rows]
        model_report = {}
        for cond in CONDITIONS:
            if cond == "reply_only":
                continue
            other = [int(r[f"{cond}_correct"]) for r in rows]
            model_report[f"reply_only_vs_{cond}"] = metrics.mcnemar_test(base, other)
        report[model] = model_report

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)

    for model, r in report.items():
        print(f"\n=== {model}: McNemar vs reply_only ===")
        print(f"{'comparison':28s} {'b':>5} {'c':>5} {'stat':>8} {'p':>10} {'method':>16}")
        for name, res in r.items():
            print(f"{name:28s} {res['b']:5d} {res['c']:5d} {res['statistic']:8.3f} "
                  f"{res['p_value']:10.4g} {res['method']:>16}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
