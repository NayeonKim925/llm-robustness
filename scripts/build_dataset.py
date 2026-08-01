#!/usr/bin/env python3
"""Build ``context_conditions.json`` from the raw RumourEval-2019 trees.

Runs on CPU with only the standard library. Reproduces the six context
conditions per labelled reply and writes a single JSON file plus a small data
card summarising splits and label balance.

Usage::

    python scripts/build_dataset.py \
        --data-root rumoureval-2019-training-data \
        --out rumoureval-2019-training-data/context_conditions.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llm_robustness import data  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="rumoureval-2019-training-data")
    parser.add_argument("--out", default="rumoureval-2019-training-data/context_conditions.json")
    parser.add_argument("--card", default="results/data_card.json")
    args = parser.parse_args()

    dataset = data.build_dataset(args.data_root)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    summary = data.summarize(dataset)
    os.makedirs(os.path.dirname(args.card) or ".", exist_ok=True)
    with open(args.card, "w") as f:
        json.dump({"n_total": len(dataset), "by_split": summary}, f, indent=2)

    print(f"Wrote {len(dataset)} examples -> {args.out}")
    for split, info in summary.items():
        print(f"  {split:8s} n={info['n']:5d}  labels={info['labels']}  "
              f"valid_cc={info['valid_cc_frac']}  valid_ti={info['valid_ti_frac']}")
    print(f"Wrote data card -> {args.card}")


if __name__ == "__main__":
    main()
