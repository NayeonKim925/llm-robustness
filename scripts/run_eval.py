#!/usr/bin/env python3
"""Generation-based evaluation across all context conditions (GPU).

Examples::

    # zero-shot base model
    python scripts/run_eval.py --base-model Qwen/Qwen2.5-3B-Instruct \
        --out results/eval_qwen3b_zeroshot.json

    # a fine-tuned adapter
    python scripts/run_eval.py --base-model Qwen/Qwen2.5-1.5B-Instruct \
        --adapter result/qwen_1.5b_adv/final \
        --out results/eval_qwen1.5b_adv.json
"""

from __future__ import annotations

import argparse

from llm_robustness import evaluate
from llm_robustness.config import Config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--split", default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    cfg = Config.from_yaml(args.config)
    split = args.split or cfg.eval_split
    results = evaluate.run(
        base_model=args.base_model, adapter_dir=args.adapter,
        conditions_file=cfg.conditions_file, split=split,
        out_path=args.out, max_new_tokens=cfg.max_new_tokens)

    print(f"{'condition':12s} {'n':>5} {'acc':>7} {'macroF1':>8} {'majAcc':>7}")
    for cond, m in results.items():
        print(f"{cond:12s} {m['n']:5d} {m['accuracy']:7.3f} {m['macro_f1']:8.3f} "
              f"{m['baseline_majority_acc']:7.3f}")


if __name__ == "__main__":
    main()
