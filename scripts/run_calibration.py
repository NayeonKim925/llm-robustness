#!/usr/bin/env python3
"""Corrected confidence / calibration measurement across conditions (GPU).

Fixes the three bugs of the original confidence notebook (chat template,
leading-space label tokens, clean split). See llm_robustness.calibrate.

Example::

    python scripts/run_calibration.py --base-model Qwen/Qwen2.5-1.5B-Instruct \
        --adapter result/qwen_1.5b_ft/final \
        --out results/calibration_qwen1.5b_ft.json
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llm_robustness.config import Config  # noqa: E402
from llm_robustness import calibrate  # noqa: E402


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
    results = calibrate.run(
        base_model=args.base_model, adapter_dir=args.adapter,
        conditions_file=cfg.conditions_file, split=split,
        out_path=args.out, n_bins=cfg.calibration_bins)

    print(f"{'condition':12s} {'acc':>7} {'ECE':>7} {'conf|right':>11} {'conf|wrong':>11}")
    for cond, m in results.items():
        print(f"{cond:12s} {m['accuracy']:7.3f} {m['ece']:7.3f} "
              f"{m['mean_conf_when_right']:11.3f} {m['mean_conf_when_wrong']:11.3f}")


if __name__ == "__main__":
    main()
