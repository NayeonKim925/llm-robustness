#!/usr/bin/env python3
"""Fine-tune a LoRA adapter (GPU). Thin CLI over llm_robustness.train.

Examples::

    # standard fine-tune on the clean 'useful' condition
    python scripts/run_finetune.py --config configs/experiment.yaml \
        --output-dir result/qwen_1.5b_ft

    # adversarial fine-tune (mixes perturbed contexts into training)
    python scripts/run_finetune.py --config configs/experiment.yaml \
        --adversarial --output-dir result/qwen_1.5b_adv
"""

from __future__ import annotations

import argparse

from llm_robustness import train
from llm_robustness.config import Config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--base-model")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--adversarial", action="store_true")
    p.add_argument("--imbalance-strategy", choices=["none", "weighted_sampler"])
    p.add_argument("--seed", type=int)
    args = p.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.base_model:
        cfg.base_model = args.base_model
    if args.adversarial:
        cfg.adversarial = True
    if args.imbalance_strategy:
        cfg.imbalance_strategy = args.imbalance_strategy
    if args.seed is not None:
        cfg.seed = args.seed

    train.train(cfg, args.output_dir)


if __name__ == "__main__":
    main()
