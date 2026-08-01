"""Experiment configuration and reproducibility helpers.

Centralises the paths, hyper-parameters and -- importantly -- the random seed
that the original notebooks never set. ``set_seed`` seeds Python, NumPy and
Torch when they are available, so a run can be reproduced.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field, asdict
from typing import List, Optional

try:  # optional; config must import even in a bare stdlib environment
    import yaml
except Exception:  # pragma: no cover
    yaml = None


CONDITIONS: List[str] = ["reply_only", "useful", "irrelevant", "conflicting", "mixed", "lexical"]


@dataclass
class Config:
    # data
    data_root: str = "rumoureval-2019-training-data"
    conditions_file: str = "rumoureval-2019-training-data/context_conditions.json"
    train_split: str = "train"
    eval_split: str = "dev"

    # model / training
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    warmup_steps: int = 100

    # class-imbalance handling: "none" | "weighted_sampler"
    imbalance_strategy: str = "weighted_sampler"
    train_condition: str = "useful"  # which context condition to train on
    adversarial: bool = False  # if True, mix perturbed conditions into training

    # evaluation
    max_new_tokens: int = 10
    calibration_bins: int = 10

    # io
    output_dir: str = "result"
    results_dir: str = "results"
    figures_dir: str = "figures"

    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        if yaml is None:
            raise RuntimeError("PyYAML is required to load a YAML config.")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)


def set_seed(seed: int) -> None:
    """Seed all RNGs we might touch. No-ops for libraries that aren't installed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
