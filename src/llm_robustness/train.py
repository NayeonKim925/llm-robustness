"""LoRA supervised fine-tuning (requires trl + peft + torch).

Cleaned version of the original fine-tuning notebook, with two substantive
additions:

* **Class-imbalance handling.** ``comment`` is ~72% of the training replies, so
  a plain SFT run has little incentive to learn the minority stances. With
  ``imbalance_strategy="weighted_sampler"`` we draw examples with probability
  inversely proportional to class frequency, which is the cheapest fix that
  does not throw data away.
* **Adversarial training option.** With ``adversarial=True`` the perturbed
  conditions (``conflicting`` / ``mixed`` / ``lexical``) are mixed into the
  training set alongside the clean ``useful`` condition, so the model sees
  distractor context during training.

All formatting goes through :func:`prompts.format_for_sft`, so the training
text matches inference exactly.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import List

from . import data, prompts
from .config import Config, set_seed


def _load_fit_rows(cfg: Config) -> List[dict]:
    """Training rows = the `fit` portion of a thread-level train/val split.

    The `val` portion (and the `dev`/test split) are held out. See
    ``data.train_val_split`` for why the partition is by thread, not by reply.
    """
    with open(cfg.conditions_file) as f:
        full = json.load(f)
    fit, val = data.train_val_split(full, cfg.val_fraction, cfg.seed)
    print(f"train/val split (seed {cfg.seed}): fit={len(fit)} rows, "
          f"val held out={len(val)} rows")
    return fit


def _build_examples(rows: List[dict], cfg: Config, tokenizer) -> List[dict]:
    """Render (text, label) SFT examples for the configured condition(s)."""
    conditions = [cfg.train_condition]
    if cfg.adversarial:
        conditions += ["conflicting", "mixed", "lexical"]

    examples = []
    for d in rows:
        for cond in conditions:
            ctx = d.get(cond)
            if ctx is None:
                continue
            examples.append({
                "text": prompts.format_for_sft(ctx, d["label"], tokenizer),
                "label": d["label"],
            })
    return examples


def _sample_weights(examples: List[dict]) -> List[float]:
    """Inverse-frequency weight per example, for a WeightedRandomSampler."""
    freq = Counter(e["label"] for e in examples)
    n = len(examples)
    return [n / (len(freq) * freq[e["label"]]) for e in examples]


def train(cfg: Config, output_dir: str) -> str:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    set_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = _load_fit_rows(cfg)
    examples = _build_examples(rows, cfg, tokenizer)
    print(f"train examples: {len(examples)} | label balance: "
          f"{dict(Counter(e['label'] for e in examples))}")
    hf_dataset = Dataset.from_list(examples)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, torch_dtype=torch.float16, device_map="auto")
    model = get_peft_model(model, LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules, bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        lr_scheduler_type="cosine",
        fp16=True, logging_steps=50, save_strategy="epoch",
        seed=cfg.seed, report_to="none",
    )

    trainer = SFTTrainer(
        model=model, args=sft_config, train_dataset=hf_dataset,
        processing_class=tokenizer)

    # Optional inverse-frequency sampling to counter the class imbalance.
    if cfg.imbalance_strategy == "weighted_sampler":
        from torch.utils.data import WeightedRandomSampler
        weights = _sample_weights(examples)
        trainer._get_train_sampler = lambda *a, **k: WeightedRandomSampler(  # type: ignore
            weights, num_samples=len(weights), replacement=True)

    trainer.train()

    final_dir = os.path.join(output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    with open(os.path.join(final_dir, "train_config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    print(f"saved adapter -> {final_dir}")
    return final_dir
