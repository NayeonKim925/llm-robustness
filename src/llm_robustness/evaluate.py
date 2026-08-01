"""Generation-based evaluation harness (requires transformers + torch).

This is the *correct* evaluation path: it renders prompts with the model's chat
template, generates a short completion, and parses the label with the robust
:func:`labels.parse_label`. It reports accuracy, macro-F1 and per-class F1 for
every context condition, always alongside the majority-class and random
baselines.

The harness is model-agnostic: point it at a base model, or at a base model
plus a LoRA adapter directory.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from . import labels, metrics, prompts
from .config import CONDITIONS


def load_model(base_model: str, adapter_dir: Optional[str] = None):
    """Load a base model (+ optional LoRA adapter) and tokenizer for inference."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, device_map="auto")
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer


def predict_one(model, tokenizer, context_text: str, max_new_tokens: int = 10) -> str:
    """Greedy-decode a label for one example (chat-formatted)."""
    import torch

    text = prompts.build_inference_text(context_text, tokenizer)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id)
    new_tokens = out[0][input_ids.shape[-1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return labels.parse_label(raw)


def evaluate_condition(model, tokenizer, dataset: List[dict], condition: str,
                       max_new_tokens: int = 10) -> Dict:
    """Evaluate one context condition over a dataset split."""
    golds, preds, invalid = [], [], 0
    for d in dataset:
        ctx = d.get(condition)
        if ctx is None:  # condition undefined for this example (e.g. no conflict)
            continue
        pred = predict_one(model, tokenizer, ctx, max_new_tokens)
        gold_id = labels.to_id(d["label"])
        if pred == labels.INVALID:
            invalid += 1
            # Count invalid as a wrong prediction rather than dropping it, so the
            # denominator is honest. Map to a sentinel that never equals a gold.
            preds.append(-1)
        else:
            preds.append(labels.to_id(pred))
        golds.append(gold_id)

    n_classes = len(labels.LABELS)
    maj = metrics.majority_baseline(golds, n_classes)
    return {
        "condition": condition,
        "n": len(golds),
        "invalid": invalid,
        "golds": golds,
        "preds": preds,
        "accuracy": metrics.accuracy(golds, preds),
        "macro_f1": metrics.macro_f1(golds, preds, n_classes),
        "per_class_f1": metrics.per_class_f1(golds, preds, n_classes),
        "pred_distribution": dict(zip(
            labels.LABELS, metrics.prediction_distribution(
                [p for p in preds if p >= 0], n_classes))),
        "baseline_majority_acc": maj["accuracy"],
        "baseline_majority_macro_f1": maj["macro_f1"],
    }


def evaluate_all(model, tokenizer, dataset: List[dict],
                 conditions: List[str] = CONDITIONS, max_new_tokens: int = 10) -> Dict:
    return {c: evaluate_condition(model, tokenizer, dataset, c, max_new_tokens)
            for c in conditions}


def run(base_model: str, adapter_dir: Optional[str], conditions_file: str,
        split: str, out_path: str, max_new_tokens: int = 10) -> Dict:
    """End-to-end: load data + model, evaluate every condition, save JSON."""
    with open(conditions_file) as f:
        dataset = [d for d in json.load(f) if d["split"] == split]
    model, tokenizer = load_model(base_model, adapter_dir)
    results = evaluate_all(model, tokenizer, dataset, CONDITIONS, max_new_tokens)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return results
