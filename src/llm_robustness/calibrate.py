"""Confidence / calibration measurement (requires transformers + torch).

This module replaces the original ``confidence scores`` notebook, which had
three bugs that together invalidated its numbers:

1. **No chat template.** It fed the fine-tuned chat model a bare
   ``"Rumor: ...\nAnswer:"`` string, though the adapter was trained on the full
   ``<|im_start|>system ...`` format. Here we always use
   :func:`prompts.build_inference_text`.

2. **Wrong label-token ids.** It scored ``tokenizer.encode(label)[0]`` -- the
   *no-leading-space* token -- but in context the model emits the
   *leading-space* variant (e.g. ``" deny"``), a different BPE id. This module
   scores the correct in-context tokens; see :func:`label_token_ids`.

3. **Contaminated split.** Its dev set was built from a merged key map and
   silently included training threads. Calibration here runs on whichever split
   the caller passes from the clean ``context_conditions.json``.

Confidence is the softmax over the four label tokens' next-token logits at the
generation position (a constrained, well-defined distribution over the label
set), and we report ECE and reliability bins via :mod:`metrics`.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from . import labels, metrics, prompts
from .config import CONDITIONS


def label_token_ids(tokenizer) -> Dict[str, int]:
    """Map each label to the token id the model actually emits in context.

    We tokenize with a leading space and take the first piece, because after
    the assistant turn / an ``Answer:`` cue the label surfaces as ``" support"``
    etc. Asserts the four ids are distinct so a silent collision can't pass.
    """
    ids = {}
    for name in labels.LABELS:
        pieces = tokenizer.encode(" " + name, add_special_tokens=False)
        ids[name] = pieces[0]
    assert len(set(ids.values())) == len(labels.LABELS), \
        f"label tokens collide: {ids}"
    return ids


def score_one(model, tokenizer, context_text: str, tok_ids: Dict[str, int]) -> Dict:
    """Return the constrained label distribution + confidence stats for one input."""
    import torch
    import torch.nn.functional as F

    text = prompts.build_inference_text(context_text, tokenizer)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    label_logits = torch.tensor([logits[tok_ids[lab]] for lab in labels.LABELS])
    probs = F.softmax(label_logits, dim=0).tolist()

    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    top, second = order[0], order[1]
    ent = -sum(p * (p and _log(p)) for p in probs)
    return {
        "pred": labels.LABELS[top],
        "prob": {labels.LABELS[i]: probs[i] for i in range(len(probs))},
        "confidence": probs[top],
        "entropy": ent,
        "margin": probs[top] - probs[second],
    }


def _log(x: float) -> float:
    import math
    return math.log(x)


def calibrate_condition(model, tokenizer, dataset: List[dict], condition: str,
                        tok_ids: Dict[str, int], n_bins: int = 10) -> Dict:
    confs, correct, golds, preds = [], [], [], []
    for d in dataset:
        ctx = d.get(condition)
        if ctx is None:
            continue
        s = score_one(model, tokenizer, ctx, tok_ids)
        is_correct = int(s["pred"] == d["label"])
        confs.append(s["confidence"])
        correct.append(is_correct)
        golds.append(labels.to_id(d["label"]))
        preds.append(labels.to_id(s["pred"]))

    n_classes = len(labels.LABELS)
    right, wrong = metrics.confidence_when_right_vs_wrong(confs, correct)
    return {
        "condition": condition,
        "n": len(confs),
        "accuracy": metrics.accuracy(golds, preds),
        "macro_f1": metrics.macro_f1(golds, preds, n_classes),
        "avg_confidence": sum(confs) / len(confs) if confs else 0.0,
        "ece": metrics.expected_calibration_error(confs, correct, n_bins),
        "mean_conf_when_right": right,
        "mean_conf_when_wrong": wrong,
        "reliability_bins": metrics.reliability_bins(confs, correct, n_bins),
    }


def run(base_model: str, adapter_dir: Optional[str], conditions_file: str,
        split: str, out_path: str, n_bins: int = 10) -> Dict:
    from .evaluate import load_model

    with open(conditions_file) as f:
        dataset = [d for d in json.load(f) if d["split"] == split]
    model, tokenizer = load_model(base_model, adapter_dir)
    tok_ids = label_token_ids(tokenizer)
    results = {c: calibrate_condition(model, tokenizer, dataset, c, tok_ids, n_bins)
               for c in CONDITIONS}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return results
