"""RumourEval-2019 parsing and controlled context-condition construction.

This is a cleaned, importable version of the original
``dataset_parsing_preprocessing`` notebook. Behaviour is preserved; the changes
are engineering ones:

* paths are function arguments instead of hard-coded ``/home/ubuntu/...``;
* the dev/train split is decided per-``reply_id`` from the key files (the
  earlier ``'dev' if src_id in label_map`` check used a *merged* key map, so it
  labelled essentially every thread ``dev`` -- which is why the confidence
  pipeline ended up evaluating on training data);
* the parsing and condition-building logic is grouped into small functions.

The six context conditions
--------------------------
``reply_only``   target reply only (no context).
``useful``       source rumour + direct parent + target.
``irrelevant``   source + an on-topic but stance-free ``comment`` (lowest
                 Jaccard overlap with the target) + target.
``conflicting``  source + a real reply with the opposing stance + target.
``mixed``        source + parent + a conflicting reply + target.
``lexical``      a hard-coded distractor sentence that asserts the *opposite* of
                 the target's true stance + target (a label-aware adversarial
                 stress test; see ``LD_TEMPLATES``).
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional

STANCE_KEYWORDS = [
    "confirmed", "confirm", "true", "false", "fake", "real",
    "denied", "deny", "hoax", "verified", "debunked", "proven",
    "official", "breaking", "fact",
]

# Label-aware lexical distractors: each asserts the opposite of the true stance.
LD_TEMPLATES = {
    "support": "This has been widely denied and officially debunked by experts.",
    "deny": "Multiple credible sources have confirmed this to be completely true.",
    "query": "There is absolutely no doubt about this. It has been verified.",
    "comment": "This claim has been officially confirmed as false by authorities.",
}

CONFLICT_PREF = {
    "support": ["deny"],
    "deny": ["support"],
    "query": ["support", "deny"],
    "comment": ["deny", "support"],
}


# ---------------------------------------------------------------------------
# small text utilities
# ---------------------------------------------------------------------------

def has_stance_keyword(text: str) -> bool:
    return any(kw in text.lower() for kw in STANCE_KEYWORDS)


def jaccard_similarity(text_a: str, text_b: str) -> float:
    set_a, set_b = set(text_a.lower().split()), set(text_b.lower().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _load_key(path: str) -> Dict[str, str]:
    with open(path) as f:
        return json.load(f).get("subtaskaenglish", {})


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def parse_rumoureval(data_root: str) -> List[dict]:
    """Parse the Twitter + Reddit trees into flat reply records.

    Splits are assigned per-id: an id in ``dev-key.json`` is ``dev``, in
    ``train-key.json`` is ``train``, otherwise ``unknown``.
    """
    dev_key = _load_key(os.path.join(data_root, "dev-key.json"))
    train_key = _load_key(os.path.join(data_root, "train-key.json"))
    label_map = {**dev_key, **train_key}

    def split_of(reply_id: str) -> str:
        if reply_id in dev_key:
            return "dev"
        if reply_id in train_key:
            return "train"
        return "unknown"

    records: List[dict] = []

    # Reddit
    for subdir in ("reddit-dev-data", "reddit-training-data"):
        base = os.path.join(data_root, subdir)
        if not os.path.isdir(base):
            continue
        for thread_id in sorted(os.listdir(base)):
            thread_path = os.path.join(base, thread_id)
            src_dir = os.path.join(thread_path, "source-tweet")
            if not os.path.isdir(src_dir) or not os.listdir(src_dir):
                continue
            with open(os.path.join(src_dir, sorted(os.listdir(src_dir))[0])) as f:
                src = json.load(f)["data"]["children"][0]["data"]
            src_id = str(src.get("id", thread_id))
            records.append(dict(
                reply_id=src_id, thread_id=thread_id, parent_id=None, depth=-1,
                text=src.get("selftext") or src.get("title", ""),
                label=label_map.get(thread_id), platform="reddit",
                split=split_of(thread_id), is_source=True))
            rep_dir = os.path.join(thread_path, "replies")
            if not os.path.isdir(rep_dir):
                continue
            for rep_file in sorted(os.listdir(rep_dir)):
                with open(os.path.join(rep_dir, rep_file)) as f:
                    body = json.load(f)["data"]
                rep_id = str(body.get("id", rep_file.replace(".json", "")))
                records.append(dict(
                    reply_id=rep_id, thread_id=thread_id,
                    parent_id=str(body.get("parent_id", "").split("_")[-1]),
                    depth=body.get("depth", 0), text=body.get("body", ""),
                    label=label_map.get(rep_id), platform="reddit",
                    split=split_of(rep_id), is_source=False))

    # Twitter
    twitter_dir = os.path.join(data_root, "twitter-english")
    if os.path.isdir(twitter_dir):
        for event_id in sorted(os.listdir(twitter_dir)):
            event_path = os.path.join(twitter_dir, event_id)
            if not os.path.isdir(event_path):
                continue
            for thread_id in sorted(os.listdir(event_path)):
                thread_path = os.path.join(event_path, thread_id)
                src_dir = os.path.join(thread_path, "source-tweet")
                if not os.path.isdir(src_dir) or not os.listdir(src_dir):
                    continue
                with open(os.path.join(src_dir, sorted(os.listdir(src_dir))[0])) as f:
                    src = json.load(f)
                src_id = str(src.get("id", thread_id))
                records.append(dict(
                    reply_id=src_id, thread_id=thread_id, parent_id=None, depth=-1,
                    text=src.get("text", ""), label=label_map.get(src_id),
                    platform="twitter", split=split_of(src_id), is_source=True))
                rep_dir = os.path.join(thread_path, "replies")
                if not os.path.isdir(rep_dir):
                    continue
                for rep_file in sorted(os.listdir(rep_dir)):
                    with open(os.path.join(rep_dir, rep_file)) as f:
                        rep = json.load(f)
                    rep_id = str(rep.get("id", rep_file.replace(".json", "")))
                    records.append(dict(
                        reply_id=rep_id, thread_id=thread_id,
                        parent_id=str(rep.get("in_reply_to_status_id", "")),
                        depth=0, text=rep.get("text", ""),
                        label=label_map.get(rep_id), platform="twitter",
                        split=split_of(rep_id), is_source=False))
    return records


# ---------------------------------------------------------------------------
# condition construction
# ---------------------------------------------------------------------------

def build_conditions(target: dict, thread_records: List[dict],
                     id_to_record: Dict[str, dict]) -> Optional[dict]:
    src = next((r for r in thread_records if r["is_source"]), None)
    if not src or not target["label"]:
        return None

    src_text, tgt_text, tgt_label = src["text"], target["text"], target["label"]
    parent = id_to_record.get(target["parent_id"])

    reply_only = tgt_text

    if parent and not parent["is_source"]:
        useful = f"[Source] {src_text}\n[Context] {parent['text']}\n[Target] {tgt_text}"
    else:
        useful = f"[Source] {src_text}\n[Target] {tgt_text}"

    # irrelevant: on-topic stance-free comment with lowest lexical overlap
    ti_candidates = [
        r for r in thread_records
        if r["label"] == "comment" and r["reply_id"] not in (target["reply_id"], target["parent_id"])
        and not r["is_source"] and not has_stance_keyword(r["text"])
    ]
    if ti_candidates:
        ctx = min(ti_candidates, key=lambda r: jaccard_similarity(r["text"], tgt_text))
        irrelevant = f"[Source] {src_text}\n[Context] {ctx['text']}\n[Target] {tgt_text}"
        valid_ti = True
    else:
        irrelevant = f"[Source] {src_text}\n[Target] {tgt_text}"
        valid_ti = False

    # conflicting: a real opposing-stance reply (most stance keywords)
    cc_candidates = [
        r for r in thread_records
        if r["label"] in CONFLICT_PREF.get(tgt_label, []) and r["label"] != tgt_label
        and r["reply_id"] != target["reply_id"] and not r["is_source"]
        and len(r["text"].split()) >= 5
    ]
    if cc_candidates:
        cc_ctx = max(cc_candidates,
                     key=lambda r: sum(kw in r["text"].lower() for kw in STANCE_KEYWORDS))
        conflicting = f"[Source] {src_text}\n[Context] {cc_ctx['text']}\n[Target] {tgt_text}"
        valid_cc = True
    else:
        cc_ctx, conflicting, valid_cc = None, None, False

    if valid_cc:
        if parent and not parent["is_source"]:
            mixed = (f"[Source] {src_text}\n[Context] {parent['text']}\n"
                     f"[Misleading] {cc_ctx['text']}\n[Target] {tgt_text}")
        else:
            mixed = f"[Source] {src_text}\n[Misleading] {cc_ctx['text']}\n[Target] {tgt_text}"
    else:
        mixed = None

    lexical = f"[Context] {LD_TEMPLATES[tgt_label]}\n[Target] {tgt_text}"

    return dict(
        reply_id=target["reply_id"], thread_id=target["thread_id"], label=tgt_label,
        platform=target["platform"], split=target["split"],
        reply_only=reply_only, useful=useful, irrelevant=irrelevant,
        conflicting=conflicting, mixed=mixed, lexical=lexical,
        valid_ti=valid_ti, valid_cc=valid_cc)


def build_dataset(data_root: str) -> List[dict]:
    """Full pipeline: parse -> index threads -> build conditions for each reply."""
    records = parse_rumoureval(data_root)
    threads: Dict[str, list] = defaultdict(list)
    for r in records:
        threads[r["thread_id"]].append(r)
    id_to_record = {r["reply_id"]: r for r in records}

    dataset = []
    for target in records:
        if target["is_source"] or not target["label"]:
            continue
        out = build_conditions(target, threads[target["thread_id"]], id_to_record)
        if out:
            dataset.append(out)
    return dataset


def summarize(dataset: List[dict]) -> dict:
    """Split/label summary, handy for a data card."""
    summary = {}
    for split in ("train", "dev", "unknown"):
        sub = [d for d in dataset if d["split"] == split]
        if sub:
            summary[split] = {
                "n": len(sub),
                "labels": dict(Counter(d["label"] for d in sub)),
                "valid_cc_frac": round(sum(d["valid_cc"] for d in sub) / len(sub), 3),
                "valid_ti_frac": round(sum(d["valid_ti"] for d in sub) / len(sub), 3),
            }
    return summary
