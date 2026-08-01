"""Unit tests for the train/validation split (run: python -m pytest -q).

The key property under test is **no thread-level leakage**: because the context
conditions splice in text from other replies in the same thread, no thread may
appear in both the fit and validation folds.
"""

from llm_robustness import data


def _toy_dataset():
    # 10 train threads x 3 replies each, plus a couple of dev rows.
    rows = []
    for t in range(10):
        for r in range(3):
            rows.append({"reply_id": f"t{t}_r{r}", "thread_id": f"t{t}",
                         "label": ["support", "deny", "query", "comment"][r % 4],
                         "split": "train"})
    for d in range(4):
        rows.append({"reply_id": f"dev_{d}", "thread_id": f"dev_t{d}",
                     "label": "comment", "split": "dev"})
    return rows


def test_split_is_deterministic():
    ds = _toy_dataset()
    fit1, val1 = data.train_val_split(ds, val_fraction=0.2, seed=42)
    fit2, val2 = data.train_val_split(ds, val_fraction=0.2, seed=42)
    assert [r["reply_id"] for r in fit1] == [r["reply_id"] for r in fit2]
    assert [r["reply_id"] for r in val1] == [r["reply_id"] for r in val2]


def test_no_thread_leakage_between_fit_and_val():
    ds = _toy_dataset()
    fit, val = data.train_val_split(ds, val_fraction=0.2, seed=42)
    fit_threads = {r["thread_id"] for r in fit}
    val_threads = {r["thread_id"] for r in val}
    assert fit_threads.isdisjoint(val_threads)


def test_no_reply_overlap_and_covers_all_train():
    ds = _toy_dataset()
    fit, val = data.train_val_split(ds, val_fraction=0.2, seed=42)
    fit_ids = {r["reply_id"] for r in fit}
    val_ids = {r["reply_id"] for r in val}
    assert fit_ids.isdisjoint(val_ids)
    assert len(fit_ids) + len(val_ids) == 30  # all train replies, none dropped


def test_val_fraction_is_thread_level():
    ds = _toy_dataset()
    _, val = data.train_val_split(ds, val_fraction=0.2, seed=42)
    # 20% of 10 threads == 2 threads -> 6 replies.
    assert len({r["thread_id"] for r in val}) == 2
    assert len(val) == 6


def test_dev_split_is_untouched():
    ds = _toy_dataset()
    fit, val = data.train_val_split(ds, val_fraction=0.2, seed=42)
    all_ids = {r["reply_id"] for r in fit} | {r["reply_id"] for r in val}
    assert not any(rid.startswith("dev_") for rid in all_ids)


def test_manifest_folds_sum_to_dataset():
    ds = _toy_dataset()
    man = data.split_manifest(ds, val_fraction=0.2, seed=42)
    assert man["fit"]["n"] + man["val"]["n"] == 30
    assert man["test"]["n"] == 4
    assert man["val"]["n_threads"] == 2
