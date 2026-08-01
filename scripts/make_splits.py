#!/usr/bin/env python3
"""Emit the deterministic train/val/test split manifest (CPU).

Thin wrapper over ``llm_robustness.cli.splits_main`` (also installed as
``llmr-splits``). The `test` fold is the RumourEval dev split (held out); `fit`
and `val` are a thread-level partition of `train` (see ``data.train_val_split``
for the leakage rationale).
"""

from llm_robustness.cli import splits_main

if __name__ == "__main__":
    splits_main()
