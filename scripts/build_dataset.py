#!/usr/bin/env python3
"""Build ``context_conditions.json`` from the raw RumourEval-2019 trees (CPU).

Thin wrapper over ``llm_robustness.cli.build_dataset_main`` (also installed as
``llmr-dataset``). Requires ``pip install -e .`` first.
"""

from llm_robustness.cli import build_dataset_main

if __name__ == "__main__":
    build_dataset_main()
