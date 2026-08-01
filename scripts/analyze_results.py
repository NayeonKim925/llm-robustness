#!/usr/bin/env python3
"""Recompute metrics + baselines from the prediction dumps, and render figures.

Thin wrapper over ``llm_robustness.cli.analyze_main`` (also installed as the
``llmr-analyze`` console command). Requires ``pip install -e .`` first.
See ``llm_robustness.report`` for the logic.
"""

from llm_robustness.cli import analyze_main

if __name__ == "__main__":
    analyze_main()
