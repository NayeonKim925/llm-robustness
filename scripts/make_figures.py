#!/usr/bin/env python3
"""Render all result figures from the distilled analysis JSON (CPU).

Thin wrapper over ``llm_robustness.cli.figures_main`` (also installed as
``llmr-figures``). Requires ``pip install -e .`` first.
"""

from llm_robustness.cli import figures_main

if __name__ == "__main__":
    figures_main()
