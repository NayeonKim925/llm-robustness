#!/usr/bin/env python3
"""Paired McNemar tests comparing context conditions (CPU).

Thin wrapper over ``llm_robustness.cli.significance_main`` (also installed as
``llmr-significance``). Requires ``pip install -e .`` first.

Pairs come from the confidence dumps, which store every condition's prediction
for the same ``reply_id`` on each row. Those dumps are from the original
(flawed) confidence pipeline, so the *substantive* p-values describe those
predictions; the reusable, tested machinery is ``metrics.mcnemar_test``. Re-run
on corrected ``evaluate.py`` outputs for headline claims.
"""

from llm_robustness.cli import significance_main

if __name__ == "__main__":
    significance_main()
