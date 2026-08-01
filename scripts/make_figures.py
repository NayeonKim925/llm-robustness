#!/usr/bin/env python3
"""Render all result figures from the distilled analysis JSON (CPU).

Reads ``results/classification_analysis.json`` and
``results/calibration_analysis.json`` (produced by ``analyze_results.py``) and
writes PNGs to ``figures/``. Separated from analysis so plotting can be
re-run/iterated without recomputing metrics.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llm_robustness import viz  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--figures-dir", default="figures")
    args = ap.parse_args()
    for p in viz.make_all(args.results_dir, args.figures_dir):
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
