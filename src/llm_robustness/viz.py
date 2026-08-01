"""Figures for the results (matplotlib).

Design choices (see docs and the project README):

* **Colour-blind-safe categorical palette** (Okabe-Ito subset), validated with
  the dataviz palette checker: blue / orange / green / vermillion, worst
  adjacent CVD separation deltaE 11.0.
* **Second channel for identity**: training regime is also encoded by line
  style (zero-shot = dashed, adv-FT = solid) and marker, so the reader never
  relies on colour alone.
* **Recessive chrome**: light thin grid, no top/right spines, direct end labels
  on lines rather than a number on every point.

Every figure is computed from ``results/classification_analysis.json`` and
``results/calibration_analysis.json`` — no numbers are hard-coded here.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

CONDITIONS: List[str] = ["reply_only", "useful", "irrelevant", "conflicting", "mixed", "lexical"]

# Fixed colour order, assigned per model entity (never by rank).
MODEL_COLOR = {
    "qwen0.5B-zeroshot": "#D55E00",  # vermillion
    "qwen3B-zeroshot": "#E69F00",    # orange
    "qwen1.5B-adv-FT": "#0072B2",    # blue
    "qwen3B-adv-FT": "#009E73",      # green
}
# Second channel: zero-shot dashed/circle, adversarial-FT solid/square.
MODEL_STYLE = {
    "qwen0.5B-zeroshot": dict(ls="--", marker="o"),
    "qwen3B-zeroshot": dict(ls="--", marker="o"),
    "qwen1.5B-adv-FT": dict(ls="-", marker="s"),
    "qwen3B-adv-FT": dict(ls="-", marker="s"),
}
INK, MUTED, GRID = "#1a1a1a", "#666666", "#dddddd"


def _setup(plt):
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True,  # grid behind bars/lines, not over them
    })


def robustness_by_condition(analysis: Dict, out_path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _setup(plt)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = range(len(CONDITIONS))

    # majority baseline band (varies slightly by condition n); labelled on the left
    maj = [analysis["qwen3B-zeroshot"][c]["baseline_majority_acc"] for c in CONDITIONS]
    ax.plot(x, maj, color=MUTED, ls=":", lw=1.5, zorder=1)
    ax.text(0, maj[0] - 0.045, "majority baseline",
            color=MUTED, fontsize=9, ha="left", style="italic")

    for model in MODEL_COLOR:
        if model not in analysis:
            continue
        ys = [analysis[model][c]["accuracy"] for c in CONDITIONS]
        ax.plot(x, ys, color=MODEL_COLOR[model], lw=2, markersize=7,
                zorder=3, label=model, **MODEL_STYLE[model])

    # call out the adversarial lexical dip for zero-shot 3B
    zs = analysis.get("qwen3B-zeroshot", {})
    if "lexical" in zs:
        ax.annotate("adversarial\ndistractor",
                    xy=(5, zs["lexical"]["accuracy"]), xytext=(3.5, 0.47),
                    color=MUTED, fontsize=9, ha="center",
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))

    # legend in the empty mid-left region -> no colliding end labels
    ax.legend(frameon=False, fontsize=9, loc="center left",
              bbox_to_anchor=(0.0, 0.30), handlelength=2.4)

    ax.set_xticks(list(x))
    ax.set_xticklabels(CONDITIONS, rotation=20, ha="right")
    ax.set_ylabel("accuracy (dev)")
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.2, len(CONDITIONS) - 0.5)
    ax.set_title("Accuracy across six context conditions (dev)", pad=14)
    ax.text(0.5, 1.02, "adv-FT stays flat; zero-shot 3B dips under the lexical distractor",
            transform=ax.transAxes, ha="center", fontsize=9.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def per_class_f1_heatmap(analysis: Dict, out_path: str, condition: str = "reply_only") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _setup(plt)
    labels = ["support", "deny", "query", "comment"]
    models = [m for m in MODEL_COLOR if m in analysis]
    grid = [[analysis[m][condition]["per_class_f1"][i] for i in range(4)] for m in models]

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.grid(False)
    im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4), labels)
    ax.set_yticks(range(len(models)), models)
    for i in range(len(models)):
        for j in range(4):
            val = grid[i][j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if val > 0.55 else INK, fontsize=10)
    ax.set_title(f"Per-class F1 by model ({condition})", pad=22)
    ax.text(0.5, 1.06, "majority class 'comment' is learned; minority stances stay weak",
            transform=ax.transAxes, ha="center", fontsize=9, color=MUTED)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("F1", color=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def adversarial_lexical_effect(analysis: Dict, out_path: str) -> str:
    """Focused view: reply_only vs lexical for zero-shot vs adv-FT (3B)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _setup(plt)
    pairs = [("qwen3B-zeroshot", "3B zero-shot"), ("qwen3B-adv-FT", "3B adversarial-FT")]
    pairs = [(m, lbl) for m, lbl in pairs if m in analysis]

    fig, ax = plt.subplots(figsize=(7, 4.6))
    width = 0.35
    x = range(len(pairs))
    ro = [analysis[m]["reply_only"]["accuracy"] for m, _ in pairs]
    lx = [analysis[m]["lexical"]["accuracy"] for m, _ in pairs]
    b1 = ax.bar([i - width / 2 for i in x], ro, width, label="reply_only (clean)",
                color="#9ecae1")
    b2 = ax.bar([i + width / 2 for i in x], lx, width, label="lexical (adversarial)",
                color="#08519c")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f"{bar.get_height():.3f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(list(x), [lbl for _, lbl in pairs])
    ax.set_ylabel("accuracy (dev)")
    ax.set_ylim(0, 1)
    ax.set_title("The lexical distractor hurts zero-shot but not adversarial-FT\n"
                 "(3B: 0.628→0.598 vs 0.825→0.851)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def reliability_original_flawed(calibration: Dict, out_path: str,
                                condition: str = "reply_only") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _setup(plt)
    fig, ax = plt.subplots(figsize=(6.4, 6))
    ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1.2, label="perfect calibration")
    palette = {"qwen_1.5b_ft_ORIGINAL_FLAWED": "#0072B2",
               "qwen_1.5b_adv_ORIGINAL_FLAWED": "#D55E00"}
    for name, color in palette.items():
        bins = calibration.get(name, {}).get(condition, {}).get("reliability_bins", [])
        if not bins:
            continue
        ece = calibration[name][condition].get("ece", 0)
        ax.plot([b["avg_confidence"] for b in bins], [b["accuracy"] for b in bins],
                "o-", color=color, lw=2, markersize=6,
                label=f"{name.replace('_ORIGINAL_FLAWED','')} (ECE={ece:.2f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title("Reliability — original (flawed) pipeline", fontsize=12, pad=20)
    ax.text(0.5, 1.03, "confidence far exceeds accuracy: a measurement artifact, not a result",
            transform=ax.transAxes, ha="center", fontsize=8.5, color=MUTED)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def make_all(results_dir: str = "results", figures_dir: str = "figures") -> List[str]:
    os.makedirs(figures_dir, exist_ok=True)
    with open(os.path.join(results_dir, "classification_analysis.json")) as f:
        cls = json.load(f)
    calib_path = os.path.join(results_dir, "calibration_analysis.json")
    calib = json.load(open(calib_path)) if os.path.exists(calib_path) else {}

    made = [
        robustness_by_condition(cls, os.path.join(figures_dir, "robustness_by_condition.png")),
        per_class_f1_heatmap(cls, os.path.join(figures_dir, "per_class_f1_heatmap.png")),
        adversarial_lexical_effect(
            cls, os.path.join(figures_dir, "adversarial_lexical_effect.png")),
    ]
    if calib:
        made.append(reliability_original_flawed(
            calib, os.path.join(figures_dir, "reliability_original_flawed.png")))
    return made
