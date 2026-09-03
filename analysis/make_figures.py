#!/usr/bin/env python3
"""Figure: individual-presentation self-recognition yes-rates.

For each evaluator and dataset, plots P(yes | own code) next to P(yes | other model's code).
Genuine self-recognition would show the first bar well above the second; response bias
shows both bars at the same height.

Usage: .venv/bin/python analysis/make_figures.py [--out DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = Path("/Users/ehsan/LatexProjects/llm-collusion-paper/latex/figures")
TABLES = Path("/Users/ehsan/LatexProjects/llm-collusion-paper/latex/tables")

MODELS = [
    ("openai/gpt-5", "GPT-5"),
    ("anthropic/claude-haiku-4.5", "Claude"),
    ("deepseek/deepseek-chat-v3-0324", "DeepSeek"),
    ("google/gemini-2.5-flash", "Gemini"),
    ("x-ai/grok-4-fast", "Grok"),
]
DATASETS = [("humaneval", "HumanEval"), ("mbpp-sanitized", "MBPP"), ("ds1000", "DS-1000")]
C_OWN, C_OTHER = "#2a78d6", "#eb6834"  # categorical slots 1 and 2 (validated palette)
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#d9d8d0"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ipp = json.load((TABLES / "summary.json").open())["ipp"]

    plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.2), sharey=True)
    w = 0.36
    for ax, (ds, dsname) in zip(axes, DATASETS):
        xs = range(len(MODELS))
        own = [100 * ipp[f"{ds}|{m}"]["tpr"] for m, _ in MODELS]
        oth = [100 * ipp[f"{ds}|{m}"]["fpr"] for m, _ in MODELS]
        ax.bar([x - w / 2 for x in xs], own, width=w - 0.03, color=C_OWN, label="own code", zorder=3)
        ax.bar([x + w / 2 for x in xs], oth, width=w - 0.03, color=C_OTHER, label="other model's code", zorder=3)
        for x, (a, b) in enumerate(zip(own, oth)):
            ax.text(x - w / 2, a + 2, f"{a:.0f}", ha="center", va="bottom", fontsize=6.5, color=INK)
            ax.text(x + w / 2, b + 2, f"{b:.0f}", ha="center", va="bottom", fontsize=6.5, color=INK)
        ax.set_title(dsname, fontsize=8.5, color=INK, pad=4)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([s for _, s in MODELS], fontsize=6.5)
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="both", length=0)
    axes[0].set_ylabel('P("yes") in %', color=INK)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=7, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06), handlelength=1.0)
    fig.tight_layout(w_pad=0.8)
    for ext in ("pdf", "png"):
        fig.savefig(args.out / f"ipp_yes_rates.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {args.out / 'ipp_yes_rates.pdf'}")


if __name__ == "__main__":
    main()
