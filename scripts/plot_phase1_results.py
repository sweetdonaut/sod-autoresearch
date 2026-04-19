"""Visualize Phase 1 P2 vs baseline_yolo26n results.

Outputs two PNGs into the run dir:
  - phase1_summary.png         per-bin baseline vs P2 final (the trade-off)
  - phase1_training_curves.png 3 bottleneck metrics over 30 epochs, lines by bin
"""
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

RUN_DIR  = Path("/workspace/sod-autoresearch/runs/p2/p1_exp1_sat30")
BASE_CSV = Path("/workspace/sod-autoresearch/runs/baselines/baseline_yolo26n/baseline_metrics.csv")

BINS = ["0-8", "8-16", "16-32", "32-64", "64-128", "128-9999"]
BIN_LABELS = ["0-8", "8-16", "16-32", "32-64", "64-128", "128+"]


def load_baseline():
    with open(BASE_CSV) as f:
        return {r["size_bin"]: r for r in csv.DictReader(f)}


def load_monitor():
    with open(RUN_DIR / "training_monitor.csv") as f:
        rows = list(csv.DictReader(f))
    by_epoch = {}
    for r in rows:
        e = int(r["epoch"])
        by_epoch.setdefault(e, {})[r["size_bin"]] = r
    return by_epoch


def fig_summary(base, by_epoch):
    last_e = max(by_epoch)
    final = by_epoch[last_e]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    metrics = [
        ("recall_at_05", "recall@0.5", "↑ better"),
        ("median_iou", "median IoU (TPs)", "↑ better"),
        ("fp_above_tp_median", "FP above TP-median conf", "↓ better"),
    ]
    x = np.arange(len(BINS))
    w = 0.38
    for ax, (m, title, hint) in zip(axes, metrics):
        b_vals = [float(base[b][m]) for b in BINS]
        p_vals = [float(final[b][m]) for b in BINS]
        ax.bar(x - w/2, b_vals, w, label="baseline yolo26n", color="#888")
        ax.bar(x + w/2, p_vals, w, label="P2 ep30", color="#d62728")
        # annotate delta on P2 bars
        for i, (bv, pv) in enumerate(zip(b_vals, p_vals)):
            d = pv - bv
            color = "darkgreen" if (d > 0 if "recall" in m or "iou" in m else d < 0) else "darkred"
            ax.text(x[i] + w/2, pv + 0.01, f"{d:+.3f}",
                    ha="center", va="bottom", fontsize=9, color=color, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(BIN_LABELS)
        ax.set_xlabel("GT short-side (px)")
        ax.set_title(f"{title}  ({hint})")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(max(b_vals), max(p_vals)) * 1.18)
        if m == "recall_at_05":
            ax.legend(loc="lower right")

    fig.suptitle(f"Phase 1: P2 head trade-off — small-bin gain vs large-bin loss (ep {last_e+1})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = RUN_DIR / "phase1_summary.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved: {out}")
    plt.close(fig)


def fig_training_curves(base, by_epoch):
    epochs = sorted(by_epoch.keys())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [
        ("recall_at_05", "recall@0.5"),
        ("median_iou", "median IoU"),
        ("fp_above_tp_median", "FP above TP-median conf"),
    ]
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / (len(BINS) - 1)) for i in range(len(BINS))]

    for ax, (m, title) in zip(axes, metrics):
        for bin_label, label, color in zip(BINS, BIN_LABELS, colors):
            ys = [float(by_epoch[e][bin_label][m]) for e in epochs]
            ax.plot(epochs, ys, marker="o", ms=3, lw=1.4,
                    color=color, label=label)
            base_v = float(base[bin_label][m])
            ax.axhline(base_v, color=color, ls="--", lw=0.8, alpha=0.5)
        # annotate phases
        ax.axvspan(0, 2.5, alpha=0.08, color="orange", label="warmup" if m == "recall_at_05" else None)
        ax.axvspan(19.5, max(epochs) + 0.5, alpha=0.08, color="green",
                   label="mosaic off" if m == "recall_at_05" else None)
        ax.set_xlabel("epoch")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        if m == "recall_at_05":
            ax.legend(loc="lower right", fontsize=8, ncol=2,
                      title="size bin (px)\n— = baseline")

    fig.suptitle("Phase 1 P2 training curves — solid = P2 by epoch, dashed = baseline yolo26n",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = RUN_DIR / "phase1_training_curves.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved: {out}")
    plt.close(fig)


def main():
    base = load_baseline()
    by_epoch = load_monitor()
    fig_summary(base, by_epoch)
    fig_training_curves(base, by_epoch)


if __name__ == "__main__":
    main()
