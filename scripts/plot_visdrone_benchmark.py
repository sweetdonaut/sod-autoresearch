"""Benchmark figure: YOLO26 {n..x} × {no-P2, +P2} vs Dome-DETR-L on
VisDrone-DET val. Styled after typical paper-figure conventions.

X-axis: GFLOPs @ imgsz=1280 (deployment resolution for our models).
Y-axis: mAP @[0.5:0.95], primary detection metric.

Two aggregations are supported:
  --metric ultralytics  -> uses peak per-run metrics/mAP50-95(B)
                           from results.csv (ultralytics 10-class eval)
  --metric dome_detr    -> uses AP_all@500 from dome_detr_stats.json
                           (12-class Dome-DETR protocol — only comparable
                           view to Dome-DETR's paper number)

Missing data points are silently skipped so the plot can be regenerated
anytime during the sweep.

Output: docs/figures/visdrone_benchmark_{metric}.{png,pdf}
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT = Path(__file__).resolve().parent.parent

# GFLOPs @ imgsz=1280, taken directly from each run's AutoBatch probe
# (ultralytics' thop measurement on a (1, 3, 1280, 1280) input). These
# reflect the *actual* compute after nc=10 Detect-head adjustment, so
# they're more accurate than scaling the yaml's nc=80 numbers.
# x / x+P2 from the original Phase 2 P2-ablation run logs.
GFLOPS_1280 = {
    ("n", False):   23.2,  ("n", True):    30.2,
    ("s", False):   90.2,  ("s", True):   105.8,
    ("m", False):  299.2,  ("m", True):   365.6,   # m+P2 not run, est 1.22x
    ("l", False):  372.8,  ("l", True):   461.2,   # l+P2 not run, est 1.24x
    ("x", False):  838.0,  ("x", True):  1020.0,
}

# Round 1 variants (50 ep, original sweep). x/x+P2 come from the
# pre-sweep training runs.
VARIANTS = [
    ("n", False), ("n", True),
    ("s", False), ("s", True),
    ("m", False),
    ("l", False),
    ("x", False), ("x", True),
]

# Round 2: extended-budget P2 variants (100 ep). Plotted with a
# distinct marker so the budget-fairness caveat is visible at a glance.
VARIANTS_100EP = [
    ("n", True),
    ("s", True),
]

# Reference methods — full Dome-DETR family from arXiv:2505.05741 Table 2
# @ 800x800 input (their deployment resolution). GFLOPs are density-
# adaptive; reported values are averages across VisDrone val.
REFERENCE_METHODS = [
    # (name, gflops, mAP_5095, mAP_50, marker, color)
    ("Dome-DETR-S",  176.5, 0.335, 0.566, "*", "#d62728"),
    ("Dome-DETR-M",  284.6, 0.361, 0.598, "*", "#d62728"),
    ("Dome-DETR-L",  376.4, 0.390, 0.611, "*", "#d62728"),
]
REFERENCE_LINE_LABEL = "Dome-DETR (ACM MM 2025)"


def _sweep_run_dir(scale, p2, epochs=50):
    """Where to find the trained run for a given variant.
    x and x+P2 (50 ep) live under runs/p2/ (pre-sweep naming);
    the 6 sweep variants (n, n+p2, s, s+p2, m, l) live under runs/sweep/.
    Round 2 (100 ep) variants also live under runs/sweep/ with the
    matching epoch suffix."""
    if scale == "x" and epochs == 50:
        # Pre-sweep runs used a different naming convention
        name = "visdrone_p2_x_1280_50ep" if p2 else "visdrone_nop2_x_1280_50ep"
        return PROJECT / "runs" / "p2" / name
    tag = f"{scale}_p2" if p2 else scale
    return PROJECT / "runs" / "sweep" / f"visdrone_{tag}_1280_{epochs}ep"


def _dome_eval_dir(scale, p2, epochs=50):
    """Dome-DETR eval output dir."""
    tag = f"{scale}_p2" if p2 else scale
    if scale == "x" and epochs == 50:
        # existing two runs were named differently
        name = "p2_at_1280_letterbox" if p2 else "nop2_at_1280_letterbox"
    elif epochs == 50:
        name = f"{tag}_at_1280_letterbox"
    else:
        # round 2 / extended-budget eval naming
        name = f"{tag}_{epochs}ep_at_1280_letterbox"
    return PROJECT / "runs" / "dome_detr_eval" / name


def load_ultralytics_peak(run_dir: Path):
    """Peak mAP50-95(B) from results.csv. Returns None if run incomplete."""
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return None
    peak_5095 = 0.0
    peak_50 = 0.0
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        m5095_col = next((k for k in reader.fieldnames
                          if "mAP50-95(B)" in k), None)
        m50_col = next((k for k in reader.fieldnames
                        if "mAP50(B)" in k and "95" not in k), None)
        if m5095_col is None:
            return None
        for row in reader:
            try:
                v = float(row[m5095_col])
                if v > peak_5095:
                    peak_5095 = v
                    peak_50 = float(row[m50_col]) if m50_col else 0.0
            except (ValueError, KeyError):
                continue
    if peak_5095 == 0:
        return None
    return {"mAP50-95": peak_5095, "mAP50": peak_50}


def load_dome_detr(eval_dir: Path):
    json_path = eval_dir / "dome_detr_stats.json"
    if not json_path.exists():
        return None
    with open(json_path) as f:
        d = json.load(f)
    return {"mAP50-95": d["AP_all@500"], "mAP50": d["AP50@500"]}


def collect(metric: str, epochs: int = 50):
    """Return list of {scale, p2, gflops, mAP_5095, mAP_50, epochs} for
    each variant with trained data available."""
    rows = []
    variants = VARIANTS if epochs == 50 else VARIANTS_100EP
    for scale, p2 in variants:
        if metric == "ultralytics":
            d = load_ultralytics_peak(_sweep_run_dir(scale, p2, epochs))
        elif metric == "dome_detr":
            d = load_dome_detr(_dome_eval_dir(scale, p2, epochs))
        else:
            raise ValueError(metric)
        if d is None:
            continue
        rows.append({
            "scale": scale, "p2": p2, "epochs": epochs,
            "gflops": GFLOPS_1280[(scale, p2)],
            "mAP_5095": d["mAP50-95"],
            "mAP_50": d["mAP50"],
        })
    return rows


def plot(metric: str, ap_variant: str, out_path: Path):
    """ap_variant: '50-95' (primary) or '50' (IoU=0.5 only)."""
    assert ap_variant in ("50-95", "50")
    key = "mAP_5095" if ap_variant == "50-95" else "mAP_50"
    ref_idx = 2 if ap_variant == "50-95" else 3   # REFERENCE_METHODS tuple idx

    rows = collect(metric, epochs=50)
    plain = sorted([r for r in rows if not r["p2"]], key=lambda r: r["gflops"])
    p2    = sorted([r for r in rows if r["p2"]],     key=lambda r: r["gflops"])
    rows_100 = collect(metric, epochs=100)

    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    if plain:
        ax.plot([r["gflops"] for r in plain],
                [r[key] for r in plain],
                "o-", color="#1f77b4", linewidth=2.0, markersize=9,
                label="YOLO26 (ours, 50 ep)")
        for r in plain:
            ax.annotate(
                r["scale"],
                xy=(r["gflops"], r[key]),
                xytext=(7, -10), textcoords="offset points",
                fontsize=11, color="#1f77b4", fontweight="bold",
            )

    if p2:
        ax.plot([r["gflops"] for r in p2],
                [r[key] for r in p2],
                "s--", color="#ff7f0e", linewidth=2.0, markersize=9,
                label="YOLO26 + P2 (ours, 50 ep)")
        for r in p2:
            # 50ep +P2 sits below the 100ep variant for the same scale
            # (less budget = lower AP) — anchor label BELOW point.
            ax.annotate(
                f"{r['scale']}+P2",
                xy=(r["gflops"], r[key]),
                xytext=(7, -14), textcoords="offset points",
                fontsize=10, color="#ff7f0e", fontweight="bold",
            )

    if rows_100:
        rows_100_sorted = sorted(rows_100, key=lambda r: r["gflops"])
        ax.plot([r["gflops"] for r in rows_100_sorted],
                [r[key] for r in rows_100_sorted],
                "^-.", color="#2ca02c", linewidth=2.0, markersize=10,
                label="YOLO26 + P2 (ours, 100 ep)")
        for r in rows_100_sorted:
            # Per-variant placement: right-down works for n+P2 (clear of
            # neighbours), but for s+P2 the right side is occupied by
            # Dome-DETR-S's annotation, so anchor s+P2's label centred
            # directly above its triangle instead.
            if r["scale"] == "s":
                xytext, ha = (0, 12), "center"
            else:
                xytext, ha = (12, -10), "left"
            ax.annotate(
                f"{r['scale']}+P2 (100ep)",
                xy=(r["gflops"], r[key]),
                xytext=xytext, textcoords="offset points",
                fontsize=9, color="#2ca02c", fontweight="bold", ha=ha,
            )

    if REFERENCE_METHODS:
        ref_sorted = sorted(REFERENCE_METHODS, key=lambda x: x[1])
        ax.plot([m[1] for m in ref_sorted],
                [m[ref_idx] for m in ref_sorted],
                linestyle=":", color="#d62728", alpha=0.55,
                linewidth=1.5, zorder=5)
        first_label_used = False
        for entry in ref_sorted:
            name, gf, marker, color = entry[0], entry[1], entry[4], entry[5]
            ap_val = entry[ref_idx]
            label = REFERENCE_LINE_LABEL if not first_label_used else None
            ax.scatter([gf], [ap_val], marker=marker, s=260, color=color,
                       edgecolor="black", linewidth=0.8, zorder=10,
                       label=label)
            first_label_used = True
            ax.annotate(name, xy=(gf, ap_val),
                        xytext=(8, -4), textcoords="offset points",
                        fontsize=9, color=color, ha="left", va="center")

    ax.set_xlabel("GFLOPs (inference @ imgsz=1280 for ours, 800 for Dome-DETR)",
                  fontsize=11)
    ap_pretty = "0.5:0.95" if ap_variant == "50-95" else "0.5"
    proto = ("Dome-DETR protocol, 12 cls, maxDet=500"
             if metric == "dome_detr"
             else "ultralytics protocol, 10 cls")
    ax.set_ylabel(f"mAP @ {ap_pretty}  ({proto})", fontsize=11)
    ax.set_title("YOLO26 × P2 on VisDrone-DET val (A100, imgsz=1280, 50 ep)",
                 fontsize=12)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)

    ax.set_xlim(left=0)
    y_values = ([r[key] for r in rows + rows_100]
                + [m[ref_idx] for m in REFERENCE_METHODS])
    if y_values:
        y_min, y_max = min(y_values), max(y_values)
        pad = (y_max - y_min) * 0.12 or 0.02
        ax.set_ylim(y_min - pad, y_max + pad)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[plot] saved: {out_path.with_suffix('.png')}")
    print(f"[plot] saved: {out_path.with_suffix('.pdf')}")
    print(f"[plot] {len(rows)} 50-ep points "
          f"({len(plain)} plain, {len(p2)} +P2) + {len(rows_100)} 100-ep "
          f"points  metric=mAP@{ap_pretty}")
    for r in rows:
        tag = f"{r['scale']}{'+P2' if r['p2'] else ''}"
        print(f"  {tag:<6s} 50ep   GFLOPs={r['gflops']:>6.1f}  "
              f"mAP50={r['mAP_50']:.4f}  mAP50-95={r['mAP_5095']:.4f}")
    for r in rows_100:
        tag = f"{r['scale']}{'+P2' if r['p2'] else ''}"
        print(f"  {tag:<6s} 100ep  GFLOPs={r['gflops']:>6.1f}  "
              f"mAP50={r['mAP_50']:.4f}  mAP50-95={r['mAP_5095']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="dome_detr",
                    choices=["ultralytics", "dome_detr"])
    ap.add_argument("--ap", default="50-95", choices=["50-95", "50"],
                    help="Which AP metric to plot as Y-axis")
    ap.add_argument("--all", action="store_true",
                    help="Generate all 4 figure combinations "
                         "(metric x ap) in one pass")
    ap.add_argument("--out", default=None,
                    help="output path stem (no extension); defaults to "
                         "docs/figures/visdrone_benchmark_{metric}_ap{ap}")
    args = ap.parse_args()

    figures_dir = PROJECT / "docs" / "figures"
    if args.all:
        # Default: only Dome-DETR protocol (direct comparison to the paper's
        # 39.0% number), both AP variants. Use explicit --metric ultralytics
        # --ap {50,50-95} if the 10-class view is also wanted.
        for ap_variant in ("50-95", "50"):
            out = figures_dir / f"visdrone_benchmark_dome_detr_ap{ap_variant}"
            plot("dome_detr", ap_variant, out)
            print()
    else:
        out = (Path(args.out) if args.out
               else figures_dir /
                    f"visdrone_benchmark_{args.metric}_ap{args.ap}")
        plot(args.metric, args.ap, out)


if __name__ == "__main__":
    main()
