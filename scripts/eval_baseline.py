"""Evaluate a baseline (pretrained, non-P2) model and compute the same
per-size-bin metrics as training_monitor, so Phase 1 P2 results have
something to beat.

Typical use on pod:
    python scripts/eval_baseline.py --model yolo26n.pt --name baseline_yolo26n

Output: runs/baselines/{name}/baseline_metrics.csv
        (1 row per size bin; format identical to training_monitor.csv minus epoch)

Runtime: ~2-3 min on a single GPU (one val pass on COCO val2017).
"""
import argparse
import csv
import json
import os
from pathlib import Path

from ultralytics import YOLO

from training_monitor import (
    DEFAULT_SIZE_BINS,
    compute_bottleneck_metrics,
    _load_gt_by_image,
)

PROJECT = Path(__file__).resolve().parent.parent
COCO_ROOT = os.environ.get("COCO_ROOT", "/workspace/coco")
DEFAULT_GT = f"{COCO_ROOT}/annotations/instances_val2017.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Weights (e.g. yolo26n.pt)")
    ap.add_argument("--name", required=True, help="Run name for output dir")
    ap.add_argument("--data", default="coco.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--gt", default=DEFAULT_GT)
    ap.add_argument("--project-dir", default=str(PROJECT / "runs" / "baselines"))
    args = ap.parse_args()

    out_dir = Path(args.project_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[baseline] model={args.model}  imgsz={args.imgsz}")
    model = YOLO(args.model)
    # end2end=False matches Phase 0 baseline methodology (NMS, not one-stage decode)
    results = model.val(
        data=args.data,
        imgsz=args.imgsz,
        save_json=True,
        end2end=False,
        project=str(out_dir.parent),
        name=args.name,
        exist_ok=True,
    )

    # predictions.json written by ultralytics when save_json=True
    pred_path = Path(results.save_dir) / "predictions.json"
    if not pred_path.exists():
        raise FileNotFoundError(f"Expected predictions at {pred_path}")

    print(f"[baseline] loading predictions from {pred_path}")
    with open(pred_path) as f:
        preds = json.load(f)

    print(f"[baseline] loading GT from {args.gt}")
    gt_by_image = _load_gt_by_image(args.gt)

    metrics = compute_bottleneck_metrics(preds, gt_by_image, DEFAULT_SIZE_BINS)

    csv_path = out_dir / "baseline_metrics.csv"
    fields = ["size_bin", "recall_at_05", "median_iou", "fp_above_tp_median",
              "median_tp_conf", "gt_count", "tp_count", "fp_count"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for label, vals in metrics.items():
            writer.writerow({"size_bin": label, **vals})

    print(f"\n[baseline] saved: {csv_path}\n")
    # Pretty-print summary
    print(f"{'bin':<10}{'R@0.5':>8}{'medIoU':>8}{'FP>TPmed':>10}{'GT':>7}{'TP':>7}{'FP':>9}")
    for label, m in metrics.items():
        print(f"{label:<10}{m['recall_at_05']:>8.3f}{m['median_iou']:>8.3f}"
              f"{m['fp_above_tp_median']:>10.3f}{m['gt_count']:>7}"
              f"{m['tp_count']:>7}{m['fp_count']:>9}")


if __name__ == "__main__":
    main()
