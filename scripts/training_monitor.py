"""Training-time bottleneck monitor for YOLO detectors.

After each epoch's val, computes 3 per-size-bin direct metrics that track the
three bottlenecks identified in Phase 0.5d:

    - recall_at_05       : tracks Recall bottleneck
    - median_iou_matched : tracks Regression bottleneck
    - fp_above_tp_median : tracks FP/Calibration bottleneck

Usage (attach to an ultralytics YOLO model before calling train):

    from scripts.training_monitor import register_training_monitor
    register_training_monitor(
        model,
        gt_path="/path/to/instances_val2017.json",
        output_csv=Path("runs/train1/training_monitor.csv"),
    )
    model.train(...)

Each epoch appends one row per size bin to the CSV.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

DEFAULT_SIZE_BINS = [
    (0, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 9999),
]


def _iou_xywh(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[0] + b1[2], b2[0] + b2[2])
    y2 = min(b1[1] + b1[3], b2[1] + b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = b1[2] * b1[3] + b2[2] * b2[3] - inter
    return inter / union if union > 0 else 0.0


def _bin_idx(short_side, bins):
    for i, (lo, hi) in enumerate(bins):
        if lo <= short_side < hi:
            return i
    return len(bins) - 1


def _load_gt_by_image(gt_path):
    import json
    with open(gt_path) as f:
        coco = json.load(f)
    gt_by_image = {}
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        gt_by_image.setdefault(ann["image_id"], []).append(ann)
    return gt_by_image


def compute_bottleneck_metrics(jdict, gt_by_image, size_bins=None, match_iou=0.5):
    """Compute 3 per-size-bin metrics from COCO-format predictions.

    Args:
        jdict:         list of {image_id, category_id, bbox, score}
        gt_by_image:   dict img_id -> list of GT anns
        size_bins:     list of (lo, hi) in pixels, binning by short-side
        match_iou:     IoU threshold for matching (default 0.5)

    Returns:
        dict: size_bin_label -> {recall_at_05, median_iou, fp_above_tp_median,
                                 gt_count, tp_count, fp_count, median_tp_conf}
    """
    bins = size_bins or DEFAULT_SIZE_BINS
    bin_labels = [f"{lo}-{hi}" for lo, hi in bins]

    preds_by_image = {}
    for p in jdict:
        preds_by_image.setdefault(p["image_id"], []).append(p)

    per_bin = {
        i: {"gt_count": 0, "gt_found": 0, "tp_ious": [],
            "tp_confs": [], "fp_confs": []}
        for i in range(len(bins))
    }

    for img_id, gts in gt_by_image.items():
        # Count all GTs per bin
        for gt in gts:
            gw, gh = gt["bbox"][2], gt["bbox"][3]
            per_bin[_bin_idx(min(gw, gh), bins)]["gt_count"] += 1

        img_preds = sorted(preds_by_image.get(img_id, []),
                           key=lambda x: -x["score"])
        if not img_preds:
            continue

        claimed = set()
        pred_is_tp = [False] * len(img_preds)

        for pi, pred in enumerate(img_preds):
            best_iou, best_gi = 0, -1
            for gi, gt in enumerate(gts):
                if gi in claimed or pred["category_id"] != gt["category_id"]:
                    continue
                iou = _iou_xywh(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            if best_iou >= match_iou and best_gi >= 0:
                claimed.add(best_gi)
                pred_is_tp[pi] = True
                gt = gts[best_gi]
                gw, gh = gt["bbox"][2], gt["bbox"][3]
                bi = _bin_idx(min(gw, gh), bins)
                per_bin[bi]["gt_found"] += 1
                per_bin[bi]["tp_ious"].append(best_iou)
                per_bin[bi]["tp_confs"].append(float(pred["score"]))

        for pi, pred in enumerate(img_preds):
            if not pred_is_tp[pi]:
                pw, ph = pred["bbox"][2], pred["bbox"][3]
                bi = _bin_idx(min(pw, ph), bins)
                per_bin[bi]["fp_confs"].append(float(pred["score"]))

    results = {}
    for i, label in enumerate(bin_labels):
        d = per_bin[i]
        recall = d["gt_found"] / d["gt_count"] if d["gt_count"] > 0 else 0.0
        median_iou = float(np.median(d["tp_ious"])) if d["tp_ious"] else 0.0
        median_tp_conf = float(np.median(d["tp_confs"])) if d["tp_confs"] else 0.0
        fp_above = sum(1 for c in d["fp_confs"] if c > median_tp_conf)
        fp_ratio = fp_above / len(d["fp_confs"]) if d["fp_confs"] else 0.0
        results[label] = {
            "recall_at_05": round(recall, 4),
            "median_iou": round(median_iou, 4),
            "fp_above_tp_median": round(fp_ratio, 4),
            "median_tp_conf": round(median_tp_conf, 4),
            "gt_count": d["gt_count"],
            "tp_count": len(d["tp_ious"]),
            "fp_count": len(d["fp_confs"]),
        }
    return results


class TrainingMonitor:
    """Attach as ultralytics callbacks to log bottleneck metrics per epoch."""

    CSV_FIELDS = [
        "epoch", "size_bin",
        "recall_at_05", "median_iou", "fp_above_tp_median",
        "median_tp_conf", "gt_count", "tp_count", "fp_count",
    ]

    def __init__(self, gt_path, output_csv, size_bins=None,
                 match_iou=0.5, every_n_epochs=1):
        self.gt_by_image = _load_gt_by_image(gt_path)
        self.output_csv = Path(output_csv)
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self.size_bins = size_bins or DEFAULT_SIZE_BINS
        self.match_iou = match_iou
        self.every_n_epochs = max(1, int(every_n_epochs))
        self._logged_epochs = set()  # dedupe — ultralytics fires callback again
                                     # on post-training best-weights final val

        # Write header if new
        if not self.output_csv.exists():
            with open(self.output_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
                writer.writeheader()

    def on_val_start(self, validator):
        validator.args.save_json = True

    def on_fit_epoch_end(self, trainer):
        epoch = int(getattr(trainer, "epoch", -1))
        if (epoch + 1) % self.every_n_epochs != 0:
            return
        if epoch in self._logged_epochs:
            # Callback fires again for the final best-weights val after training
            # completes; we already have this epoch recorded.
            return
        jdict = getattr(trainer.validator, "jdict", None)
        if not jdict:
            print(f"[TrainingMonitor] epoch {epoch}: jdict empty, skipping")
            return
        self._logged_epochs.add(epoch)

        metrics = compute_bottleneck_metrics(
            jdict, self.gt_by_image, self.size_bins, self.match_iou
        )

        with open(self.output_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
            for label, vals in metrics.items():
                row = {"epoch": epoch, "size_bin": label, **vals}
                writer.writerow(row)

        # Pretty print small-object row
        small_rows = [m for lbl, m in metrics.items() if lbl in ("0-8", "8-16")]
        if small_rows:
            s_recall = np.mean([r["recall_at_05"] for r in small_rows])
            s_iou = np.mean([r["median_iou"] for r in small_rows
                             if r["tp_count"] > 0]) if any(
                r["tp_count"] > 0 for r in small_rows) else 0.0
            s_fp = np.mean([r["fp_above_tp_median"] for r in small_rows])
            print(f"[TrainingMonitor] epoch {epoch}  small(<16px): "
                  f"recall@0.5={s_recall:.3f}  median_iou={s_iou:.3f}  "
                  f"fp>median_tp={s_fp:.3f}")


def register_training_monitor(model, gt_path, output_csv, size_bins=None,
                              match_iou=0.5, every_n_epochs=1):
    monitor = TrainingMonitor(gt_path, output_csv, size_bins,
                              match_iou, every_n_epochs)
    model.add_callback("on_val_start", monitor.on_val_start)
    model.add_callback("on_fit_epoch_end", monitor.on_fit_epoch_end)
    print(f"[TrainingMonitor] registered. CSV: {monitor.output_csv}")
    print(f"[TrainingMonitor] size bins: {monitor.size_bins}")
    print(f"[TrainingMonitor] logging every {monitor.every_n_epochs} epoch(s)")
    return monitor
