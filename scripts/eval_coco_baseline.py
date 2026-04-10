"""Evaluate all YOLO11 and YOLO26 pretrained models on COCO val2017.

Uses faster-coco-eval (pycocotools compatible) to extract official COCO metrics
including per-size AP (AP_S, AP_M, AP_L) from saved predictions.json.
"""

import csv
from pathlib import Path

from faster_coco_eval import COCO, COCOeval_faster
from ultralytics import YOLO

MODELS = [
    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
    "yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt",
]

ANN_FILE = Path("/home/yclaizzs/ML_exploration/datasets/coco/annotations/instances_val2017.json")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)
CSV_PATH = OUTPUT_DIR / "coco_baseline.csv"

FIELDS = ["model", "end2end", "AP50-95", "AP50", "AP75", "AP_S", "AP_M", "AP_L"]

# coco_eval.stats order:
# [AP, AP50, AP75, AP_S, AP_M, AP_L, AR1, AR10, AR100, AR_S, AR_M, AR_L]
STAT_KEYS = ["AP50-95", "AP50", "AP75", "AP_S", "AP_M", "AP_L"]


def eval_coco(pred_json: Path) -> dict:
    """Run official COCO evaluation and return AP metrics."""
    coco_gt = COCO(str(ANN_FILE))
    coco_dt = coco_gt.loadRes(str(pred_json))
    coco_eval = COCOeval_faster(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return {k: round(float(coco_eval.stats[i]), 4) for i, k in enumerate(STAT_KEYS)}


def main():
    rows = []
    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}")
        print(f"{'='*60}")

        model = YOLO(model_name)

        # Disable end2end for YOLO26 to use NMS (matches official benchmark)
        detect_head = list(model.model.model.children())[-1]
        if getattr(detect_head, "end2end", False):
            detect_head.end2end = False

        # Run validation (saves predictions.json)
        metrics = model.val(
            data="coco.yaml",
            batch=32,
            imgsz=640,
            conf=0.001,
            iou=0.7,
            max_det=300,
        )

        # Extract predictions.json path from save_dir
        pred_json = Path(metrics.save_dir) / "predictions.json"
        if not pred_json.exists():
            print(f"  WARNING: {pred_json} not found, skipping COCO eval")
            continue

        # Run official COCO evaluation
        coco_metrics = eval_coco(pred_json)

        row = {"model": model_name.replace(".pt", ""), "end2end": False, **coco_metrics}
        rows.append(row)
        print(f"  => AP={row['AP50-95']}  AP50={row['AP50']}  AP75={row['AP75']}  "
              f"AP_S={row['AP_S']}  AP_M={row['AP_M']}  AP_L={row['AP_L']}")

    # Write CSV
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # Print summary table
    print(f"\n{'='*60}")
    print("COCO val2017 Baseline Results (official COCO eval)")
    print(f"conf=0.001, iou=0.7, max_det=300, imgsz=640")
    print(f"{'='*60}")
    print(f"{'Model':<12} {'e2e':>4} {'AP50-95':>8} {'AP50':>8} {'AP75':>8} {'AP_S':>8} {'AP_M':>8} {'AP_L':>8}")
    print("-" * 68)
    for r in rows:
        e2e_str = "Y" if r["end2end"] else "N"
        print(f"{r['model']:<12} {e2e_str:>4} {r['AP50-95']:>8} {r['AP50']:>8} {r['AP75']:>8} "
              f"{r['AP_S']:>8} {r['AP_M']:>8} {r['AP_L']:>8}")

    print(f"\nResults saved to {CSV_PATH}")


if __name__ == "__main__":
    main()
