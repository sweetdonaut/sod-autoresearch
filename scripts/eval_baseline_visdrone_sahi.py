"""Baseline: yolo26x COCO-pretrained + SAHI 640x640 on VisDrone val.

Reports two views from a single inference pass:

  1. Per-bin bottleneck metrics (Phase 1 style) — class-agnostic matching.
     Any pred bbox that overlaps a GT at IoU>=0.5 counts as TP regardless of
     category. This isolates the pure detection/localization signal from the
     COCO->VisDrone class taxonomy mismatch.

  2. Overall mAP via pycocotools — class-matched with COCO->VisDrone remap
     for the 6 overlappable classes (person, bicycle, car, motorcycle, bus,
     truck). VisDrone-only classes (people, van, tricycle, awning-tricycle)
     will have 0 recall; this is the honest transfer-baseline number.

Typical use:
    python scripts/eval_baseline_visdrone_sahi.py \\
        --model yolo26x.pt --name baseline_yolo26x_sahi640
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

from training_monitor import DEFAULT_SIZE_BINS, _bin_idx, _iou_xywh


COCO_TO_VISDRONE = {
    0: 0,  # person -> pedestrian (note: collapses people=1 as well)
    1: 2,  # bicycle
    2: 3,  # car
    3: 9,  # motorcycle -> motor
    5: 8,  # bus
    7: 5,  # truck
}

VISDRONE_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van", "truck",
    "tricycle", "awning-tricycle", "bus", "motor",
]


def _yolo_to_coco_bbox(cx, cy, w, h, img_w, img_h):
    aw, ah = w * img_w, h * img_h
    ax = cx * img_w - aw / 2
    ay = cy * img_h - ah / 2
    return [ax, ay, aw, ah]


def build_gt(images_dir: Path, labels_dir: Path):
    images, annotations = [], []
    gt_by_image = {}
    id_of_stem = {}
    ann_id = 1
    for i, img_path in enumerate(sorted(images_dir.glob("*.jpg")), start=1):
        stem = img_path.stem
        id_of_stem[stem] = i
        with Image.open(img_path) as im:
            w, h = im.size
        images.append({"id": i, "file_name": img_path.name,
                       "width": w, "height": h})
        label_path = labels_dir / f"{stem}.txt"
        per_img = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    p = line.strip().split()
                    if len(p) < 5:
                        continue
                    cls = int(p[0])
                    cx, cy, bw, bh = map(float, p[1:5])
                    bbox = _yolo_to_coco_bbox(cx, cy, bw, bh, w, h)
                    if bbox[2] <= 0 or bbox[3] <= 0:
                        continue
                    ann = {"id": ann_id, "image_id": i, "category_id": cls,
                           "bbox": bbox, "area": bbox[2] * bbox[3],
                           "iscrowd": 0}
                    ann_id += 1
                    annotations.append(ann)
                    per_img.append(ann)
        gt_by_image[i] = per_img
    categories = [{"id": i, "name": n} for i, n in enumerate(VISDRONE_NAMES)]
    coco_gt = {"images": images, "annotations": annotations,
               "categories": categories}
    return gt_by_image, coco_gt, id_of_stem


def compute_class_agnostic_bin_metrics(preds_by_img, gt_by_img,
                                        size_bins=None, match_iou=0.5):
    bins = size_bins or DEFAULT_SIZE_BINS
    bin_labels = [f"{lo}-{hi}" for lo, hi in bins]
    per_bin = {i: {"gt_count": 0, "gt_found": 0, "tp_ious": [],
                   "tp_confs": [], "fp_confs": []}
               for i in range(len(bins))}

    for img_id, gts in gt_by_img.items():
        for gt in gts:
            gw, gh = gt["bbox"][2], gt["bbox"][3]
            per_bin[_bin_idx(min(gw, gh), bins)]["gt_count"] += 1

        img_preds = sorted(preds_by_img.get(img_id, []),
                           key=lambda x: -x["score"])
        if not img_preds:
            continue
        claimed = set()
        pred_is_tp = [False] * len(img_preds)
        for pi, pred in enumerate(img_preds):
            best_iou, best_gi = 0.0, -1
            for gi, gt in enumerate(gts):
                if gi in claimed:
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
        recall = d["gt_found"] / d["gt_count"] if d["gt_count"] else 0.0
        med_iou = float(np.median(d["tp_ious"])) if d["tp_ious"] else 0.0
        med_conf = float(np.median(d["tp_confs"])) if d["tp_confs"] else 0.0
        fp_above = sum(1 for c in d["fp_confs"] if c > med_conf)
        fp_ratio = fp_above / len(d["fp_confs"]) if d["fp_confs"] else 0.0
        results[label] = {
            "recall_at_05": round(recall, 4),
            "median_iou": round(med_iou, 4),
            "fp_above_tp_median": round(fp_ratio, 4),
            "median_tp_conf": round(med_conf, 4),
            "gt_count": d["gt_count"],
            "tp_count": len(d["tp_ious"]),
            "fp_count": len(d["fp_confs"]),
        }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26x.pt")
    ap.add_argument("--data-root", default="/root/datasets/VisDrone")
    ap.add_argument("--slice", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--name", default="baseline_yolo26x_sahi640")
    ap.add_argument("--project-dir",
                    default=str(PROJECT / "runs" / "baselines"))
    args = ap.parse_args()

    out_dir = Path(args.project_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    images_dir = Path(args.data_root) / "images" / "val"
    labels_dir = Path(args.data_root) / "labels" / "val"
    print(f"[setup] images={images_dir}")
    print(f"[setup] labels={labels_dir}")

    print(f"[setup] building GT ...")
    gt_by_img, coco_gt, _ = build_gt(images_dir, labels_dir)
    print(f"[setup] images={len(coco_gt['images'])}  "
          f"gt={len(coco_gt['annotations'])}")

    gt_json = out_dir / "visdrone_val_gt.json"
    with open(gt_json, "w") as f:
        json.dump(coco_gt, f)

    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    model_path = args.model
    if not Path(model_path).is_absolute():
        candidate = PROJECT / model_path
        if candidate.exists():
            model_path = str(candidate)

    print(f"[sahi] loading {model_path}")
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=args.conf,
        device="cuda:0",
    )

    preds_remap = []          # for pycocotools (COCO->VisDrone remap only)
    preds_by_img = {}         # class-agnostic for bin metrics
    print(f"[sahi] slice={args.slice} overlap={args.overlap} "
          f"conf={args.conf}")
    for img_info in tqdm(coco_gt["images"], desc="sahi-inference"):
        img_path = images_dir / img_info["file_name"]
        img_id = img_info["id"]
        res = get_sliced_prediction(
            image=str(img_path),
            detection_model=model,
            slice_height=args.slice,
            slice_width=args.slice,
            overlap_height_ratio=args.overlap,
            overlap_width_ratio=args.overlap,
            perform_standard_pred=True,
            verbose=0,
        )
        img_preds_list = []
        for op in res.object_prediction_list:
            bbox = op.bbox.to_xywh()
            coco_cat = op.category.id
            score = float(op.score.value)
            img_preds_list.append({
                "image_id": img_id,
                "category_id": 0,
                "bbox": bbox,
                "score": score,
            })
            vd_cat = COCO_TO_VISDRONE.get(coco_cat)
            if vd_cat is not None:
                preds_remap.append({
                    "image_id": img_id,
                    "category_id": vd_cat,
                    "bbox": bbox,
                    "score": score,
                })
        preds_by_img[img_id] = img_preds_list

    n_total = sum(len(v) for v in preds_by_img.values())
    print(f"[sahi] total preds={n_total}  remapped={len(preds_remap)}")
    pred_json = out_dir / "predictions_remapped.json"
    with open(pred_json, "w") as f:
        json.dump(preds_remap, f)

    print(f"\n[bins] class-agnostic bottleneck metrics ...")
    metrics = compute_class_agnostic_bin_metrics(preds_by_img, gt_by_img)
    csv_path = out_dir / "baseline_metrics.csv"
    fields = ["size_bin", "recall_at_05", "median_iou", "fp_above_tp_median",
              "median_tp_conf", "gt_count", "tp_count", "fp_count"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for label, vals in metrics.items():
            writer.writerow({"size_bin": label, **vals})
    print(f"[bins] saved: {csv_path}")
    print(f"{'bin':<10}{'R@0.5':>8}{'medIoU':>8}{'FP>TPmed':>11}"
          f"{'GT':>8}{'TP':>8}{'FP':>9}")
    for label, m in metrics.items():
        print(f"{label:<10}{m['recall_at_05']:>8.3f}"
              f"{m['median_iou']:>8.3f}{m['fp_above_tp_median']:>11.3f}"
              f"{m['gt_count']:>8}{m['tp_count']:>8}{m['fp_count']:>9}")

    print(f"\n[coco-eval] pycocotools mAP (class-matched, COCO->VisDrone) ...")
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco_gt_obj = COCO(str(gt_json))
    if preds_remap:
        coco_dt = coco_gt_obj.loadRes(str(pred_json))
        E = COCOeval(coco_gt_obj, coco_dt, iouType="bbox")
        E.evaluate(); E.accumulate(); E.summarize()
        summary = {
            "mAP_50_95": float(E.stats[0]), "mAP_50": float(E.stats[1]),
            "mAP_75": float(E.stats[2]), "mAP_small": float(E.stats[3]),
            "mAP_medium": float(E.stats[4]), "mAP_large": float(E.stats[5]),
            "AR_1": float(E.stats[6]), "AR_10": float(E.stats[7]),
            "AR_100": float(E.stats[8]), "AR_small": float(E.stats[9]),
            "AR_medium": float(E.stats[10]), "AR_large": float(E.stats[11]),
        }
        (out_dir / "coco_eval.json").write_text(json.dumps(summary, indent=2))
        print(f"[coco-eval] saved: {out_dir/'coco_eval.json'}")
    else:
        print("[coco-eval] no remapped preds — skipping")


if __name__ == "__main__":
    main()
