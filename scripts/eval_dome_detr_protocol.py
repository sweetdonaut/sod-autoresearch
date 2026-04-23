"""Evaluate a trained ultralytics YOLO model on VisDrone val using the
SAME COCO eval protocol as Dome-DETR (ACM MM 2025), so our number and
their reported 39.0% mAP@[.5:.95] speak the same language.

Dome-DETR eval specifics (from src/data/dataset/coco_eval_visdrone.py):
    - maxDets = [1, 10, 100, 500]; primary AP reported at maxDets=500
    - COCO standard area bins (small <32², med 32²-96², large >96²)
    - num_classes = 12 (VisDrone raw 0-11 scheme)
    - ignore-region filter: predictions overlapping cat=0 /
      iscrowd=1 / ignore=1 annotations at IoF >= 0.5 are DROPPED
    - VisdroneCOCOeval_faster (faster_coco_eval backend)

Our model outputs classes 0-9 (ultralytics VisDrone remap). We shift
them to 1-10 to match the Dome-DETR cat_id convention. Our model
cannot emit cat 0 (ignore) or cat 11 (others) because those weren't
in its training label set; those AP columns will show 0 / NaN.

Default imgsz: 1280 letterbox (our protocol). To also test 800 hard-resize
matching Dome-DETR exactly, pass --imgsz 800.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Use Dome-DETR's actual eval code. The file at
#   external/dome_detr_repo/src/data/dataset/coco_eval_visdrone.py
# is a verbatim copy of RicePasteM/Dome-DETR@master (SHA recorded in git).
# Dome-DETR's file has 2 relative imports we don't use:
#   from ...core import register       -> used only by VisdroneCocoEvaluator
#   from ...misc import dist_utils     -> used only by merge() (multi-GPU)
# We stub both via the mirrored directory at
#   external/dome_detr_repo/src/core/__init__.py
#   external/dome_detr_repo/src/misc/dist_utils.py
# The VisdroneCOCOeval_faster class + detections_in_ignore_regions helper
# do NOT touch those stubs, so the AP numbers come out of their code
# unchanged.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(PROJECT / "external" / "dome_detr_repo"))
from src.data.dataset.coco_eval_visdrone import (  # noqa: E402
    VisdroneCOCOeval_faster,
    VisdroneCocoEvaluator,
    detections_in_ignore_regions,
)
from faster_coco_eval import COCO  # noqa: E402


def _collect_ignore_regions(coco_gt):
    """Wrapper around Dome-DETR's classmethod (their impl lives on
    VisdroneCocoEvaluator._collect_ignore_regions)."""
    return VisdroneCocoEvaluator._collect_ignore_regions(coco_gt)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model_path, images_dir, imgsz, max_det=500, conf=0.001,
                  iou=0.7, hard_resize=False):
    """Run ultralytics YOLO inference on all JPGs, return list of COCO-format
    predictions with category_id remapped 0-9 -> 1-10 (Dome-DETR convention).

    Args:
        hard_resize: if True, pre-resize images to (imgsz, imgsz) hard
            (aspect squash, Dome-DETR style). If False, use ultralytics'
            default letterbox (our protocol).
    """
    from ultralytics import YOLO
    model = YOLO(model_path)

    coco_preds = []
    img_paths = sorted(Path(images_dir).glob("*.jpg"))
    print(f"[infer] {len(img_paths)} images  imgsz={imgsz}  "
          f"hard_resize={hard_resize}  max_det={max_det}")

    for img_path in tqdm(img_paths, desc="infer"):
        stem = img_path.stem
        if hard_resize:
            # Dome-DETR eval: hard resize to imgsz×imgsz, squashing aspect.
            # Predict on resized array, then scale boxes back to original.
            with Image.open(img_path) as im:
                W0, H0 = im.size
                resized = im.resize((imgsz, imgsz), Image.BILINEAR)
                arr = np.array(resized)
            # model(arr, ...) still letterboxes to a square multiple of 32,
            # but since arr is already square and equals imgsz, the letterbox
            # is a no-op (pad=0, scale=1). rect=False keeps it square.
            results = model.predict(
                source=arr, imgsz=imgsz, conf=conf, iou=iou,
                max_det=max_det, augment=False, agnostic_nms=False,
                verbose=False, device=0,
            )[0]
            if results.boxes is None or len(results.boxes) == 0:
                continue
            # In (imgsz × imgsz) space:
            boxes_xyxy = results.boxes.xyxy.cpu().numpy()
            # Scale back to original image dims
            sx, sy = W0 / imgsz, H0 / imgsz
            boxes_xyxy[:, [0, 2]] *= sx
            boxes_xyxy[:, [1, 3]] *= sy
        else:
            # Our protocol: letterbox to imgsz
            results = model.predict(
                source=str(img_path), imgsz=imgsz, conf=conf, iou=iou,
                max_det=max_det, augment=False, agnostic_nms=False,
                verbose=False, device=0,
            )[0]
            if results.boxes is None or len(results.boxes) == 0:
                continue
            boxes_xyxy = results.boxes.xyxy.cpu().numpy()

        scores = results.boxes.conf.cpu().numpy()
        labels = results.boxes.cls.cpu().numpy().astype(int)

        # Convert xyxy -> xywh (COCO bbox format)
        xs = boxes_xyxy[:, 0]
        ys = boxes_xyxy[:, 1]
        ws = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        hs = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]

        for i in range(len(scores)):
            # Remap ultralytics 0-9 -> Dome-DETR 1-10
            coco_preds.append({
                "image_id": stem,
                "category_id": int(labels[i]) + 1,
                "bbox": [float(xs[i]), float(ys[i]),
                         float(ws[i]), float(hs[i])],
                "score": float(scores[i]),
            })
    return coco_preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True,
                    help="Path to best.pt / last.pt")
    ap.add_argument("--gt", default="/workspace/datasets/VisDrone/"
                                    "annotations/instances_val_12cls.json")
    ap.add_argument("--images-dir",
                    default="/workspace/datasets/VisDrone/images/val")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--hard-resize", action="store_true",
                    help="Pre-resize to imgsz×imgsz (Dome-DETR style); "
                         "default is ultralytics letterbox (our protocol)")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=500,
                    help="Per-image detection cap (Dome-DETR uses 500)")
    ap.add_argument("--name", required=True, help="Run name for output dir")
    ap.add_argument("--project-dir",
                    default=str(PROJECT / "runs" / "dome_detr_eval"))
    args = ap.parse_args()

    out_dir = Path(args.project_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] weights={args.weights}")
    print(f"[eval] gt={args.gt}")
    print(f"[eval] imgsz={args.imgsz}  hard_resize={args.hard_resize}")
    print(f"[eval] conf={args.conf}  iou={args.iou}  max_det={args.max_det}")

    # 1. Load GT first (need stem->int_id map before writing preds)
    print(f"[eval] loading GT ...")
    coco_gt = COCO(args.gt)
    stem_to_id = {
        Path(img["file_name"]).stem: img["id"]
        for img in coco_gt.dataset["images"]
    }
    print(f"[eval] {len(stem_to_id)} images in GT")

    # 2. Run inference. predictions come out with stem as image_id; remap
    #    to the int id matching the GT.
    coco_preds = run_inference(
        args.weights, args.images_dir, args.imgsz,
        max_det=args.max_det, conf=args.conf, iou=args.iou,
        hard_resize=args.hard_resize,
    )
    for p in coco_preds:
        p["image_id"] = stem_to_id[p["image_id"]]
    pred_json = out_dir / "predictions_raw.json"
    with open(pred_json, "w") as f:
        json.dump(coco_preds, f)
    print(f"[eval] wrote {len(coco_preds)} raw preds -> {pred_json}")

    ignore_regions = _collect_ignore_regions(coco_gt)
    print(f"[eval] images with ignore regions: {len(ignore_regions)}")

    # 3. Apply ignore-region filter (IoF >= 0.5 -> drop)
    #    Group preds by image, filter, re-flatten.
    by_img = {}
    for p in coco_preds:
        by_img.setdefault(p["image_id"], []).append(p)

    filtered = []
    n_dropped = 0
    for img_id, preds in by_img.items():
        ig = ignore_regions.get(img_id)
        if ig is None:
            filtered.extend(preds)
            continue
        boxes_xyxy = torch.tensor(
            [[p["bbox"][0], p["bbox"][1],
              p["bbox"][0] + p["bbox"][2],
              p["bbox"][1] + p["bbox"][3]] for p in preds],
            dtype=torch.float32,
        )
        drop = detections_in_ignore_regions(boxes_xyxy, ig, 0.5)
        for keep, p in zip((~drop).tolist(), preds):
            if keep:
                filtered.append(p)
            else:
                n_dropped += 1
    print(f"[eval] ignore-filter dropped {n_dropped}/{len(coco_preds)} preds")

    filtered_json = out_dir / "predictions_filtered.json"
    with open(filtered_json, "w") as f:
        json.dump(filtered, f)

    # 4. Run Dome-DETR-protocol evaluator
    if not filtered:
        print("[eval] no predictions after filter — abort")
        sys.exit(1)
    print(f"[eval] running VisdroneCOCOeval_faster (maxDets=[1,10,100,500])")
    coco_dt = coco_gt.loadRes(str(filtered_json))
    E = VisdroneCOCOeval_faster(coco_gt, iou_type="bbox")
    # Mirror Dome-DETR's own wiring (src/data/dataset/coco_eval_visdrone.py:170):
    # cocoDt + imgIds must be set on the eval object before evaluate().
    E.cocoDt = coco_dt
    E.params.imgIds = sorted(coco_gt.getImgIds())
    E.evaluate()
    E.accumulate()
    E.summarize()

    # 5. Save stats
    stat_names = [
        "AP_all@500",           # primary
        "AP50@500", "AP75@500",
        "AP_small@500", "AP_medium@500", "AP_large@500",
        "AR@1", "AR@10", "AR@100", "AR@500",
        "AR_small@500", "AR_medium@500", "AR_large@500",
    ]
    # Dome-DETR truncates self.stats to 12 (drops AR_large); full 13-stat
    # array lives on self.all_stats.
    stats_arr = E.all_stats if hasattr(E, "all_stats") else E.stats
    stats = {name: float(stats_arr[i]) for i, name in enumerate(stat_names)}
    (out_dir / "dome_detr_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\n[eval] saved: {out_dir/'dome_detr_stats.json'}")
    print(f"\n=== Dome-DETR-protocol results ({args.name}) ===")
    for k, v in stats.items():
        print(f"  {k:<22s} {v:.4f}")


if __name__ == "__main__":
    main()
