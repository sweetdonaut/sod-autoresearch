"""Build a COCO-format GT JSON for VisDrone val where image_id = filename
stem (string). This matches the image_id ultralytics emits in jdict during
train-time val for YOLO-format datasets (int(stem) if numeric else stem,
and VisDrone filenames are non-numeric).

**Category ID convention: 1-indexed (CAT 1..10)**.
  Ultralytics' `save_json=True` (used by training_monitor via
  on_val_start) emits predictions with category_id = train_class + 1
  (COCO convention). Our GT must follow the same 1-indexed scheme so
  class-matched IoU matching in compute_bottleneck_metrics works.
  Earlier 0-indexed GT caused ~all predictions to mismatch by off-by-one,
  making per-bin recall look 5x worse than reality.

Output: {data-root}/annotations/instances_val_stem.json

Usage:
    python scripts/build_visdrone_coco_gt.py
    python scripts/build_visdrone_coco_gt.py --split train
"""
import argparse
import json
from pathlib import Path

from PIL import Image


VISDRONE_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van", "truck",
    "tricycle", "awning-tricycle", "bus", "motor",
]


def _yolo_to_coco_bbox(cx, cy, w, h, img_w, img_h):
    aw, ah = w * img_w, h * img_h
    ax = cx * img_w - aw / 2
    ay = cy * img_h - ah / 2
    return [ax, ay, aw, ah]


def build(data_root: Path, split: str):
    images_dir = data_root / "images" / split
    labels_dir = data_root / "labels" / split
    assert images_dir.is_dir(), f"missing: {images_dir}"
    assert labels_dir.is_dir(), f"missing: {labels_dir}"

    images, annotations = [], []
    ann_id = 1
    for img_path in sorted(images_dir.glob("*.jpg")):
        stem = img_path.stem  # image_id must match ultralytics jdict
        with Image.open(img_path) as im:
            w, h = im.size
        images.append({"id": stem, "file_name": img_path.name,
                       "width": w, "height": h})
        label_path = labels_dir / f"{stem}.txt"
        if not label_path.exists():
            continue
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
                # 1-indexed category_id to match ultralytics' save_json output
                annotations.append({
                    "id": ann_id, "image_id": stem,
                    "category_id": cls + 1,
                    "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": 0,
                })
                ann_id += 1
    categories = [{"id": i + 1, "name": n}
                  for i, n in enumerate(VISDRONE_NAMES)]
    return {"images": images, "annotations": annotations,
            "categories": categories}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root",
                    default="/workspace/datasets/VisDrone")
    ap.add_argument("--split", default="val",
                    choices=["train", "val", "test"])
    ap.add_argument("--out", default=None,
                    help="Output JSON path; default: "
                         "{data-root}/annotations/instances_{split}_stem.json")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out = Path(args.out) if args.out else (
        data_root / "annotations" /
        f"instances_{args.split}_stem.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[gt] building {args.split} from {data_root} ...")
    coco = build(data_root, args.split)
    with open(out, "w") as f:
        json.dump(coco, f)
    print(f"[gt] wrote {out}")
    print(f"[gt] images={len(coco['images'])}  "
          f"annotations={len(coco['annotations'])}  "
          f"categories={len(coco['categories'])}")


if __name__ == "__main__":
    main()
