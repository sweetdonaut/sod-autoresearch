"""Build a 12-class VisDrone val COCO-GT JSON matching Dome-DETR's eval
protocol (num_classes=12, ignore-region aware).

Dome-DETR categories (= raw VisDrone categories):
    0: ignored-regions  (treated as ignore mask, NOT evaluated)
    1: pedestrian       5: van       9:  bus
    2: people           6: truck     10: motor
    3: bicycle          7: tricycle  11: others
    4: car              8: awning-tricycle

Ultralytics VisDrone.yaml drops cat 0 (ignore rows filtered) and shifts
cat 1-10 down by 1 (pedestrian=0..motor=9), and silently drops cat 11.
To compare with Dome-DETR we need the 12-class GT back.

Raw VisDrone annotation row format (per line in each .txt):
    bbox_left, bbox_top, bbox_width, bbox_height, score, category,
    truncation, occlusion
  - score=0 OR category=0 → this row is an "ignored region" mask.
    Everything else is a real detection.

For Dome-DETR's evaluator:
  - ignore-region annotations are marked iscrowd=1 AND ignore=1
    (the evaluator checks either flag, plus category==0)
  - real annotations keep their raw category_id (1..11)

Output:
    /workspace/datasets/VisDrone/annotations/instances_val_12cls.json

image_id is the filename stem string (matches ultralytics jdict).
"""
import argparse
import json
from pathlib import Path

from PIL import Image


# Raw VisDrone categories (1-indexed; cat 0 = ignore mask)
VISDRONE_12CLS = {
    0:  "ignored-regions",
    1:  "pedestrian",
    2:  "people",
    3:  "bicycle",
    4:  "car",
    5:  "van",
    6:  "truck",
    7:  "tricycle",
    8:  "awning-tricycle",
    9:  "bus",
    10: "motor",
    11: "others",
}


def build(images_dir: Path, raw_ann_dir: Path):
    """Build COCO GT. image_id is a sequential integer (1-indexed)
    because faster_coco_eval's COCO() forces int(id). file_name keeps
    the stem so the eval script can look up image_id from stem."""
    images, annotations = [], []
    ann_id = 1
    n_ignore = 0
    n_valid = 0
    for img_id_int, img_path in enumerate(sorted(images_dir.glob("*.jpg")),
                                           start=1):
        stem = img_path.stem
        with Image.open(img_path) as im:
            w, h = im.size
        images.append({"id": img_id_int, "file_name": img_path.name,
                       "width": w, "height": h})
        ann_path = raw_ann_dir / f"{stem}.txt"
        if not ann_path.exists():
            continue
        with open(ann_path) as f:
            for line in f:
                parts = [p.strip() for p in line.strip().split(",")
                         if p.strip() != ""]
                if len(parts) < 6:
                    continue
                try:
                    x, y, bw, bh = map(int, parts[:4])
                    score = int(parts[4])
                    cat = int(parts[5])
                except ValueError:
                    continue
                if bw <= 0 or bh <= 0:
                    continue

                is_ignore = (score == 0) or (cat == 0)
                ann = {
                    "id": ann_id,
                    "image_id": img_id_int,
                    "category_id": cat,       # raw 0-11 scheme
                    "bbox": [x, y, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 1 if is_ignore else 0,
                }
                if is_ignore:
                    ann["ignore"] = 1
                    n_ignore += 1
                else:
                    n_valid += 1
                annotations.append(ann)
                ann_id += 1

    categories = [{"id": i, "name": n} for i, n in VISDRONE_12CLS.items()]
    coco = {"images": images, "annotations": annotations,
            "categories": categories}
    return coco, n_valid, n_ignore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir",
                    default="/workspace/datasets/VisDrone/images/val")
    ap.add_argument("--raw-ann-dir",
                    default="/tmp/visdrone_val_raw/"
                            "VisDrone2019-DET-val/annotations")
    ap.add_argument("--out",
                    default="/workspace/datasets/VisDrone/annotations/"
                            "instances_val_12cls.json")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    raw_ann_dir = Path(args.raw_ann_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[12cls-gt] images={images_dir}")
    print(f"[12cls-gt] raw annotations={raw_ann_dir}")
    assert images_dir.is_dir(), f"missing: {images_dir}"
    assert raw_ann_dir.is_dir(), f"missing: {raw_ann_dir}"

    coco, n_valid, n_ignore = build(images_dir, raw_ann_dir)
    with open(out, "w") as f:
        json.dump(coco, f)

    print(f"[12cls-gt] wrote {out}")
    print(f"[12cls-gt] images={len(coco['images'])}  "
          f"valid={n_valid}  ignore={n_ignore}  "
          f"total_ann={len(coco['annotations'])}")
    by_cat = {}
    for ann in coco["annotations"]:
        by_cat[ann["category_id"]] = by_cat.get(ann["category_id"], 0) + 1
    print(f"[12cls-gt] per-category counts:")
    for cid, name in VISDRONE_12CLS.items():
        print(f"  {cid:2d} {name:<18s} {by_cat.get(cid, 0)}")


if __name__ == "__main__":
    main()
