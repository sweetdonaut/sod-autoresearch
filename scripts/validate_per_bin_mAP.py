"""Validate compute_per_bin_mAP.py against faster-coco-eval.

Use AREA-based bins matching COCO standard (S/M/L), expect exact match.
"""
import contextlib, io, json
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from faster_coco_eval import COCOeval_faster, COCO

# Override SIZE_BINS in compute_per_bin_mAP to use COCO area thresholds via short-side proxy
# Easiest: monkey-patch bin_fn to use area
import compute_per_bin_mAP as cpb

# COCO standard: small area<32², medium 32²-96², large>=96²
COCO_AREA_BINS = [(0, 32**2), (32**2, 96**2), (96**2, 1e10)]
COCO_BIN_LABELS = ['S(<32²)', 'M(32²-96²)', 'L(>96²)']

def area_bin_bbox(bbox):
    """For DETECTIONS (no segmentation, use bbox area)."""
    a = bbox[2] * bbox[3]
    for i, (lo, hi) in enumerate(COCO_AREA_BINS):
        if lo <= a < hi: return i
    return len(COCO_AREA_BINS) - 1


def area_bin_from_value(area):
    """For GT (use ann['area'] = segmentation polygon area)."""
    for i, (lo, hi) in enumerate(COCO_AREA_BINS):
        if lo <= area < hi: return i
    return len(COCO_AREA_BINS) - 1


def load_gt_using_seg_area(path):
    """COCO-style: GT bin uses ann['area'] (segmentation area)."""
    with open(path) as f:
        coco = json.load(f)
    gt_by_image = defaultdict(list)
    gt_count = defaultdict(lambda: defaultdict(int))
    for ann in coco['annotations']:
        bi = area_bin_from_value(ann['area'])
        ann['_bin'] = bi
        ann['_is_crowd'] = bool(ann.get('iscrowd', 0))
        gt_by_image[ann['image_id']].append(ann)
        if not ann['_is_crowd']:
            gt_count[bi][ann['category_id']] += 1
    return dict(gt_by_image), gt_count


# Patch
cpb.SIZE_BINS = COCO_AREA_BINS
cpb.BIN_LABELS = COCO_BIN_LABELS

GT = "/workspace/datasets/coco/annotations/instances_val2017.json"


def faster_coco_mAP(pred_path):
    coco_gt = COCO(GT)
    coco_dt = coco_gt.loadRes(str(pred_path))
    ce = COCOeval_faster(coco_gt, coco_dt, 'bbox')
    with contextlib.redirect_stdout(io.StringIO()):
        ce.evaluate(); ce.accumulate(); ce.summarize()
    return {
        'S(<32²)':   ce.stats[3],
        'M(32²-96²)': ce.stats[4],
        'L(>96²)':   ce.stats[5],
    }


def main():
    print(f"Loading GT using ann['area'] (segmentation area, COCO-style)...")
    gt_by_image, gt_count = load_gt_using_seg_area(GT)
    print(f"GT counts per area bin (S/M/L): "
          f"{[sum(gt_count[b].values()) for b in range(3)]}")

    runs = [
        ('n',     'runs/baselines/baseline_yolo26n/predictions.json'),
        ('n+P2',  'runs/p2/p1_exp1_sat30/predictions.json'),
        ('s',     'runs/baselines/baseline_yolo26s/predictions.json'),
        ('s+P2',  'runs/p2/p1_exp2_s_15ep/predictions.json'),
    ]

    print(f"\n{'variant':<8} {'bin':<12} {'mine (mAP50-95)':>16} {'faster-coco':>15} {'diff':>10}")
    print("-" * 70)
    for name, path in runs:
        _, mine = cpb.compute_mAP_per_bin(path, gt_by_image, gt_count, area_bin_bbox)
        fco = faster_coco_mAP(path)
        for bin_label in COCO_BIN_LABELS:
            d = mine[bin_label] - fco[bin_label]
            mark = "OK" if abs(d) < 0.005 else "DIFF"
            print(f"{name:<8} {bin_label:<12} {mine[bin_label]:>16.4f} {fco[bin_label]:>15.4f} {d:>+10.4f}  {mark}")
        print()


if __name__ == "__main__":
    main()
