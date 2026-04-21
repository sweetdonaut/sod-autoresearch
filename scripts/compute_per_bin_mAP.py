"""Compute per-size-bin mAP from ultralytics predictions.json + COCO GT.

COCO-faithful matching: per (image, class, bin) — prefer in-bin GTs (treat
out-of-bin as ignore). maxDets=100 per image. 101-pt AP interpolation.
"""
import json
from collections import defaultdict
from pathlib import Path

GT_PATH = "/workspace/datasets/coco/annotations/instances_val2017.json"
SIZE_BINS = [(0,8),(8,16),(16,32),(32-1,32),(32,64),(64,128),(128,9999)]  # placeholder, overridden below
SIZE_BINS = [(0,8),(8,16),(16,32),(32,64),(64,128),(128,9999)]
BIN_LABELS = [f"{lo}-{hi}" for lo,hi in SIZE_BINS]
IOU_THRS = [round(0.5 + 0.05*i, 2) for i in range(10)]
MAX_DETS = 100


def iou_xywh(b1, b2):
    x1,y1,w1,h1 = b1; x2,y2,w2,h2 = b2
    xi1=max(x1,x2); yi1=max(y1,y2)
    xi2=min(x1+w1,x2+w2); yi2=min(y1+h1,y2+h2)
    inter = max(0,xi2-xi1) * max(0,yi2-yi1)
    union = w1*h1 + w2*h2 - inter
    return inter/union if union>0 else 0.0


def short_side_bin(bbox):
    return min_bin(min(bbox[2], bbox[3]))


def min_bin(short):
    for i,(lo,hi) in enumerate(SIZE_BINS):
        if lo <= short < hi: return i
    return len(SIZE_BINS)-1


def load_gt(path, bin_fn):
    with open(path) as f:
        coco = json.load(f)
    gt_by_image = defaultdict(list)
    gt_count = defaultdict(lambda: defaultdict(int))
    for ann in coco['annotations']:
        bi = bin_fn(ann['bbox'])
        ann['_bin'] = bi
        ann['_is_crowd'] = bool(ann.get('iscrowd', 0))
        gt_by_image[ann['image_id']].append(ann)
        if not ann['_is_crowd']:  # crowd anns don't count as positives
            gt_count[bi][ann['category_id']] += 1
    return dict(gt_by_image), gt_count


def ap_101pt(precisions, recalls):
    if not recalls: return 0.0
    prec = list(precisions); rec = list(recalls)
    for i in range(len(prec)-1, 0, -1):
        prec[i-1] = max(prec[i-1], prec[i])
    ap = 0.0
    for r in (i/100 for i in range(101)):
        idx = next((i for i,rr in enumerate(rec) if rr >= r), None)
        ap += prec[idx] if idx is not None else 0.0
    return ap / 101


def compute_per_image_iou(img_dets, img_gts):
    """Pre-compute IoU matrix [n_dets x n_gts] for one image (same class only set to 0)."""
    iou_mat = [[0.0]*len(img_gts) for _ in range(len(img_dets))]
    for di, d in enumerate(img_dets):
        for gi, g in enumerate(img_gts):
            if d['category_id'] == g['category_id']:
                iou_mat[di][gi] = iou_xywh(d['bbox'], g['bbox'])
    return iou_mat


def match_one_image_one_bin(img_dets, img_gts, iou_mat, target_bin, bin_fn, iou_thr):
    """COCO-faithful per-bin matching for one image.

    Returns list of (conf, label) where label = 1 for TP, 0 for in-bin FP.
    Out-of-bin matches and out-of-bin unmatched dets are silently dropped.
    """
    # GT order: in-bin first (non-ignore), then out-of-bin (ignore)
    in_idx  = [gi for gi, g in enumerate(img_gts) if g['_bin'] == target_bin]
    out_idx = [gi for gi, g in enumerate(img_gts) if g['_bin'] != target_bin]
    sorted_gt_idx = in_idx + out_idx
    n_in = len(in_idx)

    claimed = set()
    out_records = []
    for di, d in enumerate(img_dets):
        best_iou = iou_thr
        best_pos = -1  # position in sorted_gt_idx
        for pos, gi in enumerate(sorted_gt_idx):
            if gi in claimed:
                continue
            # If we already matched a non-ignore (in-bin) and now hitting ignore — break
            if best_pos != -1 and best_pos < n_in and pos >= n_in:
                break
            i = iou_mat[di][gi]
            if i < best_iou:
                continue
            best_iou = i
            best_pos = pos
        det_b = bin_fn(d['bbox'])
        if best_pos != -1:
            claimed.add(sorted_gt_idx[best_pos])
            if best_pos < n_in:
                out_records.append((d['score'], 1))  # TP for this bin
            # else: matched to ignore → dropped
        else:
            if det_b == target_bin:
                out_records.append((d['score'], 0))  # FP for this bin
            # else: dropped
    return out_records


def compute_mAP_per_bin(pred_path, gt_by_image, gt_count_per_bin, bin_fn=short_side_bin):
    with open(pred_path) as f:
        preds = json.load(f)

    # Group preds by (image_id, category_id) and apply maxDets per (img, cat)
    # COCO eval: top maxDets by conf per (image, cat)
    dets_by_img_cat = defaultdict(list)
    for p in preds:
        dets_by_img_cat[(p['image_id'], p['category_id'])].append(p)
    for k in dets_by_img_cat:
        dets_by_img_cat[k] = sorted(dets_by_img_cat[k], key=lambda x: -x['score'])[:MAX_DETS]

    # Group GTs by (image_id, category_id)
    gts_by_img_cat = defaultdict(list)
    for img_id, gts in gt_by_image.items():
        for g in gts:
            gts_by_img_cat[(img_id, g['category_id'])].append(g)

    n_bins = len(SIZE_BINS)
    ap_per_bin_per_iou = defaultdict(dict)

    # All (img_id, cat) keys that appear anywhere
    all_keys = set(dets_by_img_cat.keys()) | set(gts_by_img_cat.keys())

    for iou_thr in IOU_THRS:
        for bi in range(n_bins):
            per_class_records = defaultdict(list)  # cat -> [(conf, label), ...]
            for (img_id, cat) in all_keys:
                cat_dets = dets_by_img_cat.get((img_id, cat), [])
                cat_gts = gts_by_img_cat.get((img_id, cat), [])
                if not cat_dets:
                    continue
                # Pre-sort GTs: in-bin non-crowd first, then ignore (out-bin or crowd)
                in_gts  = [g for g in cat_gts if g['_bin'] == bi and not g['_is_crowd']]
                ig_gts  = [g for g in cat_gts if g['_bin'] != bi or g['_is_crowd']]
                ordered_gts = in_gts + ig_gts
                n_in = len(in_gts)
                # Track claimed by GT id, but crowd is reusable
                claimed = set()
                for d in cat_dets:
                    best_iou = iou_thr
                    best_pos = -1
                    for pos, g in enumerate(ordered_gts):
                        is_crowd = g['_is_crowd']
                        gid = g['id']
                        if gid in claimed and not is_crowd:
                            continue
                        # If we have non-ignore match and now hitting ignore region — break
                        if best_pos != -1 and best_pos < n_in and pos >= n_in:
                            break
                        i = iou_xywh(d['bbox'], g['bbox'])
                        if i < best_iou: continue
                        best_iou = i; best_pos = pos
                    det_b = bin_fn(d['bbox'])
                    if best_pos != -1:
                        matched_g = ordered_gts[best_pos]
                        if not matched_g['_is_crowd']:
                            claimed.add(matched_g['id'])
                        if best_pos < n_in:
                            per_class_records[cat].append((d['score'], 1))  # TP
                        # else: ignored
                    else:
                        if det_b == bi:
                            per_class_records[cat].append((d['score'], 0))  # FP
                        # else: ignored

            # AP per class, average
            aps = []
            for cat, n_pos in gt_count_per_bin[bi].items():
                if n_pos == 0: continue
                recs = sorted(per_class_records.get(cat, []), key=lambda x: -x[0])
                tp = fp = 0; prec = []; rec = []
                for conf, t in recs:
                    if t: tp += 1
                    else: fp += 1
                    prec.append(tp/(tp+fp))
                    rec.append(tp/n_pos)
                aps.append(ap_101pt(prec, rec))
            ap_per_bin_per_iou[(bi, iou_thr)] = sum(aps)/len(aps) if aps else 0.0

    mAP50 = {BIN_LABELS[bi]: ap_per_bin_per_iou[(bi, 0.5)] for bi in range(n_bins)}
    mAP50_95 = {}
    for bi in range(n_bins):
        vals = [ap_per_bin_per_iou[(bi, t)] for t in IOU_THRS]
        mAP50_95[BIN_LABELS[bi]] = sum(vals)/len(vals)
    return mAP50, mAP50_95


def main():
    print(f"[load GT] {GT_PATH}  (binning by short-side)")
    gt_by_image, gt_count_per_bin = load_gt(GT_PATH, short_side_bin)
    print(f"[GT] {sum(len(v) for v in gt_by_image.values())} anns, "
          f"per-bin: {[sum(gt_count_per_bin[b].values()) for b in range(len(SIZE_BINS))]}")

    runs = [
        ('n',     'runs/baselines/baseline_yolo26n/predictions.json'),
        ('n+P2',  'runs/p2/p1_exp1_sat30/predictions.json'),
        ('s',     'runs/baselines/baseline_yolo26s/predictions.json'),
        ('s+P2',  'runs/p2/p1_exp2_s_15ep/predictions.json'),
        ('m',     'runs/baselines/baseline_yolo26m/predictions.json'),
        ('l',     'runs/baselines/baseline_yolo26l/predictions.json'),
        ('x',     'runs/baselines/baseline_yolo26x/predictions.json'),
    ]

    results50, results5095 = {}, {}
    for name, path in runs:
        p = Path(path)
        if not p.exists():
            print(f"MISSING: {path}"); continue
        print(f"\n[{name}] computing...")
        m50, m5095 = compute_mAP_per_bin(p, gt_by_image, gt_count_per_bin, short_side_bin)
        results50[name] = m50; results5095[name] = m5095
        print(f"  mAP50    : " + "  ".join(f"{b}={m50[b]:.3f}" for b in BIN_LABELS))
        print(f"  mAP50-95 : " + "  ".join(f"{b}={m5095[b]:.3f}" for b in BIN_LABELS))

    Path('/tmp/mAP_per_bin.json').write_text(json.dumps({'mAP50':results50,'mAP50-95':results5095}, indent=2))
    print("\nsaved: /tmp/mAP_per_bin.json")


if __name__ == "__main__":
    main()
