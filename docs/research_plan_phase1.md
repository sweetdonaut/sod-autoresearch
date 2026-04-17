# Phase 1 — P2 Head Validation

## TL;DR

Train a YOLO26 with an added P2 (stride-4) detection head. Test whether it
primarily closes the **Recall** gap for small objects (identified as the
dominant bottleneck in Phase 0 for this model scale). Measure with three
per-size-bin direct metrics, compare against yolo26n baseline.

---

## 1. What Phase 0 Established (why P2 is the next test)

Phase 0 ran oracle-style bottleneck decomposition on 10 YOLO variants
(yolo11/26 × n/s/m/l/x) on COCO val2017. Three bottlenecks explain why
small-object AP lags large-object AP:

| Bottleneck | What it means | Evidence in yolo26s (representative n-scale-ish) |
|---|---|---|
| **Recall** | Model never emits any prediction near the GT | 47.8% of 0-8px GT unfound at IoU≥0.5 |
| **Regression** | Model finds the object but bbox drift is large | median IoU for 0-8px matched TPs: 0.67 (vs 0.94 for 128+px) |
| **FP-Calibration** | High-conf FPs outrank real TPs | 2.2% of 0-8px FPs have conf > median TP conf |

**Key finding**: the bottleneck **ranking depends on model size**:
- Small models (n/s): Recall dominates (~50% of AP_S headroom)
- Large models (l/x): FP-calibration takes over; x-scale flips to FP > Reg > Recall

A single "silver bullet" won't fix all sizes. Phase 1 targets the **smallest-model Recall bottleneck** because (a) it's the biggest single gap and (b) P2 has a principled architectural story for it.

## 2. Hypothesis: Why a P2 Head Should Help Recall

YOLO26n's detection heads are at strides `[8, 16, 32]` (P3, P4, P5). Adding P2
at stride 4 means:

1. **Denser anchor grid for small objects**. A 640×640 image gets:
   - P3 stride=8 → 80×80 = 6,400 anchor positions
   - **P2 stride=4 → 160×160 = 25,600 anchor positions (4×)**
   For a 6×6px object, whether *any* anchor can be assigned to it becomes
   much more likely on P2.

2. **Preserves spatial detail**. Small objects lose too much info when
   downsampled to stride=8; stride=4 features still resolve individual
   pixels of a 10px object.

3. **TAL (Task-Aligned Label Assigner) gets more candidates**. Phase 0.3
   simulation showed small objects routinely have zero geometric candidates
   on P3 (they fall outside all anchor points' radii). P2 gives them
   candidates to compete for.

**Expected effects on our three metrics**:

| Metric (small bins) | Expected direction | Confidence |
|---|---|---|
| `recall_at_05` | ↑ significantly | high — this is the direct mechanism |
| `median_iou` | ↑ slightly | medium — better features, but regression still hard |
| `fp_above_tp_median` | ↑ (worse) | medium — more proposals → more high-conf FPs |

Last point is the key risk: P2 may "buy recall but pay in FPs." Phase 1
quantifies the trade.

## 3. Experimental Design

### Model

- **Architecture**: `configs/yolo26-p2.yaml` — scale `n` (default).
  - 4 detection heads: P2 (stride 4), P3 (stride 8), P4 (stride 16), P5 (stride 32)
  - ~2.66M params, 9.5 GFLOPs (vs yolo26n 2.57M / 5.5 GFLOPs)
  - Parameter count barely changed; compute goes up due to 160×160 feature map

### Initialization

- **Warm-start**: load `yolo26n.pt` pretrained backbone; new P2 head gets random init.
  - Via `scripts/train_p2.py --pretrained yolo26n.pt`
  - ultralytics' `model.load()` transfers matching-shape weights and skips the rest.
- Avoids from-scratch ~300-epoch convergence; expect meaningful signal in 30 epochs.

### Data

- COCO train2017 → val2017 (standard).
- Same pipeline as Phase 0 baselines (so numbers are comparable).

### Training

| Param | Value | Rationale |
|---|---|---|
| `epochs` | 30 | Enough for backbone fine-tune + head convergence |
| `batch` | 16 | Conservative for P2 head memory on 16 GB GPU; bump to 32 if A100/4090 with plenty headroom |
| `imgsz` | 640 | Matches Phase 0 baseline; don't change — small-object metrics depend on resolution |
| `optimizer` | auto (→ SGD+momentum) | Ultralytics default, matches baseline |
| `workers` | 8 | Default; raise if CPU idle |
| `end2end` | — | Handled by ultralytics; val uses NMS by default for our config |

### Monitoring

Per-epoch, `training_monitor.py` writes to `runs/p2/p1_exp1/training_monitor.csv`:

```
epoch, size_bin, recall_at_05, median_iou, fp_above_tp_median, median_tp_conf, gt_count, tp_count, fp_count
```

Size bins: `0-8, 8-16, 16-32, 32-64, 64-128, 128-9999` (pixels, short-side).

## 4. Success Criteria

Phase 1 **passes** if, comparing final epoch of P2 run to baseline yolo26n
(per `eval_baseline.py`), **all three** hold for combined `0-16px` bins:

1. **Recall improves**: `recall_at_05` absolute ≥ **+5 percentage points**
   (e.g., baseline 0.22 → P2 ≥ 0.27 for 0-8 bin).
2. **Regression acceptable**: `median_iou` decreases by **≤ 0.02** (any increase is a bonus).
3. **FP tolerable**: `fp_above_tp_median` increases by **≤ +0.03** absolute.

**Partial pass** (1 passes, 2/3 fail): P2 works but has side effects.
Phase 2 would target the failing axis (FP-calibration adjustment or
regression loss tweak).

**Fail** (1 fails): P2 architecturally doesn't help; reconsider assumption
that stride=4 density is the binding constraint.

## 5. Deliverables

From the pod session, commit back to a branch named `phase1/<descriptive-tag>`:

| Path | What |
|---|---|
| `runs/baselines/baseline_yolo26n/baseline_metrics.csv` | Baseline snapshot |
| `runs/p2/p1_exp1/training_monitor.csv` | 30 epochs × 6 bins = ~180 rows |
| `runs/p2/p1_exp1/results.csv` | Ultralytics standard per-epoch metrics |
| `runs/p2/p1_exp1/args.yaml` | Exact training args (reproducibility) |
| `runs/p2/p1_exp1/weights/best.pt` | Best-fitness checkpoint (optional; ~10 MB) |

User will pull these from the branch; pod can then be terminated without
loss.

## 6. Cost & Time Estimate

- **Baseline eval**: 3 min on any GPU that fits yolo26n (trivial).
- **P2 training**: 30 epochs × ~20 min/epoch = **10 hours** on A6000/4090.
  - On H100: ~5 hours
  - On 3090/4080: ~12 hours
- **Pod price**: A6000 ~$0.5/hr, 4090 ~$0.6/hr → **$5-10** for the full run.

---

## 7. What This Experiment Does NOT Test

Calibrating what Phase 1 is and isn't:

- ❌ Does **not** test whether P2 generalizes to larger model scales (s/m/l/x) —
  that's Phase 2 if n works.
- ❌ Does **not** test segmentation — Phase 0 ruled out seg-trained detectors
  as a path; we stay pure bbox detect.
- ❌ Does **not** compare to SOTA small-object-detection methods (NWD, DyHead, etc.)
  — scope is "does P2 specifically move the Recall needle".
- ❌ Does **not** fine-tune hyperparameters — we use ultralytics defaults to
  keep the comparison apples-to-apples with Phase 0 baselines.
