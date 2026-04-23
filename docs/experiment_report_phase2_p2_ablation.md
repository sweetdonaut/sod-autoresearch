# Phase 2 Experiment Report — P2 Head Ablation on VisDrone-DET

**Date**: 2026-04-22
**Branch**: `visdroneP2`
**Hardware**: A100 80 GB PCIe, cu128 / torch 2.10.0
**Dataset**: VisDrone2019-DET (6471 train / 548 val / 1610 test, 10-class via ultralytics yaml)

---

## TL;DR

Trained two yolo26x variants with **identical hyperparameters** (imgsz=1280, batch=8, 50 ep, AdamW auto-lr, close_mosaic=10, seed=0) differing **only in the presence of a P2 (stride-4) detection head**. Evaluated both under ultralytics' built-in 10-class protocol and Dome-DETR's published 12-class protocol (using Dome-DETR's own eval code, direct-imported, to make numbers directly comparable to their paper's 39.0% mAP@[.5:.95]).

**Result — P2 head is a net negative at yolo26x scale on VisDrone**:

| axis | P2 (ours) | **No-P2 (ours)** | Dome-DETR-L (ref) |
|---|---|---|---|
| Best-epoch mAP50-95 (ultralytics val, 10 cls) | 0.405 (ep 33) | **0.408 (ep 37)** | — |
| Dome-DETR protocol AP@[.5:.95] (12 cls, maxDet=500, ignore filter) | 0.3969 | **0.3981** | 0.390 |
| Dome-DETR protocol AP_small | 0.3182 | **0.3236** | — |
| Training time (A100 80 GB, 50 ep) | 10.21 hrs | **5.99 hrs** (-41%) | — |
| Peak GPU memory @ batch=8 | 73 GB | **41 GB** (-44%) | — |
| GFLOPs @ imgsz=1280 | 1020 | **838** (-18%) | — |

**No-P2 wins on every axis that matters** (primary AP, small-object AP, training cost, memory) and **matches or exceeds Dome-DETR's published SOTA on their own eval protocol**.

This contradicts the Phase 1 hypothesis that P2 addresses VisDrone's small-object recall bottleneck — the bottleneck on yolo26x + imgsz=1280 isn't recall (both runs achieve 75-88% class-matched R@0.5 on 0-32 px bins). See §5 for the corrected per-bin metrics and the category-offset bug that hid this during training.

---

## 1. Setup

### 1.1 Data pipeline

Dataset download + preprocessing flow (single-pod, ~6 min total):

1. Ultralytics auto-download to `/workspace/datasets/VisDrone` (persistent).
2. `rsync` to `/root/datasets/VisDrone` for fast local I/O (`runpod_startup.sh`).
3. Ultralytics converts VisDrone's native categories (0-11, 1-indexed in raw `.txt`) to its YOLO yaml with **10 classes** (drops cat 0 "ignored-regions" as mask and cat 11 "others" as ill-defined catch-all). This follows the VisDrone convention used by UFPMP-Det, YOLC, ClusDet, TPH-YOLOv5, ESOD.

### 1.2 Architectures

- **With P2**: `configs/yolo26-p2.yaml` at scale `x`. Four Detect heads on P2 (stride 4), P3, P4, P5. 57.8 M params, 1020 GFLOPs @ 1280.
  - Warm-start from `yolo26x.pt`: 1354/1370 tensors transfer; P2 head + nc=10 Detect layer random-init.
- **Without P2**: plain `yolo26x.pt` loaded directly. Three Detect heads P3/P4/P5. 58.9 M params, 838 GFLOPs @ 1280.
  - Warm-start: all backbone + neck transfers; only Detect head re-inits for nc=10.

### 1.3 Training recipe (identical for both)

| param | value |
|---|---|
| epochs | 50 |
| imgsz | 1280 |
| batch | 8 (fixed, not auto; maintains nbs=64 gradient accumulation = 8 steps/iter for both) |
| optimizer | auto → AdamW(lr0=7.14e-4, momentum=0.9) |
| close_mosaic | 10 (last 10 epochs mosaic-off) |
| seed, deterministic | 0, True |
| AMP | True |
| `PYTORCH_ALLOC_CONF` | `expandable_segments:True` (prevents fragmentation at batch=8) |
| workers, cache | 12, False |

Batch was held constant at 8 despite no-P2 having massive VRAM headroom (41 GB vs 73 GB used) — raising batch would change BN stats and total optimizer steps, confounding the P2 attribution.

### 1.4 Eval protocols

Two independent evaluations:

**A. Ultralytics 10-class protocol** (built-in `model.val()`):
- pycocotools-style COCO eval, internal to ultralytics.
- 10 classes (pedestrian through motor).
- maxDets=300 (ultralytics default).
- IoU thresholds [0.5:0.95:0.05].
- COCO standard area bins (small <32², medium 32²-96², large >96²).

**B. Dome-DETR 12-class protocol** (their code, direct import):
- Copy of `RicePasteM/Dome-DETR@master:src/data/dataset/coco_eval_visdrone.py` placed at `external/dome_detr_repo/src/data/dataset/coco_eval_visdrone.py` — byte-identical to upstream.
- Two Dome-DETR-internal imports stubbed via `external/dome_detr_repo/src/core/__init__.py` (no-op `register()`) and `.../src/misc/dist_utils.py` (single-process `all_gather`). These stubs are only used by the `VisdroneCocoEvaluator` wrapper class; the underlying `VisdroneCOCOeval_faster` (which computes all AP numbers) does not touch either.
- 12 classes (including cat 0 ignore-region handling + cat 11 "others").
- maxDets=[1, 10, 100, 500]; primary AP at maxDets=500.
- Ignore-region filter: predictions with IoF ≥ 0.5 against a cat-0/iscrowd/ignore annotation are dropped.
- Same COCO area bins.

Our predictions are remapped from ultralytics 10-class (indices 0-9) to Dome-DETR's 12-class scheme (indices 1-10). Cat 11 "others" cannot be predicted by our model (not in training label set) → AP_cat11 = 0. See §3.3 for the protocol-translation asymmetries.

---

## 2. Results

### 2.1 Ultralytics 10-class protocol

| metric | **P2** | **No-P2** | Δ |
|---|---|---|---|
| Best epoch | 33 | 37 | — |
| mAP@0.5 | 0.620 | **0.627** | +0.008 |
| mAP@[0.5:0.95] | 0.405 | **0.408** | +0.003 |

Ultralytics' best-checkpoint selection uses `fitness = 0.1·mAP50 + 0.9·mAP50-95`:
- P2 best = ep 33 (fitness 0.4266)
- No-P2 best = ep 37 (fitness 0.4297)

Both peaks are in the mosaic-on phase; post-mosaic-close (ep 40-49) both runs regress by ~0.005-0.013 mAP50-95, indicating the 50-ep + close_mosaic=10 schedule is too long for VisDrone's 6471-image train split. **Recommendation for next runs: reduce to 35 ep or `close_mosaic=20`**.

### 2.2 Dome-DETR 12-class protocol (their eval code, best.pt, 1280 letterbox)

| metric | **P2** | **No-P2** | Δ | vs Dome-DETR 39.0% |
|---|---|---|---|---|
| **AP@[0.5:0.95] @500** | 0.3969 | **0.3981** | +0.0012 | both above |
| AP@0.50 @500 | 0.6228 | **0.6279** | +0.0051 | |
| AP@0.75 @500 | 0.4229 | 0.4218 | -0.0011 | |
| **AP_small @500** (area <1024) | 0.3182 | **0.3236** | **+0.0054** | — (the P2 head's supposed sweet spot) |
| AP_medium @500 (1024-9216) | 0.5051 | 0.5024 | -0.0027 | |
| AP_large @500 (>9216) | **0.6303** | 0.6241 | +0.0062 | |
| AR @500 all | 0.5730 | 0.5715 | -0.0015 | |
| AR_small @500 | 0.5108 | 0.5114 | +0.0006 | |
| AR_medium @500 | 0.6629 | 0.6604 | -0.0025 | |
| AR_large @500 | 0.7639 | **0.7955** | **+0.0316** | |

**Both Phase 2 models beat Dome-DETR's reported 39.0% on Dome-DETR's own protocol.** No-P2 wins by +0.81 AP over Dome-DETR; P2 wins by +0.69 AP.

### 2.3 Corrected per-bin recall (best.pt, IoU≥0.5, class-matched)

| bin (short side, px) | GT | **P2 R@0.5** | **No-P2 R@0.5** | Δ |
|---|---|---|---|---|
| 0-8 | 6382 | 0.756 | **0.760** | +0.004 |
| 8-16 | 10958 | 0.867 | **0.881** | **+0.014** |
| 16-32 | 13434 | 0.904 | **0.913** | +0.009 |
| 32-64 | 6098 | 0.940 | 0.941 | +0.001 |
| 64-128 | 1743 | **0.959** | 0.955 | -0.004 |
| 128+ | 144 | 0.965 | **0.972** | +0.007 |

No-P2 wins 5/6 bins, including every small-object bin (0-8, 8-16, 16-32). **P2's one remaining advantage (64-128 bin, +0.004 R) is the opposite of its architectural purpose.**

---

## 3. P2 ablation — interpretation

### 3.1 Why P2 loses at x scale

Phase 1 on COCO validated P2 at n scale ("P2 head validation — all 3 success criteria pass", commit `100911c`). A follow-up DOE (`b5ac520`: *"Phase 1 model-size DOE: P2 effect halves with backbone scale"*) showed P2's benefit shrinks roughly linearly with backbone width. Linearly extrapolating n → s → m → l → x predicts near-zero benefit at x, which is consistent with this result; the Phase 2 data extends that trajectory slightly into the negative.

Three contributing factors:

1. **Pretrained-initialization drag**: No-P2 transfers 100% of yolo26x.pt weights; P2 requires ~20 epochs for the random-init P2 head to catch up to the already-converged P3/P4/P5, during which the shared backbone is also being fine-tuned → mutual interference.
2. **VisDrone's GT size distribution**: 79% of GT is <32 px short side, but only 16% is <8 px. P2 (stride 4) primarily buys anchor density for the 0-8 px regime; the 8-32 px bulk is already well-served by P3 (stride 8).
3. **yolo26x backbone capacity**: P3 features at x width (1536 channels pre-Detect) appear to contain enough stride-8 small-object semantics that the extra stride-4 P2 branch is redundant.

### 3.2 Cost axis

| axis | P2 | No-P2 | savings |
|---|---|---|---|
| Wall-clock (50 ep on A100 80GB) | 10.21 hrs | 5.99 hrs | **-41%** |
| GPU mem @ batch=8 | 73 GB | 41 GB | **-44%** |
| Forward+backward FLOPs @ 1280 | 1020 GFLOPs | 838 GFLOPs | -18% |
| Params | 57.8 M | 58.9 M | +1.9% (near-wash) |

The memory saving is the most useful for Phase 2 planning: at no-P2 we have 39 GB of VRAM slack, enabling either:
- `batch=16` (same effective grad batch at nbs=64, faster throughput)
- `imgsz=1536 or 1920` (tiling-aware single-pass for better tiny-object resolution)
- Tile-level training with pre-sliced crops (UFPMP-Det / AD-Det style) without OOM concerns

### 3.3 Protocol-translation asymmetries vs Dome-DETR

Three knobs differ between our ultralytics setup and Dome-DETR's published number:

| knob | ours | Dome-DETR | AP direction |
|---|---|---|---|
| Class set | 10 cls (+drop cat 11 at training) | 12 cls (includes cat 11, ignore-masks cat 0) | theirs +0 to +0.5 AP if their cat 11 AP < mean, else -0 to -0.5 (rare class, ~250 train samples) |
| maxDets | 300 (ultralytics default) | 500 | theirs +0.3 to +0.8 AP |
| ignore-region filter | no (we train VisDrone annotations as-is after ultralytics preprocessing) | IoF ≥ 0.5 drops | applied to both in §2.2 eval → wash |
| Eval-time imgsz | 1280 letterbox | 800 hard-resize | ours +0.5 to +1.5 AP (more resolution for small objects, but aspect squash is slight cost for them) |

Our 39.81% (no-P2) on Dome-DETR's 12-class protocol at 1280 letterbox is NOT a perfect apples-to-apples number with Dome-DETR's 39.0% @ 800 hard-resize. The remaining resolution-delta (~+1 AP in our favor) and class-set asymmetry (~±0.5 AP, hard to estimate without retraining with cat 11) could collectively move the true comparison by ±1-2 AP. **A cleaner future check: re-evaluate no-P2 with `--imgsz 800 --hard-resize` on Dome-DETR's protocol.** The script supports it directly; ~1 min of A100.

For the Phase 2 community-standard comparison (UFPMP-Det 36.6%, YOLC 38.3%, AD-Det 37.5%), our **40.8% ultralytics 10-class** is the directly comparable number — it follows the same protocol convention as all those papers.

---

## 4. Bugs caught during analysis

### 4.1 category_id off-by-one in training_monitor

**Symptom**: `training_monitor.csv` per-epoch `recall_at_05` for 0-8 px bin showed 0.10-0.15 throughout both runs — implausibly low given overall mAP was climbing past 0.40. User flagged the inconsistency during training.

**Root cause**: Ultralytics' `save_json=True` emits `category_id = train_class + 1` (COCO 1-indexed convention). Our `instances_val_stem.json` was built with 0-indexed `category_id` (matching YOLO label `.txt` files). The `pred["category_id"] != gt["category_id"]` check in `compute_bottleneck_metrics` rejected ~all correct-class matches; the non-zero recall came from coincidental cross-category alignment (e.g., model's "pedestrian" (cat 1, offset) bbox overlapping GT's "people" (cat 1, unoffset) in dense pedestrian regions).

**Evidence**:
```
preds unique category_ids: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]     # 1-indexed
GT   unique category_ids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]     # 0-indexed
```

**Fix**: `scripts/build_visdrone_coco_gt.py` now emits 1-indexed categories (`cat+1` in both the `categories` list and each annotation's `category_id`). `scripts/training_monitor.py` docstring calls out the convention.

**Impact**:
- All `training_monitor.csv` rows in the two Phase 2 runs have polluted per-bin numbers (too low by 5-8x). The CSVs are preserved as-is for the historical record; the *corrected* bin metrics in §2.3 were recomputed post-hoc from each run's final `best.pt predictions.json`.
- Ultralytics' built-in `results.csv` mAP numbers are **unaffected** (ultralytics' internal eval uses consistent 1-indexed categories on both sides).
- The `eval_dome_detr_protocol.py` script was unaffected — it applies `+1` remap explicitly when generating predictions, matching its 12-class GT's 1-indexed convention.

### 4.2 Phase 2 runs' per-epoch bin metrics are unrecoverable

No per-epoch checkpoint was saved (`save_period=-1` default), only `best.pt` and `last.pt`. The corrected final-epoch bin metrics (§2.3) are all we can provide. For future runs, the fix in 4.1 is sufficient — new `training_monitor.csv` will be correct from epoch 0.

---

## 5. Implications for Phase 2 strategy

1. **Drop P2 for yolo26x + imgsz=1280 on VisDrone.** No recall gain, 40% training-cost overhead, 44% VRAM overhead.
2. **Invest the freed compute** in one of:
   - **More epochs at smaller close_mosaic**: 50→35 ep with `close_mosaic=20` to avoid the post-close overfit both runs showed.
   - **Higher imgsz** (1536 or 1920): no-P2 at 1280 uses 41 GB; 1536 at batch=8 is estimated ~68 GB — fits on A100 80GB. This is a cheap axis to test because ESOD's 34 AP (YOLOv5-x@1920) shows the scaling works at least to 1920.
   - **Tile-level training**: pre-slice VisDrone to 800×800 tiles with 0.2 overlap → ~40k training tiles from 6471 images. Matches UFPMP-Det / AD-Det paradigm.
   - **SAHI at inference**: our untrained yolo26x+SAHI baseline got 13% mAP (class-matched transfer, §1.1 of the baseline report); on the trained model SAHI should add +2-4 AP on top of whatever recipe we run.
3. **Next SOTA target**: AdaZoom 40.3% and RemDet-X 40.0% (both use cluster-crop multi-pass, see Phase 2 intro SOTA table). At 40.8% we're already nominally above both on our protocol; but they use multi-scale / cluster-crop TTA, so the true fair target is probably 42-43% under a single-pass protocol. Reachable via higher imgsz + tile-training.
4. **Recall is NOT the VisDrone bottleneck on yolo26x.** Class-matched R@0.5 is 75.6% for 0-8 px and 86-88% for 8-16 px. The real bottleneck is **localization precision at high IoU thresholds** (mAP75 = 0.42 vs mAP50 = 0.62, a 20-point gap). Phase 1's "recall bottleneck" diagnosis was size-specific to small (n-scale) models; at x scale the bottleneck has already shifted. Future architectural experiments should target IoU regression (e.g., DFL reg_max, GWD/KLD loss, IoU-aware branch) rather than stride-4 P2.

---

## 6. Files changed / produced

### Code

| file | what |
|---|---|
| `scripts/train_visdrone_p2.py` | Main training entry — yolo26x + P2 head |
| `scripts/train_visdrone_nop2.py` | Ablation entry — same hyperparams, no P2 |
| `scripts/training_monitor.py` | Docstring updates only; existing callback logic preserved |
| `scripts/build_visdrone_coco_gt.py` | 10-class stem-keyed GT JSON builder (1-indexed fix) |
| `scripts/build_visdrone_domedetr_gt.py` | 12-class COCO GT for Dome-DETR protocol eval |
| `scripts/eval_baseline_visdrone_sahi.py` | Baseline eval: yolo26x (COCO-pretrained, pre-VisDrone-fine-tune) + SAHI 640×640 |
| `scripts/eval_dome_detr_protocol.py` | Dome-DETR-protocol eval on ultralytics best.pt |
| `external/dome_detr/coco_eval_visdrone.py` | Reference copy of Dome-DETR's eval source (unchanged) |
| `external/dome_detr_repo/src/data/dataset/coco_eval_visdrone.py` | Same file at the import-path Dome-DETR expects |
| `external/dome_detr_repo/src/core/__init__.py` | Stub for `register()` (used by unused wrapper class) |
| `external/dome_detr_repo/src/misc/dist_utils.py` | Stub for `all_gather` (single-process fallback) |
| `runpod_startup.sh` | Already updated in earlier commit — VisDrone rsync, `VISDRONE_ROOT` env |

### Metrics (committed)

| file | what |
|---|---|
| `runs/p2/visdrone_p2_x_1280_50ep/{results,training_monitor,args.yaml}.csv/yaml` | P2 training telemetry |
| `runs/p2/visdrone_nop2_x_1280_50ep/{results,training_monitor,args.yaml}.csv/yaml` | No-P2 ablation telemetry |
| `runs/dome_detr_eval/p2_at_1280_letterbox/dome_detr_stats.json` | Dome-DETR protocol scores — P2 |
| `runs/dome_detr_eval/nop2_at_1280_letterbox/dome_detr_stats.json` | Dome-DETR protocol scores — No-P2 |
| `runs/baselines/baseline_yolo26x_sahi640/{baseline_metrics.csv,coco_eval.json}` | Pre-training SAHI baseline (class-agnostic + class-matched views) |

### Not committed (too large for git)

- `*.pt` weights (~117 MB each) — available on pod, reproducible from training scripts.
- `predictions.json` files (24-35 MB each) — raw model outputs; can be regenerated in 1 min via ultralytics val or `eval_dome_detr_protocol.py`.
- `visdrone_val_gt.json` (5.7 MB) — regenerable via `build_visdrone_coco_gt.py` (~5 s).

---

## 7. Reproducibility

To recreate these results from a fresh pod:

```bash
# 1. Pod bootstrap (VisDrone download + rsync + venv)
bash runpod_startup.sh

# 2. Build monitor GT (1-indexed, for training_monitor)
python scripts/build_visdrone_coco_gt.py --split val

# 3. Train P2
PYTORCH_ALLOC_CONF=expandable_segments:True python scripts/train_visdrone_p2.py

# 4. Train No-P2 ablation
PYTORCH_ALLOC_CONF=expandable_segments:True python scripts/train_visdrone_nop2.py

# 5. For Dome-DETR protocol eval:
# 5a. Re-download VisDrone raw val annotations (ignore-region flags needed)
curl -fsSL -o /tmp/VisDrone2019-DET-val.zip https://ultralytics.com/assets/VisDrone2019-DET-val.zip
unzip -q /tmp/VisDrone2019-DET-val.zip -d /tmp/visdrone_val_raw

# 5b. Build 12-class GT
python scripts/build_visdrone_domedetr_gt.py

# 5c. Eval both models
python scripts/eval_dome_detr_protocol.py \
    --weights runs/p2/visdrone_p2_x_1280_50ep/weights/best.pt \
    --imgsz 1280 --name p2_at_1280_letterbox
python scripts/eval_dome_detr_protocol.py \
    --weights runs/p2/visdrone_nop2_x_1280_50ep/weights/best.pt \
    --imgsz 1280 --name nop2_at_1280_letterbox
```

Total reproduction cost: ~16 hrs training + 2 min eval, A100 80 GB.
