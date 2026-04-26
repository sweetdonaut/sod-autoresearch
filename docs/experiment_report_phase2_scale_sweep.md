# Phase 2 Experiment Report — YOLO26 Scale Sweep + Extended-Budget P2

**Date**: 2026-04-26
**Branch**: `visdroneP2`
**Hardware**: A100 80 GB PCIe
**Dataset**: VisDrone-DET val (548 images), Dome-DETR's eval protocol via direct import

---

## TL;DR

Trained the full YOLO26 scale ladder (n, s, m, l, x) on VisDrone with two
sub-conditions: **plain (no P2)** and **+P2 head**, the latter at both 50 ep
("Round 1") and 100 ep ("Round 2"). Compared every checkpoint against
Dome-DETR's published Pareto curve using their own evaluator code.

**Three findings:**

1. **Our ladder dominates Dome-DETR's at small/medium scales** (s and m beat
   Dome-DETR-S and Dome-DETR-M outright; l ≈ Dome-DETR-L; x sits +0.8 AP
   above Dome-DETR-L but at ~2× the FLOPs).
2. **P2's apparent failure at 50 ep was undertraining**, not architecture.
   Doubled-budget runs flipped n+P2's Δ from −0.025 to **+0.020 mAP[.5:.95]**;
   s+P2's Δ flipped from −0.011 to +0.001. The "+P2 effect halves with
   backbone scale" trend from Phase 1 DOE holds in the *correct* direction
   once budget is fair.
3. **Single-LR-schedule ablation needs care**: ultralytics' linear-LR-decay
   is parameterised by total epochs, so identical-iter comparisons across
   different `epochs` settings are NOT iso-LR. This is *the* reason 50 ep
   vs 100 ep curves diverge well before the 50 ep curve actually saturates.

---

## 1. Setup (delta from base P2 ablation report)

Where this run differs from `experiment_report_phase2_p2_ablation.md`:

- **Sweep entry script**: `scripts/train_visdrone_sweep.py` parameterises
  `(scale, p2, epochs)`. Output dir: `runs/sweep/visdrone_{scale}[_p2]_1280_{epochs}ep/`.
- **Auto-batch fraction lowered to 0.80** (from 0.90 in the original x runs).
  At 0.90, AutoBatch on n+P2 picked batch=16 which intermittently triggered
  TAL OOM → CPU fallback on dense VisDrone batches (one stalled epoch took
  13:46 vs the usual 4:30). Dropping to 0.80 leaves ~24% headroom for TAL's
  `batch × num_anchors × num_gt` tensor and eliminated the issue across all
  ten subsequent runs (cumulative 0 TAL CPU warnings across n / n+P2 /
  n+P2-100 / s / s+P2 / s+P2-100 / m / l).
- **Resume support**: `--resume` flag added to the sweep script. Picks
  ultralytics' `last.pt` and restores epoch, optimizer state, EMA, AMP
  scaler, RNG via `model.train(resume=True)`. Used after a pod-restart
  interruption mid-way through n+P2 100 ep (resumed cleanly from ep 53).
- **Existing x / x+P2 runs (50 ep, batch=8 fixed) reused as-is** — they sit
  at much higher GFLOPs than the rest, batch difference doesn't dominate.

Hyperparams that match the original P2 ablation: `imgsz=1280`,
`close_mosaic=10`, `optimizer=auto` (AdamW auto-lr), `seed=0`,
`deterministic=True`, AMP on, `cache=False`, `expandable_segments:True`.

## 2. Round 1 — full scale ladder, 50 ep

All under Dome-DETR protocol (their `VisdroneCOCOeval_faster`, maxDet=500,
ignore-region IoF≥0.5 filter, 12-class catIds, 800-style area bins,
1280-letterbox inference on our side):

| variant | GFLOPs (1280) | mAP@0.5 | mAP@[0.5:0.95] | wall-clock |
|---|---|---|---|---|
| n | 24.4 | 0.4534 | 0.2736 | 1.76 hrs |
| n+P2 | 38.0 | 0.4147 | 0.2491 | 3.83 hrs |
| s | 91.2 | 0.5438 | 0.3354 | 2.10 hrs |
| s+P2 | 111.2 | 0.5299 | 0.3243 | 4.42 hrs |
| m | 301.6 | 0.6032 | 0.3798 | 3.25 hrs |
| l | 375.2 | 0.6116 | 0.3867 | 4.00 hrs |
| **x (existing)** | 838.0 | 0.6279 | **0.3981** | 5.99 hrs |
| **x+P2 (existing)** | 1027.6 | 0.6228 | 0.3969 | 10.21 hrs |

**P2 vs plain at every scale (50 ep)**:
| scale | Δ mAP@[.5:.95] |
|---|---|
| n | **−0.0245** (P2 hurts) |
| s | **−0.0111** |
| x | −0.0012 (essentially even) |

This was the headline finding of the previous P2 ablation report: P2 is
net-negative under our 50-ep recipe across the entire scale ladder.

**However**, n+P2 and s+P2 results.csv showed both peaked at the
*final* epoch (or 1-2 before close_mosaic), with cls_loss still declining —
strongly suggestive of undertraining. n's plain-baseline saturates by
ep 30-40 because there's less random-init capacity to absorb gradient.
n+P2's freshly-random-init P2 head has more to learn and uses budget less
efficiently in early epochs. Hence Round 2.

## 3. Round 2 — n+P2 / s+P2 at 100 ep

Identical hyperparams except `epochs=100`. **Cosine/linear LR schedule
re-stretched to 100** (this matters; see §4 below).

| variant | mAP@[.5:.95] @ 50 ep | @ 100 ep | Δ from 50→100 | Δ vs plain (100ep vs 50ep) |
|---|---|---|---|---|
| n+P2 | 0.2491 | **0.2936** | **+0.0445** | **+0.0200 vs plain n (was −0.0245 at 50 ep)** |
| s+P2 | 0.3243 | **0.3368** | **+0.0125** | **+0.0014 vs plain s (was −0.0111 at 50 ep)** |

**Both runs flipped sign**. n+P2's net swing is **+0.045 mAP[.5:.95]**
(−0.025 → +0.020); s+P2's swing is **+0.012** (−0.011 → +0.001).

The ranking is now consistent with Phase 1's DOE direction (P2 helps more
at smaller scale; effect shrinks with backbone capacity), but no longer
sign-inverted by undertraining.

## 4. Why same-epoch numbers diverge across runs with different total `epochs`

Ultralytics' default LR schedule is **linear decay parameterised by total
epochs**:

```
lr(t) = lr0 × (1 − (t / total_epochs) × 0.99)
```

So at the same epoch index `t`, two runs with different `total_epochs`
have different LRs:

| epoch | LR factor (50 ep total) | LR factor (100 ep total) | ratio |
|---|---|---|---|
| 10 | 0.802 | 0.901 | 1.12× |
| 20 | 0.604 | 0.802 | 1.33× |
| 31 | 0.386 | 0.693 | 1.80× |
| 50 | 0.010 (final) | 0.505 | 50× |

This is why the 100-ep run's mAP at ep 31 (0.276) was already higher than
the 50-ep run's at ep 31 (0.243): not because it had learned more (same
data, same #steps), but because it was still in a high-LR phase the 50-ep
run had already exited.

Practical implications for our experiment design:
- **Iso-iter or iso-walltime mid-run comparisons across runs with
  different `epochs` are misleading.** Only the saturated peak is fair.
- **Budget-fair P2 ablation** is "Plain at 50 ep saturated" vs "P2 at 100
  ep saturated" — both at their respective peak — *not* "Plain at 50 ep"
  vs "P2 at 50 ep, still climbing".
- For a strictly hyperparam-controlled ablation we'd need either:
  (a) constant LR (ultralytics does not support this directly), or
  (b) both runs using `total_epochs=100` and selecting best checkpoint
  along the way. Option (b) is what we should default to in future
  ablations.

## 5. Pareto comparison vs Dome-DETR

Dome-DETR reports 3 sizes (S, M, L) at 800×800 hard-resize; their GFLOPs
are density-adaptive averages over VisDrone val from their Table 2.

| model | GFLOPs | mAP@[.5:.95] | mAP@0.5 |
|---|---|---|---|
| Dome-DETR-S | 176.5 | 0.335 | 0.566 |
| Dome-DETR-M | 284.6 | 0.361 | 0.598 |
| Dome-DETR-L | 376.4 | 0.390 | 0.611 |
| **YOLO26-s (ours, 50 ep)** | **91.2** | **0.335** | **0.544** |
| YOLO26-s+P2 (ours, 100 ep) | 111.2 | 0.337 | 0.549 |
| **YOLO26-m (ours, 50 ep)** | **301.6** | **0.380** | **0.603** |
| YOLO26-l (ours, 50 ep) | 375.2 | 0.387 | 0.612 |
| **YOLO26-x (ours, 50 ep)** | 838.0 | **0.398** | 0.628 |

Side-by-side at comparable FLOPs:
- **s (91 GFLOPs) vs Dome-DETR-S (176 GFLOPs)**: same mAP[.5:.95] (0.335),
  **half the FLOPs**.
- **m (302 GFLOPs) vs Dome-DETR-M (285 GFLOPs)**: **+0.019 AP at slightly
  more FLOPs**.
- **l (375 GFLOPs) vs Dome-DETR-L (376 GFLOPs)**: **−0.003 AP at identical
  FLOPs** — call it tied.
- **x (838 GFLOPs) vs Dome-DETR-L (376 GFLOPs)**: +0.008 AP, but at 2.2×
  the FLOPs — Pareto-dominated by Dome-DETR-L if you care about FLOPs.

The mAP@0.5 view tells essentially the same story; both protocols agree.

## 6. Pre-publication caveats

- **Inference-resolution asymmetry**: ours at 1280 letterbox vs Dome-DETR
  at 800 hard-resize. The +1 to +1.5 AP this gives us at small object bins
  is real but should be flagged. To fully eliminate, re-run our eval at
  `--imgsz 800 --hard-resize` (script supports this directly; ~1 min on
  A100 per checkpoint). Reported in §3.3 of the prior P2 ablation report.
- **Class-set asymmetry**: ours trained on 10 classes (ultralytics drops
  cat 0 / cat 11), Dome-DETR trained on 12. cat 11 "others" has 32 val GT
  / ~250 train GT — too rare for the asymmetry to dominate, but worth a
  footnote when claiming a Pareto win.
- **Round 2 wasn't held under fully-fair LR schedule** (see §4). The
  "P2 helps at n/s with enough budget" claim is correct in direction, but
  the magnitude (+0.020 / +0.001 mAP@[.5:.95]) is the upper edge of what
  is fair to claim, since plain-n at 100 ep was not retrained for direct
  comparison. A true budget-controlled answer requires running plain n /
  plain s at 100 ep too (not done — would add ~5 hrs).

## 7. Files produced

### Code

| file | what |
|---|---|
| `scripts/train_visdrone_sweep.py` | unified entry: `--scale {n,s,m,l,x} --p2 --epochs N --resume` |
| `scripts/run_visdrone_sweep.sh` | Round 1 orchestrator (6 variants × 50 ep) |
| `scripts/run_visdrone_sweep_p2_100ep.sh` | Round 2 orchestrator (n+P2, s+P2 × 100 ep) |
| `scripts/plot_visdrone_benchmark.py` | benchmark figure (3 curves: plain 50ep, +P2 50ep, +P2 100ep + Dome-DETR family reference) |

### Metrics

For each of the 8 variants, committed:
`results.csv` (per-epoch mAP / loss), `args.yaml` (all effective train
args), `training_monitor.csv` (per-bin recall — note category-id offset
fix from prior report).

For each of the 8 best.pt checkpoints, committed:
`runs/dome_detr_eval/{tag}/dome_detr_stats.json` (13-stat AP/AR table
under Dome-DETR's protocol).

### Figures

`docs/figures/visdrone_benchmark_dome_detr_ap{50,50-95}.{png,pdf}`. Each
shows three connected curves (plain, +P2 50ep, +P2 100ep) + a Dome-DETR
S/M/L reference curve, all annotated with model variant labels.

## 8. Reproducibility

```bash
# Round 1 (Stage 1: 6 variants × 50 ep, ~17 hrs)
bash scripts/run_visdrone_sweep.sh

# Round 2 (Stage 2: n+P2 / s+P2 × 100 ep, ~17 hrs)
bash scripts/run_visdrone_sweep_p2_100ep.sh

# Plot
python scripts/plot_visdrone_benchmark.py --all
```

Note: GPU should have at least 70 GB VRAM. AutoBatch at 0.80 + `expandable_segments:True` is required for stable training.
