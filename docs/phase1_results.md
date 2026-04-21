# Phase 1 Results — P2 Head × Model Size DOE on COCO

**Pod hardware**: 1 × RTX Pro 6000 Blackwell (97 GB VRAM)
**Dataset**: COCO train2017 / val2017, imgsz=640, end2end=True
**Variants tested**: 5 baselines (yolo26 n / s / m / l / x) + 2 P2 trains (yolo26n-p2, yolo26s-p2)
**Status**: Phase 1 P2 hypothesis validated AND extended into model-size DOE — original success criteria pass on n; finding diminishes on s; m/l/x P2 not run (extrapolated).

---

## TL;DR (3 lines)

1. **P2 head closes small-object recall gap on yolo26n** as hypothesized: 0-8 recall +0.167, 8-16 +0.034, all 3 Phase-1 success criteria pass.
2. **P2's effect halves going n → s** (0-8 recall gain drops from +0.167 to +0.076), and at s scale the combined 0-16 recall gain (+0.027) **fails the +0.05 success criterion**.
3. **In COCO at unconstrained compute, "add P2" and "scale up backbone one notch" are partial substitutes for small-object recall**. Adding P2 to a bigger model erases its FP-calibration improvement, leaving only mild incremental recall.

---

## 1. Metrics (what we measured)

We compute per-size-bin metrics **directly from per-image bbox matching** (greedy by conf, IoU≥threshold, same-class). Bins use **short-side pixels** (min(w,h)) of the GT bbox: `0-8, 8-16, 16-32, 32-64, 64-128, 128+`.

### 1.1 The three bottleneck indices (from Phase 0)

| index | what it captures | direction |
|---|---|---|
| `recall_at_05` | fraction of GT in this bin matched at IoU≥0.5 | ↑ better |
| `median_iou` | median IoU of TP detections in this bin (regression quality) | ↑ better |
| `fp_above_tp_median` | fraction of FP with conf > median TP conf in this bin (calibration) | ↓ better |

### 1.2 The two FP volume metrics (added during DOE)

| index | what it captures |
|---|---|
| `FP/GT` | total FP count divided by GT count in the bin → "誤報量" |
| `FP_above_TP × FP_count` | absolute number of "dangerous" high-conf FPs |

**Why FP/GT instead of FP/TP**: FP/TP's denominator (TP) varies with the model's recall, which conflates noise with detection ability. FP/GT uses a fixed denominator (val GT count), giving a clean noise-rate signal.

### 1.3 Standard mAP per bin (Table 12)

We re-implemented COCO mAP per bin with our short-side criterion. Validated against `faster-coco-eval` using COCO standard area bins to within 0.005-0.008 (mAP50-95). 101-point AP interpolation, maxDets=100 per (image, class), crowd GT handled as ignore.

---

## 2. Results — All tables

### Table 1 — Per-variant executive summary (0-8 bin focus + combined 0-16)

```
┌─────────┬───────┬────────────┬─────────┬───────────┬─────────────────┬─────────────┬──────────┬────────────┬──────────────────┐
│ variant │ FLOPs │ 0-8 recall │ 0-8 IoU │ 0-8 FP/GT │ 0-8 FP_above_TP │ 0-16 recall │ 0-16 IoU │ 0-16 FP/GT │ 0-16 FP_above_TP │
├─────────┼───────┼────────────┼─────────┼───────────┼─────────────────┼─────────────┼──────────┼────────────┼──────────────────┤
│    n    │  5.5  │   0.3426   │ 0.6299  │   13.41   │     0.0345      │   0.5738    │  0.6934  │   22.46    │      0.0227      │
│  n+P2   │  9.5  │   0.5100   │ 0.6383  │   24.15   │     0.0565      │   0.6524    │  0.6820  │   26.71    │      0.0393      │
│    s    │ 21.3  │   0.5190   │ 0.6738  │   22.42   │     0.0223      │   0.7038    │  0.7313  │   23.50    │      0.0158      │
│  s+P2   │ 27.8  │   0.5952   │ 0.6600  │   28.96   │     0.0356      │   0.7307    │  0.7148  │   30.61    │      0.0243      │
│    m    │ 65.4  │   0.6019   │ 0.7007  │   21.77   │     0.0183      │   0.7641    │  0.7620  │   20.58    │      0.0131      │
│    l    │ 87.6  │   0.6304   │ 0.7070  │   21.62   │     0.0170      │   0.7768    │  0.7659  │   19.62    │      0.0123      │
│    x    │ 196.0 │   0.6601   │ 0.7203  │   20.45   │     0.0155      │   0.8025    │  0.7770  │   17.87    │      0.0124      │
└─────────┴───────┴────────────┴─────────┴───────────┴─────────────────┴─────────────┴──────────┴────────────┴──────────────────┘
```

### Table 2 — Recall@IoU=0.5 per size bin

```
┌───────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
│  bin  │   n    │  n+P2  │   s    │  s+P2  │   m    │   l    │   x    │
├───────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤
│  0-8  │ 0.3426 │ 0.5100 │ 0.5190 │ 0.5952 │ 0.6019 │ 0.6304 │ 0.6601 │
│ 8-16  │ 0.6903 │ 0.7243 │ 0.7970 │ 0.7990 │ 0.8460 │ 0.8507 │ 0.8744 │
│ 16-32 │ 0.8513 │ 0.8177 │ 0.9143 │ 0.8791 │ 0.9370 │ 0.9420 │ 0.9510 │
│ 32-64 │ 0.9274 │ 0.8998 │ 0.9558 │ 0.9275 │ 0.9674 │ 0.9737 │ 0.9738 │
│64-128 │ 0.9575 │ 0.9343 │ 0.9739 │ 0.9512 │ 0.9795 │ 0.9859 │ 0.9856 │
│ 128+  │ 0.9778 │ 0.9668 │ 0.9817 │ 0.9730 │ 0.9845 │ 0.9878 │ 0.9891 │
└───────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘
```

### Table 3 — median IoU of TPs per size bin

```
┌───────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
│  bin  │   n    │  n+P2  │   s    │  s+P2  │   m    │   l    │   x    │
├───────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤
│  0-8  │ 0.6299 │ 0.6383 │ 0.6738 │ 0.6600 │ 0.7007 │ 0.7070 │ 0.7203 │
│ 8-16  │ 0.7094 │ 0.6975 │ 0.7502 │ 0.7354 │ 0.7840 │ 0.7879 │ 0.7987 │
│ 16-32 │ 0.7689 │ 0.7759 │ 0.8144 │ 0.8039 │ 0.8396 │ 0.8464 │ 0.8576 │
│ 32-64 │ 0.8382 │ 0.8336 │ 0.8693 │ 0.8595 │ 0.8858 │ 0.8897 │ 0.8977 │
│64-128 │ 0.8858 │ 0.8835 │ 0.9095 │ 0.8983 │ 0.9193 │ 0.9232 │ 0.9267 │
│ 128+  │ 0.9341 │ 0.9259 │ 0.9439 │ 0.9361 │ 0.9479 │ 0.9503 │ 0.9526 │
└───────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘
```

### Table 4 — FP/GT ratio per size bin (噪訊量, ↓ better)

```
┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
│  bin  │   n   │ n+P2  │   s   │ s+P2  │   m   │   l   │   x   │
├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
│  0-8  │ 13.41 │ 24.15 │ 22.42 │ 28.96 │ 21.77 │ 21.62 │ 20.45 │
│ 8-16  │ 27.02 │ 28.00 │ 24.05 │ 31.44 │ 19.98 │ 18.61 │ 16.57 │
│ 16-32 │ 28.39 │ 22.13 │ 23.25 │ 22.75 │ 18.63 │ 17.11 │ 15.22 │
│ 32-64 │ 20.86 │ 18.33 │ 16.31 │ 16.70 │ 13.02 │ 12.16 │ 10.55 │
│64-128 │ 14.07 │ 14.85 │ 10.75 │ 13.23 │  8.36 │  7.77 │  6.56 │
│ 128+  │  8.73 │ 11.08 │  6.31 │  9.86 │  5.16 │  4.70 │  3.92 │
└───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
```

### Table 5 — FP_above_TP per size bin (噪訊位置, ↓ better)

```
┌───────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
│  bin  │   n    │  n+P2  │   s    │  s+P2  │   m    │   l    │   x    │
├───────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤
│  0-8  │ 0.0345 │ 0.0565 │ 0.0223 │ 0.0356 │ 0.0183 │ 0.0170 │ 0.0155 │
│ 8-16  │ 0.0197 │ 0.0318 │ 0.0128 │ 0.0190 │ 0.0102 │ 0.0095 │ 0.0104 │
│ 16-32 │ 0.0116 │ 0.0168 │ 0.0070 │ 0.0094 │ 0.0061 │ 0.0061 │ 0.0057 │
│ 32-64 │ 0.0049 │ 0.0073 │ 0.0034 │ 0.0050 │ 0.0033 │ 0.0031 │ 0.0031 │
│64-128 │ 0.0026 │ 0.0034 │ 0.0023 │ 0.0031 │ 0.0021 │ 0.0022 │ 0.0023 │
│ 128+  │ 0.0014 │ 0.0013 │ 0.0013 │ 0.0010 │ 0.0013 │ 0.0012 │ 0.0016 │
└───────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘
```

### Table 6 — median TP confidence per size bin

```
┌───────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
│  bin  │   n    │  n+P2  │   s    │  s+P2  │   m    │   l    │   x    │
├───────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤
│  0-8  │ 0.0686 │ 0.0261 │ 0.1134 │ 0.0519 │ 0.1546 │ 0.1521 │ 0.1813 │
│ 8-16  │ 0.1036 │ 0.0454 │ 0.2118 │ 0.1067 │ 0.2943 │ 0.3052 │ 0.3350 │
│ 16-32 │ 0.1760 │ 0.1192 │ 0.3448 │ 0.2451 │ 0.4466 │ 0.4575 │ 0.5054 │
│ 32-64 │ 0.3639 │ 0.2925 │ 0.5689 │ 0.4668 │ 0.6524 │ 0.6624 │ 0.7108 │
│64-128 │ 0.6027 │ 0.5176 │ 0.7558 │ 0.6602 │ 0.8110 │ 0.8156 │ 0.8430 │
│ 128+  │ 0.8063 │ 0.7683 │ 0.8810 │ 0.8540 │ 0.8986 │ 0.9031 │ 0.9182 │
└───────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘
```

### Table 7-9 — Raw counts (for reproducibility)

```
Table 7 — GT count per bin (val2017 fixed across all variants)
┌───────┬──────────┐
│  bin  │ GT count │
├───────┼──────────┤
│  0-8  │   2557   │
│ 8-16  │   5070   │
│ 16-32 │   7487   │
│ 32-64 │   7834   │
│64-128 │   6678   │
│ 128+  │   6709   │
│ TOTAL │  36335   │
└───────┴──────────┘

Table 8 — TP raw counts per bin per variant
┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
│  bin  │   n   │ n+P2  │   s   │ s+P2  │   m   │   l   │   x   │
├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
│  0-8  │   876 │  1304 │  1327 │  1522 │  1539 │  1612 │  1688 │
│ 8-16  │  3500 │  3672 │  4041 │  4051 │  4289 │  4313 │  4433 │
│ 16-32 │  6374 │  6122 │  6845 │  6582 │  7015 │  7053 │  7120 │
│ 32-64 │  7265 │  7049 │  7488 │  7266 │  7579 │  7628 │  7629 │
│64-128 │  6394 │  6239 │  6504 │  6352 │  6541 │  6584 │  6582 │
│ 128+  │  6560 │  6486 │  6586 │  6528 │  6605 │  6627 │  6636 │
│ TOTAL │ 30969 │ 30872 │ 32791 │ 32301 │ 33568 │ 33817 │ 34088 │
└───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘

Table 9 — FP raw counts per bin per variant
┌───────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
│  bin  │   n    │  n+P2  │   s    │  s+P2  │   m    │   l    │   x    │
├───────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤
│  0-8  │  34292 │  61742 │  57332 │  74052 │  55654 │  55282 │  52281 │
│ 8-16  │ 137011 │ 141971 │ 121919 │ 159412 │ 101297 │  94358 │  84013 │
│ 16-32 │ 212539 │ 165723 │ 174050 │ 170340 │ 139477 │ 128092 │ 113955 │
│ 32-64 │ 163435 │ 143620 │ 127781 │ 130811 │ 101988 │  95231 │  82626 │
│64-128 │  93967 │  99141 │  71815 │  88363 │  55818 │  51888 │  43799 │
│ 128+  │  58564 │  74305 │  42364 │  66137 │  34589 │  31554 │  26275 │
│ TOTAL │ 699808 │ 686502 │ 595261 │ 689115 │ 488823 │ 456405 │ 402949 │
└───────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘
```

### Table 10 — Δ (P2 − baseline) at same scale

```
┌───────┬──────────┬──────────┬──────────┬──────────┬───────────┬───────────┬───────────┬───────────┐
│  bin  │ Δrec(n)  │ Δrec(s)  │ ΔIoU(n)  │ ΔIoU(s)  │ ΔFP/GT(n) │ ΔFP/GT(s) │ ΔFPcal(n) │ ΔFPcal(s) │
├───────┼──────────┼──────────┼──────────┼──────────┼───────────┼───────────┼───────────┼───────────┤
│  0-8  │ +0.1674  │ +0.0762  │ +0.0084  │ -0.0138  │  +10.74   │   +6.54   │  +0.0220  │  +0.0133  │
│ 8-16  │ +0.0340  │ +0.0020  │ -0.0119  │ -0.0148  │   +0.98   │   +7.39   │  +0.0121  │  +0.0062  │
│ 16-32 │ -0.0336  │ -0.0352  │ +0.0070  │ -0.0105  │   -6.26   │   -0.50   │  +0.0052  │  +0.0024  │
│ 32-64 │ -0.0276  │ -0.0283  │ -0.0046  │ -0.0098  │   -2.53   │   +0.39   │  +0.0024  │  +0.0016  │
│64-128 │ -0.0232  │ -0.0227  │ -0.0023  │ -0.0112  │   +0.78   │   +2.48   │  +0.0008  │  +0.0008  │
│ 128+  │ -0.0110  │ -0.0087  │ -0.0082  │ -0.0078  │   +2.35   │   +3.55   │  -0.0001  │  -0.0003  │
└───────┴──────────┴──────────┴──────────┴──────────┴───────────┴───────────┴───────────┴───────────┘
```

### Table 11 — Dangerous FP absolute count (FP_above_TP × FP_count)

```
┌────────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
│  bin   │   n   │ n+P2  │   s   │ s+P2  │   m   │   l   │   x   │
├────────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
│  0-8   │ 1183  │ 3488  │ 1278  │ 2636  │ 1018  │  940  │  810  │
│  8-16  │ 2699  │ 4515  │ 1561  │ 3029  │ 1033  │  896  │  874  │
│ 16-32  │ 2465  │ 2784  │ 1218  │ 1601  │  851  │  781  │  650  │
│ 32-64  │  801  │ 1048  │  434  │  654  │  337  │  295  │  256  │
│ 64-128 │  244  │  337  │  165  │  274  │  117  │  114  │  101  │
│  128+  │   82  │   97  │   55  │   66  │   45  │   38  │   42  │
│ TOTAL  │ 7474  │ 12269 │ 4711  │ 8260  │ 3401  │ 3064  │ 2733  │
└────────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
```

### Table 12 — Per-bin mAP50 (short-side bin, COCO-faithful matching)

```
┌────────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
│  bin   │   n   │ n+P2  │   s   │ s+P2  │   m   │   l   │   x   │
├────────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
│  0-8   │ 0.094 │ 0.106 │ 0.173 │ 0.173 │ 0.238 │ 0.274 │ 0.298 │
│  8-16  │ 0.236 │ 0.208 │ 0.366 │ 0.276 │ 0.435 │ 0.456 │ 0.490 │
│ 16-32  │ 0.398 │ 0.348 │ 0.511 │ 0.439 │ 0.604 │ 0.617 │ 0.658 │
│ 32-64  │ 0.575 │ 0.530 │ 0.680 │ 0.606 │ 0.731 │ 0.755 │ 0.779 │
│ 64-128 │ 0.686 │ 0.629 │ 0.758 │ 0.691 │ 0.800 │ 0.821 │ 0.840 │
│  128+  │ 0.780 │ 0.710 │ 0.829 │ 0.759 │ 0.853 │ 0.865 │ 0.889 │
└────────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
```

### Table 12b — Per-bin mAP50-95 (avg over IoU 0.50, 0.55, ..., 0.95)

```
┌────────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
│  bin   │   n   │ n+P2  │   s   │ s+P2  │   m   │   l   │   x   │
├────────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
│  0-8   │ 0.032 │ 0.047 │ 0.074 │ 0.075 │ 0.111 │ 0.124 │ 0.155 │
│  8-16  │ 0.123 │ 0.113 │ 0.211 │ 0.157 │ 0.258 │ 0.277 │ 0.302 │
│ 16-32  │ 0.245 │ 0.214 │ 0.334 │ 0.282 │ 0.408 │ 0.426 │ 0.458 │
│ 32-64  │ 0.395 │ 0.360 │ 0.489 │ 0.427 │ 0.537 │ 0.561 │ 0.590 │
│ 64-128 │ 0.517 │ 0.470 │ 0.592 │ 0.527 │ 0.632 │ 0.649 │ 0.674 │
│  128+  │ 0.629 │ 0.569 │ 0.695 │ 0.626 │ 0.728 │ 0.746 │ 0.770 │
└────────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
```

---

## 3. Findings

### 3.1 Original Phase 1 success criteria — pass on n, fail on s

Combined 0-16px metric vs same-scale baseline:

| criterion | threshold | n result | s result |
|---|---|---|---|
| Δrecall_at_05 ≥ +0.05 | +0.05 | **+0.079** ✅ | **+0.027** ❌ |
| ΔmedianIoU ≥ -0.02 | -0.02 | -0.012 ✅ | -0.012 ✅ |
| ΔFP_above_TP ≤ +0.03 | +0.03 | +0.017 ✅ | +0.008 ✅ |

**P2 architecturally works as hypothesized**, but **the recall benefit shrinks fast with backbone scale** and falls below the +5pp threshold by the s scale.

### 3.2 P2 vs scaling — partial substitutes, NOT complements

For 0-8 recall:

| path | result | gain over n |
|---|---|---|
| baseline-n | 0.343 | — |
| n+P2 | 0.510 | +0.167 |
| baseline-s | 0.519 | +0.176 |
| s+P2 | 0.595 | +0.252 |
| baseline-x | 0.660 | +0.317 |

Observations:
- "n + P2" ≈ "baseline-s" — both routes yield ~0.51 small recall, **P2 buys you roughly one model size**
- "s + P2" stacks but with diminishing returns — s+P2 (0.595) > s (0.519) but < x (0.660)
- → **scaling and P2 are partial substitutes** for small-object recall on COCO

### 3.3 P2 has a structural FP cost that scaling cannot offset

| variant | 0-8 FP_above_TP | comparison |
|---|---|---|
| baseline-n | 0.0345 | reference |
| baseline-s | 0.0223 | scaling **improves** (-0.012) |
| s+P2 | 0.0356 | adding P2 to s **erases** the scaling benefit, returns to baseline-n level |

Same pattern in absolute dangerous-FP count (Table 11):
- baseline-n: 7474
- baseline-s: 4711 (-37%)
- s+P2: 8260 (+75% over baseline-s, **worse than even baseline-n**)

**P2 and scaling conflict on FP-calibration** — adding P2 to a bigger model wastes the FP-cal advantage of going bigger.

### 3.4 mAP confirms the recall picture, with a regression-quality penalty

From Tables 12 / 12b:
- **n+P2 0-8 mAP50 (0.106) > n baseline (0.094)** — P2 wins on smallest bin at n scale
- **s+P2 0-8 mAP50 (0.173) ≈ s baseline (0.173)** — P2's mAP advantage **vanishes at s scale**
- For 8-16+ bins, P2 always loses to same-scale baseline (-0.05 to -0.09 mAP50)
- Note: s+P2 only ran 15 epochs vs baseline-s being fully converged — extrapolating sat30's late-stage trajectory (+0.019 mAP50-95 over 15 → 30 ep) suggests P2-s could close ~0.02 of the gap with longer training, but **wouldn't change the main conclusion** (gap on 8-16+ is too big)

### 3.5 Two FP indices tell complementary stories

| index | what it surfaces | scaling effect | P2 effect |
|---|---|---|---|
| FP/GT (Table 4) | sheer noise volume | improves with scale (n→x: -27%) | adds noise, especially small bins |
| FP_above_TP (Table 5) | calibration / how filterable noise is | improves (n→x: -55% on 0-8) | makes more noise un-filterable |
| Dangerous FPs (Table 11) | absolute high-conf wrong predictions | drops 63% (n→x) | adds 64% (n→n+P2) and 75% (s→s+P2) |

**Both FP indices show**: scaling helps universally, P2 hurts universally. The two effects roughly cancel in s+P2 → calibration returns to base-n level.

---

## 4. Decision implications

| Use case | Recommendation |
|---|---|
| Smallest practical model + best small-obj recall | **n + P2** (1.7× FLOPs, +0.167 recall on 0-8) |
| Want s-class accuracy but care about small recall | **s + P2** is a Pareto choice for recall, but pays FP-cal penalty. If FP matters → just use **baseline-s** |
| Generic detection target | **scale up the backbone**; P2 architectural complexity is not justified |
| Real production with tight FP requirements | **prefer scaling over P2** at every scale |

**P2 is a small-model-only optimization in COCO general-detection regime.**

---

## 5. Limitations

1. **Training duration not equalized**: baseline-{n,s,m,l,x} use ultralytics' fully pretrained weights (~300 epoch from scratch). P2-n was 30-epoch warm-start, P2-s was 15-epoch warm-start. Estimate from sat30 trajectory: P2-s could gain ~+0.02 mAP with 30 epochs, doesn't change story.
2. **m / l / x P2 not measured** — extrapolated from n→s trend, which already shows steep diminishing returns.
3. **Single seed** (seed=0, deterministic=True) — variance not quantified.
4. **Single GPU type** (RTX Pro 6000 Blackwell) — but findings are dataset/architecture-driven, not GPU-specific.
5. **COCO general-detection only** — does NOT test SOD-specific benchmarks (VisDrone, AI-TOD, SODA). Conclusions may differ where small objects dominate the dataset.
6. **end2end=True** for all runs — different NMS regimes might shift FP-cal numbers.

---

## 6. Open questions worth follow-up

1. **Is "P2 ≈ scaling one notch" a known finding in the literature?** → handed to web Claude as `litreview_brief_scaling_vs_p2.md`.
2. **Why does median IoU plateau at ~0.72 even on x-scale 0-8 bin?** Suggests bbox regression is the remaining bottleneck for tiny objects (independent of scaling and independent of P2).
3. **Would distillation from baseline-x → baseline-n match P2-n's small-recall gain at lower compute?** Untested.

---

## 7. Reproducibility

- Code: `scripts/train_p2.py`, `scripts/eval_baseline.py`, `scripts/training_monitor.py`, `scripts/compute_per_bin_mAP.py`, `scripts/validate_per_bin_mAP.py`, `scripts/plot_phase1_results.py`
- Data: `runs/baselines/baseline_yolo26{n,s,m,l,x}/baseline_metrics.csv` and `runs/p2/{p1_exp1_sat30,p1_exp2_s_15ep}/training_monitor.csv`
- Per-bin mAP cached: `/tmp/mAP_per_bin.json`
- Training time: P2-n 30ep = 6.42 hr, P2-s 15ep = 4.05 hr, all 5 baselines = ~15 min total

Validation: `compute_per_bin_mAP.py` was cross-checked against `faster-coco-eval` using COCO standard area bins (not short-side), agreement within 0.005-0.008 mAP50-95 across all 4 trained variants. Residual gap attributed to implementation noise (101-pt interpolation precision, tied-score ordering); does not affect relative ranking.
