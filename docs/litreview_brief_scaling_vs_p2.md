# Lit Review Brief — Model Scaling vs P2 / SOD Architectural Tricks

**Audience**: Web Claude / deep-search agent.
**Goal**: Verify whether the empirical finding below is novel, known, or disputed.

---

## TL;DR — what I want you to find

Concrete answer to:

> **"Has anyone done iso-compute / iso-latency comparisons of model scaling vs P2 head (or other SOD architectural tricks) for small-object detection?"**

If yes → cite them, summarize their conclusion.
If no → confirm the gap and note adjacent work.

---

## My empirical finding (COCO, YOLO26, RTX Pro 6000)

I trained YOLO26 with an extra P2 detection head (stride 4 added to default P3/P4/P5) and compared against same-family baselines on COCO val2017 at imgsz=640.

**Per-bin recall@IoU=0.5** (object short-side in pixels):

| bin | base-n | P2-n (30 ep) | base-s | P2-s (15 ep) |
|---|---|---|---|---|
| 0-8 | 0.343 | **0.510** (+0.167) | 0.519 | **0.595** (+0.076) |
| 8-16 | 0.690 | 0.724 (+0.034) | 0.797 | 0.799 (+0.002) |
| 16-32 | 0.851 | **0.818** (-0.034) | 0.914 | **0.879** (-0.035) |
| 32-64 | 0.927 | 0.900 (-0.028) | 0.956 | 0.928 (-0.028) |
| 64-128 | 0.958 | 0.934 (-0.023) | 0.974 | 0.951 (-0.023) |
| 128+ | 0.978 | 0.967 (-0.011) | 0.982 | 0.973 (-0.009) |

**Per-bin baseline FP-calibration (`fp_above_tp_median` — fraction of FPs ranked above the median TP confidence; lower = better):**

| bin | base-n | base-s | base-m | base-l | base-x |
|---|---|---|---|---|---|
| 0-8 | 0.034 | 0.022 | 0.018 | 0.017 | 0.016 |
| 8-16 | 0.020 | 0.013 | 0.010 | 0.010 | 0.010 |

**Three observations**:

1. **P2 effect halves going n → s**: 0-8 recall gain drops from +0.167 to +0.076. P2's effect appears to diminish with backbone scale.
2. **"P2 + n" ≈ "baseline s"**: both reach ~0.51 on 0-8 recall. → P2 and "go up one size" are **partial substitutes**, not complements, on COCO.
3. **FP-calibration improves with model size, but P2 hurts it at every scale**: P2-s FP-cal returns to roughly base-n level — adding P2 to a bigger model erases the FP-cal benefit of going bigger.

**Hypothesis**: In COCO general-detection regime, with no compute budget constraint, **adding P2 is no better than scaling up the backbone**. Architectural SOD tricks may be over-prescribed when fair-compute comparisons aren't done.

---

## Sub-questions

### 1. Iso-compute / iso-latency comparisons

- Has any paper specifically compared "tiny model + P2/FPN trick" vs "next-larger model with no trick" at matched FLOPs / latency / params?
- Specifically for the **YOLO family** (v5/v8/v9/v10/v11/v12) — the de facto SOD baseline. Any ablation across (size × P2-head)?

### 2. Diminishing returns of architectural tricks at scale

- Are there published curves showing P2 / BiFPN / DyHead gains shrinking as backbone scale grows?
- Any "scaling law" style work for object detection (analog to Kaplan/Chinchilla for LM)?

### 3. Substitution between scaling and SOD-specific design

- Has anyone framed model scale and SOD architectural tricks as substitutes vs complements explicitly?
- Multi-scale ablation tables in recent SOD papers — do they include scale as a controlled variable?

### 4. Dataset-distribution effects

- COCO has many large objects. SOD-specific benchmarks (VisDrone-DET, AI-TOD, SODA-D, SODA-A, TinyPerson) are dominated by tiny objects.
- Do P2 / SOD-trick gains hold UP at larger backbone sizes on these SOD-only benchmarks?
- If yes, my finding may be COCO-specific. If no, it generalizes.

### 5. Counter-evidence

- Papers explicitly claiming "P2 is essential at all scales" — at what backbone size do they evaluate?
- Papers claiming "scaling alone is enough" — anyone bold enough to publish that?

---

## Specific things to search

### Authors / venues likely to have data points

- Ultralytics issues / blog posts on YOLO scaling
- Aerial / drone detection groups (DOTA, VisDrone organizers)
- Joseph Redmon-style ablation tables
- arXiv 2024-2026 small-object detection surveys

### Keywords

- `"small object detection" "ablation" "model size"`
- `"P2 head" OR "stride 4 detection head" YOLO`
- `"compute-optimal" detection`
- `"iso-FLOPs" OR "iso-latency" detection`
- `"scaling law" object detection`
- `"high-resolution feature pyramid" ablation scale`
- `tiny object detection benchmark backbone size`

### Specific benchmarks to spot-check

- **VisDrone-DET**: SOTA leaderboard — what's the typical model size + tricks?
- **AI-TOD**: tiny object benchmark — same question
- **SODA-D / SODA-A**: 2023/2024 SOD benchmarks
- **COCO mAP_small**: top 20 entries on PaperWithCode — model size distribution + tricks

### Specific papers to check (starting points)

- "**A Survey of Modern Deep Learning based Object Detection Models**" — Sultana et al.
- "**Towards Large-Scale Small Object Detection: Survey and Benchmarks**" (SODA paper) — Cheng et al. 2023
- Any **YOLOv9/v10/v11/v12** paper with scale × architecture ablation
- "**TPH-YOLOv5**" / "**QueryDet**" — classic SOD architectural papers
- **DyHead**, **BiFPN**, **NWD** — alternative SOD tricks; check if they were ablated against scaling

---

## Output format I want back

Per finding:

```
- [Citation] (year, venue)
  - 1-2 sentence summary of what they did
  - Relation to my hypothesis: SUPPORTS / CONTRADICTS / ORTHOGONAL
  - If CONTRADICTS: which specific number / table contradicts which of my 3 observations
```

Followed by:

**Synthesis (≤ 200 words)**:
- Is "P2 and model size are partial substitutes for small-object recall on COCO" a novel claim, well-known, or already disputed?
- What's the closest prior work?
- If novel and supportable → what's the cleanest way to publish (NeurIPS / arXiv only / workshop)?
- If not novel → which work scoops it most directly?

---

## Background on the experiment (context for evaluation)

- **Hardware**: 1 × RTX Pro 6000 Blackwell (97 GB VRAM)
- **Dataset**: COCO train2017 / val2017, 640x640 imgsz
- **Architecture**: YOLO26 (Ultralytics 8.4.36), `end2end=True`
- **P2 config**: stride-4 detection head added to default P3/P4/P5 (architectural diff in `configs/yolo26-p2.yaml`)
- **Training**: 30 ep for n, 15 ep for s (s saturated by ep 12); deterministic=True; AMP fp16; pretrained backbone warm-start, P2 head random init
- **Phase 1 success criterion** (combined 0-16px recall): +0.05 absolute over same-scale baseline
  - n: passed (+0.079)
  - s: failed (+0.027)
- Did **NOT** train P2 on m / l / x — extrapolated based on n→s trend.
