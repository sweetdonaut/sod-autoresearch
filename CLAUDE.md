# SOD-AutoResearch — Phase 1

**Goal of this repo**: Run a P2-head training experiment on COCO to validate it
addresses the small-object **Recall bottleneck** identified in Phase 0.

This CLAUDE.md is written for a **fresh Claude Code session on a RunPod pod**.
Read `docs/research_plan_phase1.md` for the full hypothesis, design, and success
criteria before making changes. **Every minute the pod is idle costs money — your
job is to execute, not explore.**

---

## Pod environment assumptions

| What | Where | Notes |
|---|---|---|
| COCO dataset | `/workspace/coco/` | On mounted network volume; **already prepared**, do NOT re-fetch |
| images | `/workspace/coco/images/{train,val}2017/` | 118K + 5K jpg |
| COCO JSON | `/workspace/coco/annotations/instances_{train,val}2017.json` | raw GT |
| YOLO labels | `/workspace/coco/labels/{train,val}2017/` | 117K + 5K txt, pre-converted |
| Python env | put on **pod root disk** `/root/sod-env` | network volume has 2x MooseFS replication = pay double, avoid for env |
| Pretrained weights | ultralytics auto-downloads on first use | goes to cwd by default |

---

## First-time pod setup (5 min)

Run once per pod instance:

```bash
# 1. Env on pod root (NOT on /workspace — avoids 2x billing)
cd /root
uv venv sod-env
source /root/sod-env/bin/activate
uv pip install ultralytics

# 2. Point ultralytics at the prepared COCO
mkdir -p /root/datasets
ln -sfn /workspace/coco /root/datasets/coco

# 3. Clone this repo (anywhere on volume is fine)
cd /workspace
git clone <repo-url> sod-autoresearch
cd sod-autoresearch

# 4. Sanity check
python -c "from ultralytics import YOLO; import os; \
  assert os.path.isdir('/workspace/coco/images/train2017'); \
  assert os.path.isdir('/workspace/coco/labels/train2017'); \
  print('COCO OK')"
```

---

## Execute Phase 1 (two commands)

**Step A — Baseline snapshot** (~3 min): establishes yolo26n's per-size-bin
metrics so P2 has something to beat.

```bash
python scripts/eval_baseline.py --model yolo26n.pt --name baseline_yolo26n
```

Output: `runs/baselines/baseline_yolo26n/baseline_metrics.csv`

**Step B — P2 training** (~10 hrs on A6000/4090): warm-starts from yolo26n
backbone, P2 head random-init.

```bash
python scripts/train_p2.py \
    --cfg configs/yolo26-p2.yaml \
    --name p1_exp1 \
    --pretrained yolo26n.pt \
    --epochs 30 --batch 16 --imgsz 640
```

Output: `runs/p2/p1_exp1/training_monitor.csv` (per-epoch × size-bin metrics)

---

## Reading `training_monitor.csv`

Columns: `epoch, size_bin, recall_at_05, median_iou, fp_above_tp_median, median_tp_conf, gt_count, tp_count, fp_count`

Each epoch writes 6 rows (one per size bin: `0-8`, `8-16`, `16-32`, `32-64`, `64-128`, `128-9999`).

**The three bottleneck indices**:

| Column | Tracks | Good trend |
|---|---|---|
| `recall_at_05` | Fraction of GTs found at IoU ≥ 0.5 | ↑ over epochs |
| `median_iou` | Median IoU of matched TPs | ↑ (better regression) |
| `fp_above_tp_median` | FPs ranked above typical TP conf | ↓ (better calibration) |

**Compare P2 vs baseline**:
- Open both CSVs at the smallest bins (`0-8`, `8-16`).
- P2 success = `recall_at_05` for `0-16px` improves ≥ +5% vs baseline.
- See `docs/research_plan_phase1.md` for full success criteria.

---

## Rules for this session (cost-aware behavior)

1. **Don't explore** — everything you need is in this CLAUDE.md and research plan.
   Resist the urge to `ls -R` or read random files.
2. **Don't re-fetch COCO** — it's already on `/workspace/coco`. Re-downloading
   is 20 GB and wastes ~25 min.
3. **Don't re-install env needlessly** — if `/root/sod-env` exists, activate it.
4. **If something fails fast** — read the error, fix the narrow cause, retry.
   Don't start broad refactors.
5. **Training progress is via `training_monitor.csv`** — don't `tail -f` the
   training log for hours; check the CSV periodically.
6. **After training finishes**: commit `runs/p2/p1_exp1/{training_monitor.csv,
   results.csv, args.yaml}` + the baseline CSV back to a new branch so the
   user can pull results from anywhere without the pod running.

---

## Files you'll touch (whole list)

| File | Purpose |
|---|---|
| `configs/yolo26-p2.yaml` | Model architecture (P2 + P3 + P4 + P5 heads) |
| `scripts/train_p2.py` | Training entrypoint with monitor registered |
| `scripts/training_monitor.py` | Per-epoch bottleneck metric callback |
| `scripts/eval_baseline.py` | One-shot baseline val + bottleneck metrics |
| `scripts/pod_fetch_coco.sh` | Pod-side COCO setup (only if volume is blank) |
| `docs/research_plan_phase1.md` | Hypothesis / design / success criteria |

**Do not touch files outside this list** without clear reason.
