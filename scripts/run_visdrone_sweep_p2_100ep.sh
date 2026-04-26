#!/bin/bash
# Round 2: extended-budget P2 trainings.
#
# Both n+P2 and s+P2 hit ep 50 still climbing in Round 1 (peak at ep 50 / 38;
# cls_loss still falling). 100 epochs gives them the budget to actually
# saturate, so the P2 vs plain comparison isn't dominated by undertraining.
#
# Apples-to-apples note: round 1 plain runs (n, s) used 50 ep and saturated
# cleanly by ep 30-48. Round 2 results should therefore be read as "P2 with
# enough budget" — not directly comparable to round 1's plain numbers, but a
# fair assessment of the P2 architecture's converged performance.

set -euo pipefail

cd /workspace/sod-autoresearch
source /root/sod-env/bin/activate
export PYTORCH_ALLOC_CONF=expandable_segments:True

SWEEP_DIR=runs/sweep
LOG_DIR=$SWEEP_DIR/_logs
mkdir -p "$LOG_DIR"

VARIANTS=(
  "n_p2_100ep|--scale n --p2 --epochs 100"
  "s_p2_100ep|--scale s --p2 --epochs 100"
)

for v in "${VARIANTS[@]}"; do
  tag=${v%%|*}
  args=${v#*|}
  # train_visdrone_sweep.py builds run_name as visdrone_{train_tag}_{imgsz}_{epochs}ep
  # where train_tag here = scale[_p2] (no "_100ep" suffix on its side)
  case "$tag" in
    n_p2_100ep) train_run_name="visdrone_n_p2_1280_100ep" ;;
    s_p2_100ep) train_run_name="visdrone_s_p2_1280_100ep" ;;
    *) train_run_name="visdrone_${tag}_1280" ;;
  esac
  run_dir="$SWEEP_DIR/$train_run_name"
  log="$LOG_DIR/${tag}.log"

  done_rows=$(awk 'NR>1' "$run_dir/results.csv" 2>/dev/null | wc -l)
  if [ "$done_rows" = "100" ] && [ -f "$run_dir/weights/best.pt" ]; then
    echo "=== SKIP $tag — already 100 epochs done ==="
    continue
  fi

  if [ -f "$run_dir/weights/last.pt" ]; then
    echo "=== RESUME $tag — last.pt exists ($done_rows/100 done) — logging to $log ==="
    python scripts/train_visdrone_sweep.py $args --resume 2>&1 | tee -a "$log"
  else
    echo "=== TRAIN $tag (args: $args) — logging to $log ==="
    python scripts/train_visdrone_sweep.py $args 2>&1 | tee "$log"
  fi
done

# Dome-DETR protocol eval on each
echo ""
echo "=== Dome-DETR protocol eval (Round 2) ==="
for tag in n_p2_100ep s_p2_100ep; do
  best_pt="$SWEEP_DIR/visdrone_${tag}_1280/weights/best.pt"
  out_dir="runs/dome_detr_eval/${tag}_at_1280_letterbox"
  if [ -f "$out_dir/dome_detr_stats.json" ]; then
    echo "    [skip] $tag — stats.json exists"
    continue
  fi
  if [ ! -f "$best_pt" ]; then
    echo "    [skip] $tag — no best.pt"
    continue
  fi
  echo "    [eval] $tag"
  python scripts/eval_dome_detr_protocol.py \
    --weights "$best_pt" --imgsz 1280 \
    --name "${tag}_at_1280_letterbox" 2>&1 | tail -18
done

echo ""
echo "=== ROUND 2 DONE ==="
