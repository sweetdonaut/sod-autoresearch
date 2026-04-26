#!/bin/bash
# Sequential training sweep for VisDrone scale comparison.
#
# Runs 6 variants back-to-back on a single A100:
#   n, n+P2, s, s+P2, m, l
#
# Skips any run whose best.pt already exists, making it safe to re-invoke
# if one crashes mid-sweep.
#
# After all trainings, runs the Dome-DETR protocol eval on every best.pt
# (including the 2 existing x runs under runs/p2/) so the benchmark plot
# has 8 data points at identical eval protocol.
set -euo pipefail

cd /workspace/sod-autoresearch
source /root/sod-env/bin/activate
export PYTORCH_ALLOC_CONF=expandable_segments:True

SWEEP_DIR=runs/sweep
LOG_DIR=$SWEEP_DIR/_logs
mkdir -p "$LOG_DIR"

# variant spec: "<tag>|<args>"
VARIANTS=(
  "n|--scale n"
  "n_p2|--scale n --p2"
  "s|--scale s"
  "s_p2|--scale s --p2"
  "m|--scale m"
  "l|--scale l"
)

for v in "${VARIANTS[@]}"; do
  tag=${v%%|*}
  args=${v#*|}
  run_name="visdrone_${tag}_1280_50ep"
  best_pt="$SWEEP_DIR/$run_name/weights/best.pt"
  log="$LOG_DIR/${tag}.log"

  if [ -f "$best_pt" ]; then
    echo "=== SKIP $tag — best.pt already exists ==="
    continue
  fi

  echo "=== TRAIN $tag (args: $args) — logging to $log ==="
  python scripts/train_visdrone_sweep.py $args 2>&1 | tee "$log"
done

# ---------------------------------------------------------------------------
# Dome-DETR protocol eval on every trained best.pt
# ---------------------------------------------------------------------------
echo ""
echo "=== Dome-DETR protocol eval on all variants ==="

run_eval() {
  local run_path="$1"
  local eval_name="$2"
  local best_pt="$run_path/weights/best.pt"
  local out_dir="runs/dome_detr_eval/$eval_name"

  if [ -f "$out_dir/dome_detr_stats.json" ]; then
    echo "    [skip] $eval_name — dome_detr_stats.json exists"
    return
  fi
  if [ ! -f "$best_pt" ]; then
    echo "    [skip] $eval_name — no best.pt at $best_pt"
    return
  fi
  echo "    [eval] $eval_name"
  python scripts/eval_dome_detr_protocol.py \
    --weights "$best_pt" --imgsz 1280 --name "$eval_name" \
    2>&1 | tail -20
}

for tag in n n_p2 s s_p2 m l; do
  run_eval "$SWEEP_DIR/visdrone_${tag}_1280_50ep" "${tag}_at_1280_letterbox"
done

echo ""
echo "=== ALL DONE ==="
echo "Sweep outputs in: $SWEEP_DIR/"
echo "Eval outputs in:  runs/dome_detr_eval/"
