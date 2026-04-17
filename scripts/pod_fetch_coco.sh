#!/usr/bin/env bash
#
# Fetch COCO dataset from official servers onto a RunPod pod's mounted volume.
# Run this ON the pod (not locally). Uses wget with resume, extracts in parallel,
# and generates ultralytics-compatible YOLO labels via convert_coco.
#
# Usage (on the pod):
#   wget -O /tmp/pod_fetch_coco.sh https://... # or just paste this script
#   chmod +x /tmp/pod_fetch_coco.sh
#   /tmp/pod_fetch_coco.sh [TARGET_DIR]
#
# Default TARGET_DIR is /workspace/coco. Change to wherever your network volume
# is mounted (e.g. /runpod-volume/coco).
#
# Skips test2017 (no GT, we never use it). Total ~20 GB; expect 10-25 minutes
# on a RunPod pod with typical bandwidth.

set -euo pipefail

TARGET="${1:-/workspace/coco}"
START_TS=$(date +%s)

echo "=============================================================="
echo "  COCO fetch to $TARGET"
echo "  started: $(date)"
echo "=============================================================="

mkdir -p "$TARGET/images" "$TARGET/annotations" "$TARGET/labels"
cd "$TARGET"

# ---- 1. Download in parallel ----
URL_VAL="http://images.cocodataset.org/zips/val2017.zip"
URL_TRAIN="http://images.cocodataset.org/zips/train2017.zip"
URL_ANN="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

echo ""
echo "[1/4] Downloading (parallel wget, with resume)..."

# -c = resume partial; --no-verbose reduces log noise; & runs in background
wget -c --no-verbose "$URL_VAL"   -O val2017.zip   &   pid_val=$!
wget -c --no-verbose "$URL_ANN"   -O annotations.zip & pid_ann=$!
wget -c --no-verbose "$URL_TRAIN" -O train2017.zip &   pid_train=$!

# Progress indicator
monitor_downloads() {
    while kill -0 $pid_val 2>/dev/null || \
          kill -0 $pid_ann 2>/dev/null || \
          kill -0 $pid_train 2>/dev/null; do
        sleep 10
        local v=$(du -b val2017.zip       2>/dev/null | awk '{print $1}' || echo 0)
        local a=$(du -b annotations.zip   2>/dev/null | awk '{print $1}' || echo 0)
        local t=$(du -b train2017.zip     2>/dev/null | awk '{print $1}' || echo 0)
        local total_mb=$(( (v + a + t) / 1024 / 1024 ))
        local elapsed=$(( $(date +%s) - START_TS ))
        printf "\r  [%ds] val:%4dMB  ann:%4dMB  train:%5dMB  total:%5dMB   " \
            "$elapsed" $((v/1024/1024)) $((a/1024/1024)) $((t/1024/1024)) "$total_mb"
    done
    echo ""
}
monitor_downloads

wait $pid_val $pid_ann $pid_train
echo "  all downloads complete."

# ---- 2. Extract ----
echo ""
echo "[2/4] Extracting zips..."

unzip -q -n val2017.zip   -d images/   &  pid_uv=$!
unzip -q -n train2017.zip -d images/   &  pid_ut=$!
unzip -q -n annotations.zip            &  pid_ua=$!
wait $pid_uv $pid_ut $pid_ua
echo "  extracted: images/val2017, images/train2017, annotations/"

# ---- 3. Generate YOLO labels via ultralytics ----
echo ""
echo "[3/4] Converting COCO annotations -> YOLO labels..."

if ! python -c "import ultralytics" 2>/dev/null; then
    echo "  Installing ultralytics..."
    pip install --quiet ultralytics
fi

python - <<'PYEOF'
from pathlib import Path
import shutil
from ultralytics.data.converter import convert_coco

target = Path(".").resolve()
ann_dir = target / "annotations"

# convert_coco processes every JSON in labels_dir — must isolate ONLY
# instances_*.json (otherwise captions_*.json / person_keypoints_*.json
# trigger KeyError: 'bbox' since those tasks have no bbox field).
instances_dir = target / "_instances_tmp"
if instances_dir.exists():
    shutil.rmtree(instances_dir)
instances_dir.mkdir()
for f in ann_dir.glob("instances_*.json"):
    shutil.copy(f, instances_dir / f.name)
    print(f"  staged: {f.name}")

out = target / "_yolo_labels_tmp"
if out.exists():
    shutil.rmtree(out)

convert_coco(
    labels_dir=str(instances_dir),
    save_dir=str(out),
    use_segments=False,
    use_keypoints=False,
    cls91to80=True,
)

# convert_coco's output directory name varies by ultralytics version:
# older versions keep the "instances_" prefix, newer versions strip it.
# Search for *train2017* / *val2017* directories to be robust.
def find_split_dir(root, split):
    candidates = [p for p in root.rglob("*") if p.is_dir() and split in p.name]
    # Pick the deepest matching dir that actually contains .txt files
    candidates = [p for p in candidates if any(p.glob("*.txt"))]
    if not candidates:
        raise FileNotFoundError(f"no directory containing {split}/*.txt found under {root}")
    return sorted(candidates, key=lambda p: len(p.parts))[-1]

src_train = find_split_dir(out, "train2017")
src_val   = find_split_dir(out, "val2017")
dst_train = target / "labels" / "train2017"
dst_val   = target / "labels" / "val2017"

dst_train.parent.mkdir(exist_ok=True)
for src, dst in [(src_train, dst_train), (src_val, dst_val)]:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(src), str(dst))
    print(f"  labels placed: {dst} ({len(list(dst.iterdir()))} files)")

shutil.rmtree(out)
shutil.rmtree(instances_dir)
PYEOF

# ---- 4. Cleanup ----
echo ""
echo "[4/4] Cleaning up zip files..."
rm -f val2017.zip train2017.zip annotations.zip

# ---- Summary ----
TOTAL_ELAPSED=$(( $(date +%s) - START_TS ))
echo ""
echo "=============================================================="
echo "  DONE in $((TOTAL_ELAPSED / 60))m $((TOTAL_ELAPSED % 60))s"
echo ""
echo "  Final layout under $TARGET/:"
find "$TARGET" -maxdepth 3 -type d | sort | head -20
echo ""
echo "  image counts:"
echo "    images/train2017: $(ls "$TARGET/images/train2017" 2>/dev/null | wc -l)"
echo "    images/val2017:   $(ls "$TARGET/images/val2017"   2>/dev/null | wc -l)"
echo "    labels/train2017: $(ls "$TARGET/labels/train2017" 2>/dev/null | wc -l)"
echo "    labels/val2017:   $(ls "$TARGET/labels/val2017"   2>/dev/null | wc -l)"
echo "=============================================================="
