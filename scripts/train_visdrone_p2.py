"""Train YOLO26x-P2 @ imgsz=1280 on VisDrone with per-epoch bottleneck
monitor.

Derived from scripts/train_p2.py (Phase 1 COCO scaffolding). Changes:
  - Model: yolo26-p2.yaml @ scale x (was n)
  - Data:  VisDrone.yaml (10 cls, 6471 train / 548 val) (was coco.yaml)
  - imgsz: 1280 (was 640) — VisDrone val's 79% GT <32px needs resolution
  - batch: 0.90 auto — fill A100 80GB (Phase 1 used 0.97 at 640)
  - epochs: 50 — VisDrone train is 18x smaller than COCO; more epochs fair
  - monitor GT: stem-keyed JSON (built via build_visdrone_coco_gt.py)

All run config is hardcoded below — edit this file to change a run.

Output: runs/p2/{RUN_NAME}/
  - training_monitor.csv   per-epoch × size-bin bottleneck metrics
  - results.csv            ultralytics standard per-epoch metrics (mAP, etc.)
  - weights/{last,best}.pt checkpoints
  - args.yaml              ultralytics auto-dumps all effective args
"""
from pathlib import Path

from ultralytics import YOLO

from training_monitor import register_training_monitor

# =============================================================================
# Run config  (edit here, not via CLI)
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- What to train -----------------------------------------------------------
# scale is parsed from filename suffix (ultralytics strips [nslmx] and
# reloads configs/yolo26-p2.yaml with scale='x'). The file yolo26x-p2.yaml
# does not need to exist on disk.
CFG_PATH   = str(PROJECT_ROOT / "configs" / "yolo26x-p2.yaml")
PRETRAINED = "yolo26x.pt"          # warm-start; P2 head stays random-init
RUN_NAME   = "visdrone_p2_x_1280_50ep"
TASK       = "detect"
DATA       = "VisDrone.yaml"       # ultralytics auto-finds in datasets_dir

# --- Where things live -------------------------------------------------------
PROJECT_DIR     = str(PROJECT_ROOT / "runs" / "p2")
VISDRONE_VAL_GT = "/workspace/datasets/VisDrone/annotations/instances_val_stem.json"

# --- Training hyperparams ----------------------------------------------------
EPOCHS        = 50
IMGSZ         = 1280
BATCH         = 8      # Manually fixed. AutoBatch's peak-forward-pass probe
                       # overestimates steady-state memory by ~2x: actual
                       # training at batch=4 only used 33.5GB / 80GB (42%).
                       # With expandable_segments ON, batch=8 should land
                       # around 60-67GB — real "fill the A100". If OOM,
                       # drop to 6.
WORKERS       = 12
CACHE         = False
AMP           = True
DEVICE        = 0
OPTIMIZER     = "auto"
CLOSE_MOSAIC  = 10
SEED          = 0
DETERMINISTIC = True
FRACTION      = 1.0

# --- Monitor / export --------------------------------------------------------
MONITOR_EVERY_N_EPOCHS = 1
EXPORT_ONNX_AT_END     = False  # yolo26x ONNX is large; defer to post-analysis
# =============================================================================


def main():
    run_dir = Path(PROJECT_DIR) / RUN_NAME

    print(f"\n{'='*60}")
    print(f"  RUN_NAME={RUN_NAME}   TASK={TASK}   EPOCHS={EPOCHS}")
    print(f"  cfg={CFG_PATH}")
    print(f"  pretrained={PRETRAINED}")
    print(f"  imgsz={IMGSZ}  batch={BATCH}  workers={WORKERS}  cache={CACHE}")
    print(f"  data={DATA}")
    print(f"{'='*60}\n")

    model = YOLO(CFG_PATH, task=TASK)
    print(f"Loading pretrained weights: {PRETRAINED}")
    print("  (matching layers transfer; P2 head inits randomly;")
    print("   Detect head will re-init for nc=10 VisDrone)")
    model = model.load(PRETRAINED)
    model.info()

    if not Path(VISDRONE_VAL_GT).exists():
        print(f"[warn] monitor GT not present at: {VISDRONE_VAL_GT}")
        print(f"       run: python scripts/build_visdrone_coco_gt.py --split val")
        print(f"       (monitor will no-op until GT appears)")

    register_training_monitor(
        model,
        gt_path=VISDRONE_VAL_GT,
        output_csv=run_dir / "training_monitor.csv",
        every_n_epochs=MONITOR_EVERY_N_EPOCHS,
    )

    model.train(
        data=DATA,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        cache=CACHE,
        amp=AMP,
        device=DEVICE,
        optimizer=OPTIMIZER,
        close_mosaic=CLOSE_MOSAIC,
        seed=SEED,
        deterministic=DETERMINISTIC,
        fraction=FRACTION,
        name=RUN_NAME,
        project=PROJECT_DIR,
        exist_ok=True,
    )

    if EXPORT_ONNX_AT_END:
        print(f"\n--- Exporting {RUN_NAME} to ONNX ---")
        onnx_path = model.export(format="onnx")
        print(f"Done: {onnx_path}")


if __name__ == "__main__":
    main()
