"""Train YOLO26n-P2 on COCO with per-epoch bottleneck monitor.

All run config is hardcoded below — no CLI args. To change a run, edit this
file directly (the file itself is the audit record).

Output: runs/p2/{RUN_NAME}/
  - training_monitor.csv   per-epoch × size-bin bottleneck metrics
  - weights/{last,best}.pt ultralytics checkpoints
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
CFG_PATH   = str(PROJECT_ROOT / "configs" / "yolo26s-p2.yaml")
PRETRAINED = "yolo26s.pt"       # warm-start; P2 head stays random-init
RUN_NAME   = "p1_exp2_s_15ep"
TASK       = "detect"
DATA       = "coco.yaml"        # ultralytics auto-downloads to datasets_dir

# --- Where things live (ultralytics datasets_dir = /workspace/datasets) ------
PROJECT_DIR  = str(PROJECT_ROOT / "runs" / "p2")
COCO_VAL_GT  = "/workspace/datasets/coco/annotations/instances_val2017.json"

# --- Training hyperparams ----------------------------------------------------
EPOCHS        = 15
IMGSZ         = 640
BATCH         = 0.97    # auto: fill 97% VRAM (ultralytics float-batch convention)
WORKERS       = 8      # 64 CPU cores → 32 dataloader workers
CACHE         = False  # GPU-bound; cache gives 0% speedup. Keep False for determinism.
                        # after epoch 0 (first epoch still does the ingest).
AMP           = True    # fp16 on A100 tensor cores
DEVICE        = 0
OPTIMIZER     = "auto"  # ultralytics picks SGD/AdamW per schedule heuristic
CLOSE_MOSAIC  = 10      # standard: disable mosaic for last 10 epochs
SEED          = 0
DETERMINISTIC = True
FRACTION = 1.0

# --- Monitor / export --------------------------------------------------------
MONITOR_EVERY_N_EPOCHS = 1
EXPORT_ONNX_AT_END     = True
# =============================================================================


def main():
    run_dir = Path(PROJECT_DIR) / RUN_NAME

    print(f"\n{'='*60}")
    print(f"  RUN_NAME={RUN_NAME}   TASK={TASK}   EPOCHS={EPOCHS}")
    print(f"  cfg={CFG_PATH}")
    print(f"  pretrained={PRETRAINED}")
    print(f"  imgsz={IMGSZ}  batch={BATCH}  workers={WORKERS}  cache={CACHE}")
    print(f"  data={DATA}   datasets_dir=/workspace/datasets")
    print(f"{'='*60}\n")

    model = YOLO(CFG_PATH, task=TASK)
    print(f"Loading pretrained weights: {PRETRAINED}")
    print("  (matching layers transfer; P2 head inits randomly)")
    model = model.load(PRETRAINED)
    model.info()

    if not Path(COCO_VAL_GT).exists():
        print(f"[info] monitor GT not present yet: {COCO_VAL_GT}")
        print(f"       ultralytics will download COCO on first train step;")
        print(f"       monitor reads the file lazily on first val end.")

    register_training_monitor(
        model,
        gt_path=COCO_VAL_GT,
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
