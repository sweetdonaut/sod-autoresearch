"""Ablation: train yolo26x WITHOUT P2 head on VisDrone, identical
hyperparameters to train_visdrone_p2.py. The only variable is P2
presence, so the delta between the two runs isolates the P2 contribution.

Compared to train_visdrone_p2.py the only differences are:
  1. CFG_PATH = None — we skip the custom YAML and load the plain
     architecture directly from yolo26x.pt (3 Detect heads at P3/P4/P5).
  2. RUN_NAME swapped to visdrone_nop2_x_1280_50ep.

Everything else is held constant:
  - imgsz=1280, batch=8, epochs=50, close_mosaic=10
  - optimizer=auto (AdamW auto-lr), seed=0, deterministic=True
  - warm-start from yolo26x.pt, AMP, same workers / cache settings
  - same training monitor + 12-class GT used for per-epoch bottleneck
    metrics (training_monitor.py compute_bottleneck_metrics is
    class-matched, nc=10; still valid here)
"""
from pathlib import Path

from ultralytics import YOLO

from training_monitor import register_training_monitor

# =============================================================================
# Run config
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- What to train -----------------------------------------------------------
CFG_PATH   = None                 # use .pt directly (no custom YAML)
PRETRAINED = "yolo26x.pt"         # architecture + weights both from here
RUN_NAME   = "visdrone_nop2_x_1280_50ep"
TASK       = "detect"
DATA       = "VisDrone.yaml"

# --- Where things live -------------------------------------------------------
PROJECT_DIR     = str(PROJECT_ROOT / "runs" / "p2")
VISDRONE_VAL_GT = "/workspace/datasets/VisDrone/annotations/instances_val_stem.json"

# --- Training hyperparams (IDENTICAL to train_visdrone_p2.py) ---------------
EPOCHS        = 50
IMGSZ         = 1280
BATCH         = 8      # keep same as P2 run for apples-to-apples gradient
                       # batch. NoP2 has more VRAM headroom (no 320x320 P2
                       # feature map), but raising batch would change BN
                       # stats and optimizer step count — confound.
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
EXPORT_ONNX_AT_END     = False
# =============================================================================


def main():
    run_dir = Path(PROJECT_DIR) / RUN_NAME

    print(f"\n{'='*60}")
    print(f"  RUN_NAME={RUN_NAME}   TASK={TASK}   EPOCHS={EPOCHS}")
    print(f"  cfg={CFG_PATH}  (None -> architecture from {PRETRAINED})")
    print(f"  pretrained={PRETRAINED}")
    print(f"  imgsz={IMGSZ}  batch={BATCH}  workers={WORKERS}  cache={CACHE}")
    print(f"  data={DATA}")
    print(f"{'='*60}\n")

    if CFG_PATH is not None:
        model = YOLO(CFG_PATH, task=TASK)
        print(f"Loading pretrained weights: {PRETRAINED}")
        model = model.load(PRETRAINED)
    else:
        # Plain path: load architecture + weights together from .pt.
        # Detect head will still re-init for nc=10 (VisDrone) via data.yaml.
        model = YOLO(PRETRAINED, task=TASK)
        print(f"Loaded architecture + weights from: {PRETRAINED}")
        print("  (Detect head re-inits to nc=10 on first train step via data.yaml)")
    model.info()

    if not Path(VISDRONE_VAL_GT).exists():
        print(f"[warn] monitor GT not present at: {VISDRONE_VAL_GT}")
        print(f"       run: python scripts/build_visdrone_coco_gt.py --split val")

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
