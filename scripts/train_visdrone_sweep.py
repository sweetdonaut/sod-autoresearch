"""Unified training entry for the VisDrone scale sweep. One script, one
model variant per invocation, selected by --scale and --p2.

Produces the same output layout as train_visdrone_{p2,nop2}.py so the
plot / eval scripts can glob across all runs uniformly.

Usage:
    python scripts/train_visdrone_sweep.py --scale n
    python scripts/train_visdrone_sweep.py --scale s --p2
    PYTORCH_ALLOC_CONF=expandable_segments:True \\
        python scripts/train_visdrone_sweep.py --scale l

Outputs:
    runs/sweep/visdrone_{scale}[_p2]_1280_50ep/
       weights/{best,last}.pt
       results.csv, training_monitor.csv, args.yaml, predictions.json
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

from training_monitor import register_training_monitor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_DIR = str(PROJECT_ROOT / "runs" / "sweep")
VISDRONE_VAL_GT = (
    "/workspace/datasets/VisDrone/annotations/instances_val_stem.json"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", required=True,
                    choices=["n", "s", "m", "l", "x"])
    ap.add_argument("--p2", action="store_true",
                    help="Add P2 (stride-4) detection head")
    ap.add_argument("--batch", type=float, default=0.80,
                    help="auto-batch fraction (0.80 = fill 80% VRAM). "
                         "Lower than 0.90 to leave headroom for "
                         "TaskAlignedAssigner's data-dependent tensors; "
                         "at 0.90 auto-batch picked batch=16 for n+P2 which "
                         "triggered TAL OOM -> CPU fallback on dense images.")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--close-mosaic", type=int, default=10)
    ap.add_argument("--resume", action="store_true",
                    help="Resume from {run_dir}/weights/last.pt if present. "
                         "Restores LR schedule, optimizer, RNG state. "
                         "Other hyperparam args are ignored on resume.")
    args = ap.parse_args()

    scale = args.scale
    pretrained = f"yolo26{scale}.pt"
    if args.p2:
        # ultralytics parses scale from filename suffix; reloads
        # configs/yolo26-p2.yaml with scale=<suffix>. The file
        # yolo26{scale}-p2.yaml does not need to exist on disk.
        cfg = str(PROJECT_ROOT / "configs" / f"yolo26{scale}-p2.yaml")
        tag = f"{scale}_p2"
    else:
        cfg = None     # use the pretrained .pt as both architecture + weights
        tag = scale

    run_name = f"visdrone_{tag}_{args.imgsz}_{args.epochs}ep"
    run_dir = Path(PROJECT_DIR) / run_name

    print(f"\n{'='*64}")
    print(f"  RUN_NAME={run_name}")
    print(f"  scale={scale}  P2={args.p2}")
    print(f"  cfg={cfg}   pretrained={pretrained}")
    print(f"  imgsz={args.imgsz}  batch={args.batch}  "
          f"epochs={args.epochs}  seed={args.seed}")
    print(f"{'='*64}\n")

    last_pt = run_dir / "weights" / "last.pt"
    resume_mode = args.resume and last_pt.exists()
    if resume_mode:
        print(f"[resume] picking up from {last_pt}")
        model = YOLO(str(last_pt))
    elif cfg is not None:
        model = YOLO(cfg, task="detect")
        print(f"Loading pretrained weights: {pretrained}")
        model = model.load(pretrained)
    else:
        model = YOLO(pretrained, task="detect")
        print(f"Loaded architecture + weights from {pretrained}")
    model.info()

    if not Path(VISDRONE_VAL_GT).exists():
        print(f"[warn] monitor GT missing: {VISDRONE_VAL_GT}")
        print(f"       run: python scripts/build_visdrone_coco_gt.py --split val")

    register_training_monitor(
        model,
        gt_path=VISDRONE_VAL_GT,
        output_csv=run_dir / "training_monitor.csv",
        every_n_epochs=1,
    )

    if resume_mode:
        # On resume, ultralytics restores all hyperparams from the
        # checkpoint's args.yaml; passing args.epochs etc. is unnecessary
        # and can cause schedule mismatches.
        model.train(resume=True)
    else:
        model.train(
            data="VisDrone.yaml",
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            cache=False,
            amp=True,
            device=0,
            optimizer="auto",
            close_mosaic=args.close_mosaic,
            seed=args.seed,
            deterministic=True,
            fraction=1.0,
            name=run_name,
            project=PROJECT_DIR,
            exist_ok=True,
        )


if __name__ == "__main__":
    main()
