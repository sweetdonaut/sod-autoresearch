"""Train a P2-head YOLO26 on COCO with per-epoch bottleneck monitor.

Designed to run on a RunPod pod where COCO lives on a mounted network volume.
Uses env var COCO_ROOT (default /workspace/coco) to locate the dataset.

Example:
    # Warm-start from pretrained yolo26n backbone (P2 head random-init)
    python scripts/train_p2.py \\
        --cfg configs/yolo26-p2.yaml \\
        --name p1_exp1 \\
        --pretrained yolo26n.pt \\
        --epochs 30 --batch 16 --imgsz 640
"""
import argparse
import os
from pathlib import Path

from ultralytics import YOLO

from training_monitor import register_training_monitor

PROJECT = Path(__file__).resolve().parent.parent
COCO_ROOT = os.environ.get("COCO_ROOT", "/workspace/coco")
DEFAULT_GT_PATH = f"{COCO_ROOT}/annotations/instances_val2017.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True,
                        help="Model yaml (e.g. configs/yolo26-p2.yaml)")
    parser.add_argument("--name", required=True,
                        help="Run name; output under runs/p2/{name}/")
    parser.add_argument("--task", default="detect", choices=["detect", "segment"])
    parser.add_argument("--pretrained", default=None,
                        help="Optional .pt weights to warm-start from. "
                             "Matching layers load; new layers (e.g. P2 head) "
                             "stay random-init.")
    parser.add_argument("--data", default="coco.yaml",
                        help="Dataset yaml (default: coco.yaml — ultralytics "
                             "will resolve via its datasets_dir setting)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--optimizer", type=str, default="auto")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project-dir", default=str(PROJECT / "runs" / "p2"))
    parser.add_argument("--monitor-gt", default=DEFAULT_GT_PATH,
                        help=f"COCO val annotations JSON (default: $COCO_ROOT/annotations/instances_val2017.json)")
    parser.add_argument("--monitor-freq", type=int, default=1,
                        help="Run monitor every N epochs")
    parser.add_argument("--no-monitor", action="store_true",
                        help="Skip bottleneck monitor (stock YOLO val only)")
    parser.add_argument("--no-export", action="store_true",
                        help="Skip ONNX export after training")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  {args.name} | task={args.task} | epochs={args.epochs}")
    print(f"  cfg={args.cfg}  pretrained={args.pretrained or 'none'}")
    print(f"  imgsz={args.imgsz} batch={args.batch} optimizer={args.optimizer}")
    print(f"  COCO_ROOT={COCO_ROOT}")
    print(f"{'='*60}")

    model = YOLO(args.cfg, task=args.task)
    if args.pretrained:
        print(f"\nLoading pretrained weights: {args.pretrained}")
        print("  (matching layers transfer; new layers init randomly)")
        model = model.load(args.pretrained)
    model.info()

    run_dir = Path(args.project_dir) / args.name
    if not args.no_monitor:
        if not Path(args.monitor_gt).exists():
            raise FileNotFoundError(
                f"Monitor GT file not found: {args.monitor_gt}\n"
                f"Set COCO_ROOT env var, pass --monitor-gt, or --no-monitor"
            )
        register_training_monitor(
            model,
            gt_path=args.monitor_gt,
            output_csv=run_dir / "training_monitor.csv",
            every_n_epochs=args.monitor_freq,
        )

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        project=args.project_dir,
        exist_ok=True,
        workers=args.workers,
        optimizer=args.optimizer,
    )

    if not args.no_export:
        print(f"\n--- Exporting {args.name} to ONNX ---")
        onnx_path = model.export(format="onnx")
        print(f"Done: {onnx_path}")


if __name__ == "__main__":
    main()
