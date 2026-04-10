# COCO val2017 Baseline Results

## Overview

使用 YOLO11 和 YOLO26 全系列預訓練模型在 COCO val2017 上評估，建立 small object detection 的 baseline。所有指標透過 `faster-coco-eval`（與 pycocotools bit-for-bit identical）的 `COCOeval.stats` API 取得。

## Evaluation Settings

| Parameter | Value |
|-----------|-------|
| Dataset | COCO val2017 (5000 images) |
| Image size | 640 |
| Confidence threshold | 0.001 |
| NMS IoU threshold | 0.7 |
| Max detections | 300 |
| TTA (augment) | False |
| Rect inference | False |
| Evaluation tool | faster-coco-eval (COCOeval API) |
| Ultralytics version | 8.4.36 |
| GPU | NVIDIA RTX 5070 Ti 16GB |

## Results

| Model | End2End | AP | AP50 | AP75 | AP_S | AP_M | AP_L |
|-------|---------|------|------|------|------|------|------|
| yolo11n | No | 0.394 | 0.553 | 0.428 | 0.200 | 0.433 | 0.571 |
| yolo11s | No | 0.469 | 0.639 | 0.506 | 0.297 | 0.516 | 0.643 |
| yolo11m | No | 0.515 | 0.684 | 0.557 | 0.335 | 0.571 | 0.678 |
| yolo11l | No | 0.533 | 0.701 | 0.582 | 0.356 | 0.591 | 0.693 |
| yolo11x | No | 0.546 | 0.716 | 0.595 | 0.377 | 0.597 | 0.703 |
| yolo26n | Yes | 0.400 | 0.557 | 0.434 | 0.198 | 0.441 | 0.580 |
| yolo26s | Yes | 0.477 | 0.645 | 0.521 | 0.291 | 0.525 | 0.645 |
| yolo26m | Yes | 0.525 | 0.698 | 0.573 | 0.363 | 0.569 | 0.685 |
| yolo26l | Yes | 0.543 | 0.715 | 0.594 | 0.378 | 0.586 | 0.704 |
| yolo26x | Yes | 0.568 | 0.742 | 0.621 | 0.413 | 0.612 | 0.727 |

## Key Observations

### 1. Small Object Detection Gap

AP_S 顯著低於 AP_M 和 AP_L，差距約 15-20 個百分點：

- yolo11n: AP_S=0.200 vs AP_M=0.433 vs AP_L=0.571
- yolo26x: AP_S=0.413 vs AP_M=0.612 vs AP_L=0.727

即使最大的 yolo26x，AP_S 也只有 0.413，小物體偵測有很大的改進空間。

### 2. YOLO26 vs YOLO11

同尺寸比較，YOLO26 在 m/l/x 上整體略優於 YOLO11（AP 差距 +0.5~2.2），但在 n/s 尺寸差距很小。YOLO26 是 end-to-end 架構（無 NMS），YOLO11 仍使用傳統 NMS。

### 3. AP_S 隨模型增大的提升幅度

| Scale | YOLO11 AP_S | YOLO26 AP_S |
|-------|-------------|-------------|
| n → x | 0.200 → 0.377 (+0.177) | 0.198 → 0.413 (+0.215) |

YOLO26 在小物體上隨模型增大的提升幅度（+0.215）大於 YOLO11（+0.177）。

## Reproducibility Notes

### 與官方公佈值的差距

我們的結果與 Ultralytics 官方公佈值存在 0.7-1.0 的一致性偏低差距（例如 yolo26x: 我們 0.568 vs 官方 0.575）。經調查：

- 這是社群廣泛反映的已知問題（GitHub issues #19195, #3608, #20772）
- 官方 checkpoint 使用 internal training branch 產出，包含公開版沒有的參數（`cls_w`, `muon_w`, `sgd_w` 等）
- TTA 和 rect inference 經測試對結果無影響
- 官方未公開完整的 evaluation methodology

**本 baseline 所有模型在同一環境下評估，相對比較公平且一致。**

### COCO AP_S 定義

- Small: area < 32^2 = 1024 px^2
- Medium: 1024 <= area < 96^2 = 9216 px^2
- Large: area >= 9216 px^2

AP_S/AP_M/AP_L 是在完整 val2017 上評估，由 pycocotools 按 annotation area 自動分類計算。

## Data & Scripts

- Evaluation script: `scripts/eval_coco_baseline.py`
- Results CSV: `results/coco_baseline.csv`
- Dataset: `/home/yclaizzs/ML_exploration/datasets/coco/` (COCO 2017, via ultralytics)
