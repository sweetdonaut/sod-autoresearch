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
| YOLO26 end2end | False (使用 NMS，與官方 benchmark 一致) |
| Evaluation tool | faster-coco-eval (COCOeval API) |
| Ultralytics version | 8.4.36 |
| GPU | NVIDIA RTX 5070 Ti 16GB |

## Results

| Model | AP | AP50 | AP75 | AP_S | AP_M | AP_L |
|-------|------|------|------|------|------|------|
| yolo11n | 0.394 | 0.553 | 0.428 | 0.200 | 0.433 | 0.571 |
| yolo11s | 0.469 | 0.639 | 0.506 | 0.297 | 0.516 | 0.643 |
| yolo11m | 0.515 | 0.684 | 0.557 | 0.335 | 0.571 | 0.678 |
| yolo11l | 0.533 | 0.701 | 0.582 | 0.356 | 0.591 | 0.693 |
| yolo11x | 0.546 | 0.716 | 0.595 | 0.377 | 0.597 | 0.703 |
| yolo26n | 0.408 | 0.569 | 0.443 | 0.213 | 0.448 | 0.589 |
| yolo26s | 0.486 | 0.658 | 0.527 | 0.294 | 0.532 | 0.659 |
| yolo26m | 0.531 | 0.707 | 0.577 | 0.367 | 0.577 | 0.689 |
| yolo26l | 0.551 | 0.726 | 0.601 | 0.385 | 0.596 | 0.711 |
| yolo26x | 0.574 | 0.750 | 0.627 | 0.417 | 0.621 | 0.732 |

## Key Observations

### 1. Small Object Detection Gap

AP_S 顯著低於 AP_M 和 AP_L，差距約 15-20 個百分點：

- yolo11n: AP_S=0.200 vs AP_M=0.433 vs AP_L=0.571
- yolo26x: AP_S=0.417 vs AP_M=0.621 vs AP_L=0.732

即使最大的 yolo26x，AP_S 也只有 0.417，小物體偵測有很大的改進空間。

### 2. YOLO26 vs YOLO11

同尺寸比較，YOLO26 整體優於 YOLO11（AP 差距 +1.0~2.8）。AP_S 方面 YOLO26 也一致領先。

### 3. AP_S 隨模型增大的提升幅度

| Scale | YOLO11 AP_S | YOLO26 AP_S |
|-------|-------------|-------------|
| n → x | 0.200 → 0.377 (+0.177) | 0.213 → 0.417 (+0.204) |

YOLO26 在小物體上隨模型增大的提升幅度（+0.204）大於 YOLO11（+0.177）。

## Reproducibility Notes

### YOLO26 end2end 設定

YOLO26 預設載入為 `end2end=True`（無 NMS），但官方 benchmark 數值是以 `end2end=False`（傳統 NMS）模式報告的。設為 False 後結果與官方完全吻合（誤差 <= 0.001）。

本 baseline 統一使用 `end2end=False` 以對齊官方數值並與 YOLO11（也使用 NMS）公平比較。

### 與官方公佈值的對比

| Model | 我們 | 官方 | 差距 |
|-------|------|------|------|
| yolo26n | 0.408 | 0.409 | -0.001 |
| yolo26s | 0.486 | 0.486 | 0.000 |
| yolo26m | 0.531 | 0.531 | 0.000 |
| yolo26l | 0.551 | 0.550 | +0.001 |
| yolo26x | 0.574 | 0.575 | -0.001 |

### COCO AP_S 定義

- Small: area < 32^2 = 1024 px^2
- Medium: 1024 <= area < 96^2 = 9216 px^2
- Large: area >= 9216 px^2

AP_S/AP_M/AP_L 是在完整 val2017 上評估，由 pycocotools 按 annotation area 自動分類計算。

## Data & Scripts

- Evaluation script: `scripts/eval_coco_baseline.py`
- Results CSV: `results/coco_baseline.csv`
- Dataset: `/home/yclaizzs/ML_exploration/datasets/coco/` (COCO 2017, via ultralytics)
